//! Hy4-flash behind lanes.Backend: one stream on the engine's one sequence; rank 0 sends each call to the others.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const dfly = @import("dfly.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Sampling = lanes.sampling.Sampling;
const Model = engine.Model;

/// The widest window a round verifies: the last token and DFly's drafts (one graph replays it).
pub const max_window = engine.verify_rows;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;
const no_row = std.math.maxInt(u32);

/// The calls rank 0 makes on the model, in its order: the other ranks make each too (their collectives meet).
pub const Kind = enum(u32) { clear = 1, step, keep, propose, end };

/// A call: step (n ids, a last, b logits, c kept row), keep (a length, b row), propose (a token).
pub const Header = extern struct { kind: u32, n: u32 = 0, a: u32 = 0, b: u32 = 0, c: u32 = no_row, pad: [3]u32 = @splat(0) };

/// Rank 0's calls to the other ranks over NCCL: a header, then a step's ids.
pub const Link = struct {
    g: *const cuda.tp_group.Group,
    d: *const cuda.Driver,
    buf: cuda.DeviceBuffer,
    ids: []i32,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *const cuda.tp_group.Group, context: usize) !Link {
        var buf = try cuda.DeviceBuffer.alloc(d, @sizeOf(Header) + context * 4);
        errdefer buf.free();
        return .{ .g = g, .d = d, .buf = buf, .ids = try gpa.alloc(i32, context) };
    }

    pub fn deinit(l: *Link, gpa: std.mem.Allocator) void {
        l.buf.free();
        gpa.free(l.ids);
    }

    /// Rank 0: the call to every other rank, before it runs it; the buffer is free again on return.
    pub fn send(l: *Link, s: cuda.Stream, h: Header, ids: []const i32) !void {
        try l.put(s, 0, std.mem.asBytes(&h));
        if (ids.len > 0) try l.put(s, @sizeOf(Header), std.mem.sliceAsBytes(ids));
        for (1..l.g.world) |peer| {
            try l.g.send(s, l.buf.ptr, @sizeOf(Header), peer);
            if (ids.len > 0) try l.g.send(s, l.buf.ptr + @sizeOf(Header), ids.len * 4, peer);
        }
        try s.synchronize();
    }

    /// Host bytes into the buffer on `s`, ordered before the sends that read them.
    fn put(l: *Link, s: cuda.Stream, at: usize, bytes: []const u8) !void {
        try l.d.check(l.d.api.cuMemcpyHtoDAsync_v2(l.buf.ptr + at, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    /// Another rank: rank 0's next call and its ids.
    pub fn receive(l: *Link, s: cuda.Stream) !struct { h: Header, ids: []const i32 } {
        var h: Header = undefined;
        try l.g.recv(s, l.buf.ptr, @sizeOf(Header), 0);
        try s.synchronize();
        try l.buf.download(0, std.mem.asBytes(&h));
        if (h.kind < @intFromEnum(Kind.clear) or h.kind > @intFromEnum(Kind.end)) return error.BadCall;
        if (h.n > l.ids.len) return error.CallTooWide;
        if (h.n > 0) {
            try l.g.recv(s, l.buf.ptr + @sizeOf(Header), h.n * 4, 0);
            try s.synchronize();
            try l.buf.download(@sizeOf(Header), std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};

/// The model's calls on every rank: rank 0 sends each one first.
const Calls = struct {
    m: *Model,
    draft: ?*dfly.Draft,
    link: ?*Link,

    fn tell(c: Calls, h: Header, ids: []const i32) !void {
        if (c.link) |l| if (c.m.rank == 0) try l.send(c.m.s, h, ids);
    }

    fn clear(c: Calls) !void {
        try c.tell(.{ .kind = @intFromEnum(Kind.clear) }, &.{});
        c.m.tokens.clearRetainingCapacity();
    }

    /// One step; `keep`: then its row the drafter reads (DFly's anchor).
    fn step(c: Calls, ids: []const i32, last: bool, out: []i32, logits: ?[]f32, kept: ?u32) !void {
        try c.tell(.{ .kind = @intFromEnum(Kind.step), .n = @intCast(ids.len), .a = @intFromBool(last), .b = @intFromBool(logits != null), .c = kept orelse no_row }, ids);
        try c.m.step(ids, last, out, logits);
        if (kept) |r| if (c.draft != null) try c.m.keepRow(r);
    }

    /// The cache back to `len` positions, row `row` of the last step the drafter's anchor.
    fn keep(c: Calls, len: usize, row: u32) !void {
        try c.tell(.{ .kind = @intFromEnum(Kind.keep), .a = @intCast(len), .b = row }, &.{});
        c.m.tokens.shrinkRetainingCapacity(len);
        if (c.draft != null) try c.m.keepRow(row);
    }

    fn propose(c: Calls, t: i32, out: *[dfly.drafts]i32) !void {
        try c.tell(.{ .kind = @intFromEnum(Kind.propose), .a = @bitCast(t) }, &.{});
        try c.draft.?.propose(c.m, t, out);
    }
};

/// A rank above 0: rank 0's calls until it ends.
pub fn follow(gpa: std.mem.Allocator, m: *Model, draft: ?*dfly.Draft, link: *Link) !void {
    const logits = try gpa.alloc(f32, max_window * m.c.vocab);
    defer gpa.free(logits);
    const c: Calls = .{ .m = m, .draft = draft, .link = link };
    while (true) {
        const call = try link.receive(m.s);
        const h = call.h;
        switch (@as(Kind, @enumFromInt(h.kind))) {
            .clear => try c.clear(),
            .step => {
                var out: [max_window]i32 = undefined;
                const rows = if (h.a != 0) 1 else call.ids.len;
                try c.step(call.ids, h.a != 0, &out, if (h.b != 0) logits[0 .. rows * m.c.vocab] else null, if (h.c == no_row) null else h.c);
            },
            .keep => try c.keep(h.a, h.b),
            .propose => {
                var out: [dfly.drafts]i32 = undefined;
                try c.propose(@bitCast(h.a), &out);
            },
            .end => return,
        }
    }
}

pub const Hy4 = struct {
    gpa: std.mem.Allocator,
    calls: Calls,
    /// The one stream the sequence holds (null: free).
    stream: ?*const Stream = null,
    sampling: ?Sampling = null,
    /// A sampled stream's last step's logits [rows][vocab], and `choose`'s inputs for a row.
    logits: []f32,
    values: []f64,
    ids: []u64,
    /// The step's argmax tokens (greedy draws), the prompt's ids and the drawn tokens' ring.
    out: [max_window]i32 = undefined,
    prompt: std.ArrayList(i32) = .empty,
    toks: [ring]u32 = undefined,
    next: u64 = 0,
    /// The cache length the last verify started from and its rows.
    base: usize = 0,
    rows: [max_window]i32 = undefined,
    n: usize = 0,
    /// DFly's drafts held for the next verify.
    held: [dfly.drafts]i32 = undefined,

    pub fn init(gpa: std.mem.Allocator, m: *Model, head: ?*dfly.Draft, link: ?*Link) !Hy4 {
        const v = m.c.vocab;
        const logits = try gpa.alloc(f32, max_window * v);
        errdefer gpa.free(logits);
        const values = try gpa.alloc(f64, v);
        errdefer gpa.free(values);
        const ids = try gpa.alloc(u64, v);
        for (ids, 0..) |*o, i| o.* = i;
        return .{ .gpa = gpa, .calls = .{ .m = m, .draft = head, .link = link }, .logits = logits, .values = values, .ids = ids };
    }

    pub fn deinit(x: *Hy4) void {
        if (x.calls.link != null and x.calls.m.rank == 0) x.calls.tell(.{ .kind = @intFromEnum(Kind.end) }, &.{}) catch {};
        x.prompt.deinit(x.gpa);
        x.gpa.free(x.logits);
        x.gpa.free(x.values);
        x.gpa.free(x.ids);
    }

    pub fn backend(x: *Hy4) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release };

    /// The round loop's facts: one stream; DFly holds 5 drafts a round when loaded.
    pub fn facts(x: *const Hy4) lanes.Model {
        const head = x.calls.draft != null;
        return .{ .exact_width = 1, .gpu_tokens = false, .mtp = head, .speculate = false, .speculate_early = false, .drafts = if (head) dfly.drafts else 1, .draft_prior = if (head) &.{ 0.85, 0.7, 0.55, 0.45, 0.35 } else &.{}, .max_streams = 1 };
    }

    fn self(ptr: *anyopaque) *Hy4 {
        return @ptrCast(@alignCast(ptr));
    }

    fn vocab(x: *const Hy4) usize {
        return x.calls.m.c.vocab;
    }

    /// The logits a step leaves for the host when the stream samples (null: greedy, the step's argmax).
    fn logitsFor(x: *Hy4, rows: usize) ?[]f32 {
        return if (x.sampling != null) x.logits[0 .. rows * x.vocab()] else null;
    }

    /// Row `r`'s token drawn at `position`: the step's argmax, or the stream's sampling over the row's logits.
    fn draw(x: *Hy4, r: usize, position: u64) !u32 {
        const s = x.sampling orelse return @intCast(x.out[r]);
        const v = x.vocab();
        for (x.values, x.logits[r * v ..][0..v]) |*o, l| o.* = l;
        return @intCast(try lanes.sampling.choose(x.gpa, x.values, x.ids, position, s));
    }

    fn hold(x: *Hy4, t: u32) u64 {
        const h = x.next;
        x.toks[h % ring] = t;
        x.next += 1;
        return h;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.stream != null) return error.NoFreeSlot;
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        const m = x.calls.m;
        if (p.len + s.max_new + max_window > m.context) return error.PastContext;
        x.sampling = if (s.sampling) |sm| (if (sm.temperature > 0) sm else null) else null;
        x.prompt.clearRetainingCapacity();
        for (p) |t| try x.prompt.append(x.gpa, @intCast(t));
        try x.calls.clear();
        try x.calls.step(x.prompt.items, true, &x.out, x.logitsFor(1), 0);
        x.stream = s;
        s.cached = 0;
        s.fill_at.store(@intCast(p.len), .monotonic);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        if (x.stream != s) return error.UnknownStream;
        if (position != x.calls.m.tokens.items.len) return error.PositionMismatch;
        return x.hold(try x.draw(0, position));
    }

    fn value(x: *Hy4, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        if (x.stream != s) return error.UnknownStream;
        const t: i32 = @intCast(try x.value(feed));
        if (position != x.calls.m.tokens.items.len + 1) return error.PositionMismatch;
        try x.calls.step(&.{t}, true, &x.out, x.logitsFor(1), 0);
        return x.hold(try x.draw(0, position));
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        return x.toks[h % ring];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        if (windows.len != 1) return error.OneStream;
        const w = windows[0];
        if (x.stream != w.stream) return error.UnknownStream;
        if (w.parents != null) return error.ChainsOnly;
        if (w.masks.len != 0) return error.StructuresUnsupported;
        if (w.rows() > max_window) return error.WindowTooWide;
        x.base = x.calls.m.tokens.items.len;
        x.n = w.rows();
        x.rows[0] = @intCast(w.pending);
        if (w.held > 0) @memcpy(x.rows[1..x.n], x.held[0..w.held]);
        for (w.tokens, 1..) |t, r| x.rows[r] = @intCast(t);
        for (w.positions, 0..) |pos, r| if (pos != x.base + r + 1) return error.PositionMismatch;
        try x.calls.step(x.rows[0..x.n], false, &x.out, x.logitsFor(x.n), null);
        for (0..x.n) |r| out[0].sampled[r] = try x.draw(r, w.positions[r]);
        for (x.rows[1..x.n], 0..) |t, r| out[0].drafts[r] = @intCast(t);
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        const path = paths[0];
        if (windows.len != 1 or path.len == 0 or path.len > x.n) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.ChainsOnly; // a chain keeps a prefix of its rows
        try x.calls.keep(x.base + path.len, @intCast(path.len - 1));
    }

    /// DFly: `depth` drafts after the stream's pending token, from the kept row's hidden state; depth 0 absorbs.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            if (r.depth == 0 and r.lanes == null) continue;
            if (x.calls.draft == null) return error.NoDraftHead;
            if (r.lanes != null or r.depth > dfly.drafts) return error.ChainsOnly;
            const t: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else if (r.first) |f| try x.value(f) else return error.NoPendingToken;
            try x.calls.propose(@intCast(t), &x.held);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.stream == s) x.stream = null;
    }
};
