//! Hy4-flash on one GPU: every layer's weights on the device, a latent cache a layer, a chunk of rows a step.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("cuda.zig");
const config = @import("config.zig");
const engram = @import("engram.zig");

const X = cuda.experts.Nvfp4Experts;
const Tensor = core.checkpoint.Tensor;
const Bufs = std.ArrayList(cuda.DeviceBuffer);

/// A latent cache row: the 512 normed latent values, then the 64 roped ones (bf16).
pub const row_width = 576;
const hc_eps: f32 = 1e-6;
const hc_magnitude: f32 = 2.0;
/// Engram's nn.RMSNorm(eps=None): the reference matches float32's eps (hy4-real-check).
const eps_engram: f32 = std.math.floatEps(f32);
const sparse_theta: f64 = 1e7;
const sliding_theta: f64 = 1e4;

/// The shared kernels the engine launches beside the family's own.
pub const Kernels = struct {
    k: hy4.Kernels,
    mla: cuda.mla.Mla,
    dense: cuda.dense_rows.DenseRows,
    mod: cuda.Module,
    router: cuda.grouped.Router,
    x: X,
    glue: cuda.moe_glue.MoeGlue,

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer mod.unload();
        return .{ .k = try hy4.Kernels.load(d), .mla = try cuda.mla.Mla.load(d), .dense = try cuda.dense_rows.DenseRows.load(d), .mod = mod, .router = try cuda.grouped.Router.resolve(mod), .x = try X.load(d, sms), .glue = try cuda.moe_glue.MoeGlue.load(d) };
    }

    pub fn unload(k: *Kernels) void {
        k.glue.unload();
        k.x.unload();
        k.mod.unload();
        k.dense.unload();
        k.mla.unload();
        k.k.unload();
    }
};

const Attn = struct { q_a: u64, q_a_norm: u64, q_b: u64, kv_a: u64, kv_a_norm: u64, wk: u64, wv: u64, sink: u64, gate: u64, o: u64, post_norm: u64, inv_freq: u64, heads: usize, sliding: bool };
const Moe = struct { router: u64, bias: u64, down: u64, up: u64, routed: cuda.experts.Layer, shared: cuda.experts.Layer, norm: u64 };
const Mlp = union(enum) { dense: struct { t: cuda.experts.Layer, norm: u64 }, moe: Moe };
const Engram = struct { ids: []align(1) const i32, offsets: u64, moduli: [engram.heads]u32, mult: [engram.max_ngram]u64, table: hy4.EngramTable, row_norm: u64, value: u64, value_bias: u64, key: [4]u64, key_bias: [4]u64, mix: hy4.EngramMix };
const Layer = struct { attn: Attn, hc_attn: hy4.Hc, hc_ffn: hy4.Hc, in_norm: u64, post_norm: u64, mlp: Mlp, engram: ?Engram };

/// Device buffers that live as long as the model, and the checkpoint they came from.
const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    bufs: *Bufs,
    rank: usize = 0,
    world: usize = 1,

    fn tensor(l: Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [200]u8 = undefined;
        return l.ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    }

    fn up(l: Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(l.d, bytes);
        try l.bufs.append(l.gpa, b);
        return b.ptr;
    }

    fn w(l: Loader, comptime fmt: []const u8, args: anytype) !u64 {
        return l.up((try l.tensor(fmt, args)).bytes);
    }

    /// bf16 tensors (or one) as fp32 on the device, one after the other.
    fn f32s(l: Loader, comptime fmt: []const u8, args: anytype, comptime indexed: bool, count: usize) !u64 {
        var all: std.ArrayList(f32) = .empty;
        defer all.deinit(l.gpa);
        for (0..count) |j| {
            const t = try l.tensor(fmt, if (indexed) args ++ .{j} else args);
            for (std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes))) |b| try all.append(l.gpa, @bitCast(@as(u32, b) << 16));
        }
        return l.up(std.mem.sliceAsBytes(all.items));
    }

    fn hc(l: Loader, comptime prefix: []const u8, args: anytype) !hy4.Hc {
        return .{ .fn_ = try l.w(prefix ++ "_fn", args), .base = try l.w(prefix ++ "_base", args), .scale = try l.w(prefix ++ "_scale", args) };
    }

    /// NVFP4 SwiGLU experts e0..e0+count (or one) packed as X lays them out, cut to intermediate rows lo..lo+ni.
    fn experts(l: Loader, comptime pre: []const u8, args: anytype, comptime single: bool, e0: usize, count: usize, total: usize, lo: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
        const dc = try l.gpa.alloc(u8, d * ni / 2);
        defer l.gpa.free(dc);
        const dsc = try l.gpa.alloc(u8, d * ni / 16);
        defer l.gpa.free(dsc);
        const up_len = X.bytes(1, ni, d, 2) / 4;
        const down_len = X.bytes(1, d, ni, 1) / 4;
        const ups = try l.gpa.alloc(u32, count * up_len);
        defer l.gpa.free(ups);
        const downs = try l.gpa.alloc(u32, count * down_len);
        defer l.gpa.free(downs);
        const us = try l.gpa.alloc(f32, count * 2);
        defer l.gpa.free(us);
        const ds = try l.gpa.alloc(f32, count);
        defer l.gpa.free(ds);
        for (0..count) |e| {
            var q: [3][2][]const u8 = undefined;
            var g: [3]f32 = undefined;
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const name = if (single) pre ++ proj ++ "." else pre ++ "{d}." ++ proj ++ ".";
                const xa = if (single) args else args ++ .{e0 + e};
                q[m][0] = (try l.tensor(name ++ "weight", xa)).bytes;
                q[m][1] = (try l.tensor(name ++ "weight_scale", xa)).bytes;
                g[m] = std.mem.bytesAsValue(f32, (try l.tensor(name ++ "weight_scale_2", xa)).bytes[0..4]).*;
            }
            if (ni != total) { // gate and up: whole rows; down: each row's columns lo..lo+ni
                for (0..2) |m| {
                    q[m][0] = q[m][0][lo * d / 2 ..][0 .. ni * d / 2];
                    q[m][1] = q[m][1][lo * d / 16 ..][0 .. ni * d / 16];
                }
                for (0..d) |r| {
                    @memcpy(dc[r * ni / 2 ..][0 .. ni / 2], q[2][0][r * total / 2 + lo / 2 ..][0 .. ni / 2]);
                    @memcpy(dsc[r * ni / 16 ..][0 .. ni / 16], q[2][1][r * total / 16 + lo / 16 ..][0 .. ni / 16]);
                }
                q[2] = .{ dc, dsc };
            }
            X.packGateUp(ups[e * up_len ..][0..up_len], q[0], q[1], ni, d);
            X.packOne(downs[e * down_len ..][0..down_len], q[2][0], q[2][1], d, ni);
            us[e * 2] = g[0];
            us[e * 2 + 1] = g[1];
            ds[e] = g[2];
        }
        return .{ .format = .nvfp4, .up = try l.up(std.mem.sliceAsBytes(ups)), .down = try l.up(std.mem.sliceAsBytes(downs)), .up_scale = try l.up(std.mem.sliceAsBytes(us)), .down_scale = try l.up(std.mem.sliceAsBytes(ds)), .width = ni, .dims = d, .experts = count, .limit = limit };
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const Kernels,
    c: config.Text,
    eg: config.Engram,
    layers: []Layer,
    embed: []const u8,
    norm: u64,
    head: hy4.Hc,
    lm: u64,
    weights: Bufs = .empty,
    scratch: Bufs = .empty,
    /// The latent cache of each layer: `context` rows of row_width bf16.
    cache: []u64,
    context: usize,
    /// The streams [4][hidden] bf16 of the rows at hand, and every token so far (Engram reads the two before).
    x: u64 = 0,
    tokens: std.ArrayList(i32) = .empty,
    /// Tensor parallel: this rank's heads, experts and Engram tables; partial sums meet in `group`.
    group: ?*const cuda.tp_group.Group = null,
    rank: usize = 0,
    world: usize = 1,

    fn sum(m: *Model, p: u64, count: usize) !void {
        if (m.group) |g| try g.allReduceSum(m.s, p, count);
    }

    /// Every layer of `c` from `ck` onto the device; `heads` the Engram head sizes by layer id (config.json).
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const Kernels, ck: *core.checkpoint.Checkpoint, cfg: config.Config, heads: []const [engram.heads]u32, context: usize, group: ?*const cuda.tp_group.Group) !Model {
        const c = cfg.text;
        const rank = if (group) |g| g.rank else 0;
        const world = if (group) |g| g.world else 1;
        if (c.sparse.q % world != 0 or c.sliding.q % world != 0 or c.experts % world != 0) return error.TensorParallelSplit;
        var m: Model = .{ .gpa = gpa, .d = d, .s = s, .k = k, .c = c, .eg = cfg.engram, .layers = try gpa.alloc(Layer, c.layers), .embed = undefined, .norm = 0, .head = undefined, .lm = 0, .cache = try gpa.alloc(u64, c.layers), .context = context, .group = group, .rank = rank, .world = world };
        errdefer m.close();
        const l: Loader = .{ .gpa = gpa, .d = d, .ck = ck, .bufs = &m.weights, .rank = rank, .world = world };
        m.embed = (try l.tensor("model.embed_tokens.weight", .{})).bytes;
        m.norm = try l.w("model.norm.weight", .{});
        m.head = try l.hc("model.hc_head.hc_head", .{});
        const lm = try l.tensor("lm_head.weight", .{});
        m.lm = if (lm.dtype == .bf16) try l.up(lm.bytes) else blk: {
            const h = try gpa.alloc(u16, lm.numel());
            defer gpa.free(h);
            for (h, 0..) |*o, j| o.* = bf16of(std.mem.bytesAsValue(f32, lm.bytes[j * 4 ..][0..4]).*);
            break :blk try l.up(std.mem.sliceAsBytes(h));
        };
        var engram_n: usize = 0;
        for (0..c.layers) |i| {
            const owner = c.engram[i] and engram_n % world == rank; // Engram tables a rank by turns
            m.layers[i] = try loadLayer(l, c, i, if (owner) heads[engram_n] else undefined, owner);
            engram_n += @intFromBool(c.engram[i]);
            var b = try cuda.DeviceBuffer.alloc(d, context * row_width * 2);
            try m.weights.append(gpa, b);
            try b.fill8(0, null);
            m.cache[i] = b.ptr;
            std.log.info("hy4: layer {d} of {d} on the GPU", .{ i + 1, c.layers });
        }
        return m;
    }

    pub fn close(m: *Model) void {
        m.reset();
        m.scratch.deinit(m.gpa);
        for (m.weights.items) |*b| b.free();
        m.weights.deinit(m.gpa);
        m.tokens.deinit(m.gpa);
        m.gpa.free(m.layers);
        m.gpa.free(m.cache);
    }

    fn reset(m: *Model) void {
        for (m.scratch.items) |*b| b.free();
        m.scratch.clearRetainingCapacity();
    }

    fn zeros(m: *Model, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, n);
        errdefer b.free();
        try b.fill8(0, m.s.handle);
        try m.scratch.append(m.gpa, b);
        return b.ptr;
    }

    fn up(m: *Model, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(m.d, bytes);
        try m.scratch.append(m.gpa, b);
        return b.ptr;
    }

    /// Rows `ids` at positions tokens.len.. through every layer: fp32 logits of every row (or the `last`) into `out`.
    pub fn step(m: *Model, ids: []const i32, last: bool, out: []f32) !void {
        defer m.reset();
        const c = m.c;
        const n = ids.len;
        const p0 = m.tokens.items.len;
        if (p0 + n > m.context) return error.PastContext;
        try m.tokens.appendSlice(m.gpa, ids);
        const hd: usize = c.hidden;
        const x0 = try m.gpa.alloc(u8, n * c.hc * hd * 2);
        defer m.gpa.free(x0);
        for (ids, 0..) |id, r| for (0..c.hc) |s| {
            @memcpy(x0[(r * c.hc + s) * hd * 2 ..][0 .. hd * 2], m.embed[@as(usize, @intCast(id)) * hd * 2 ..][0 .. hd * 2]);
        };
        m.x = try m.up(x0);
        const pos_h = [1]i32{@intCast(p0)};
        const pos = try m.up(std.mem.sliceAsBytes(&pos_h));
        var stage: []const u8 = "embed";
        var at: usize = 0;
        errdefer std.log.err("hy4 step: failed in layer {d}'s {s} ({d} rows at {d})", .{ at, stage, n, p0 });
        for (m.layers, 0..) |*L, i| {
            at = i;
            stage = "engram";
            if (c.engram[i]) try m.engramAdd(if (L.engram) |*e| e else null, p0, n);
            stage = "attention";
            const h = try m.zeros(n * hd * 2);
            const post = try m.zeros(n * c.hc * 4);
            try m.k.k.hcPre(m.s, m.x, L.hc_attn, h, post, n, hd, c.eps, hc_eps, hc_magnitude);
            const hn = try m.zeros(n * hd * 2);
            try m.k.k.rmsNorm16(m.s, h, hd, L.in_norm, hn, hd, n, hd, c.eps);
            try m.k.k.hcPost(m.s, m.x, try m.attend(&L.attn, m.cache[i], hn, pos, p0, n), post, n, hd);
            stage = "mlp";
            const h2 = try m.zeros(n * hd * 2);
            const post2 = try m.zeros(n * c.hc * 4);
            try m.k.k.hcPre(m.s, m.x, L.hc_ffn, h2, post2, n, hd, c.eps, hc_eps, hc_magnitude);
            const hn2 = try m.zeros(n * hd * 2);
            try m.k.k.rmsNorm16(m.s, h2, hd, L.post_norm, hn2, hd, n, hd, c.eps);
            try m.k.k.hcPost(m.s, m.x, try m.mlp(&L.mlp, hn2, n), post2, n, hd);
        }
        stage = "head";
        const rows = if (last) 1 else n;
        const xs = m.x + (n - rows) * c.hc * hd * 2;
        const h = try m.zeros(rows * hd * 2);
        try m.k.k.hcHead(m.s, xs, m.head, h, rows, hd, c.eps, hc_eps);
        const hn = try m.zeros(rows * hd * 2);
        try m.k.k.rmsNorm16(m.s, h, hd, m.norm, hn, hd, rows, hd, c.eps);
        const logits = try m.zeros(rows * c.vocab * 4);
        try m.k.dense.rows(m.s, hn, hd, m.lm, logits, c.vocab, rows, c.vocab, hd);
        try m.s.synchronize();
        try cuda.memory.download(m.d, logits, std.mem.sliceAsBytes(out[0 .. rows * c.vocab]));
    }

    /// Rows p0..: latent rows into `cache`, then MLA over it (sliding: the last `window` keys a row as a token list).
    fn attend(m: *Model, a: *const Attn, cache: u64, x: u64, pos: u64, p0: usize, n: usize) !u64 {
        const c = m.c;
        const hd: usize = c.hidden;
        const H = a.heads;
        const k = &m.k.k;
        const qr_n: usize = c.q_rank;
        const qa_f = try m.zeros(n * qr_n * 4);
        try m.k.dense.rows(m.s, x, hd, a.q_a, qa_f, qr_n, n, qr_n, hd);
        const qr = try m.zeros(n * qr_n * 2);
        try k.rmsNorm(m.s, qa_f, qr_n, a.q_a_norm, qr, qr_n, n, qr_n, c.eps);
        const q_f = try m.zeros(n * H * 256 * 4);
        try m.k.dense.rows(m.s, qr, qr_n, a.q_b, q_f, H * 256, n, H * 256, qr_n);
        const q = try m.zeros(n * H * 256 * 2);
        try k.castRows(m.s, q_f, H * 256, q, H * 256, n, H * 256);
        const qa = try m.zeros(n * H * row_width * 2);
        try m.k.mla.rope(m.s, .{ .x = q + 192 * 2, .x_row = H * 256, .x_head = 256, .out = qa + 512 * 2, .o_row = H * row_width, .o_head = row_width }, n, H, pos, a.inv_freq, 1.0);
        const kv_f = try m.zeros(n * row_width * 4);
        try m.k.dense.rows(m.s, x, hd, a.kv_a, kv_f, row_width, n, row_width, hd);
        const rows = cache + p0 * row_width * 2;
        try k.rmsNorm(m.s, kv_f, row_width, a.kv_a_norm, rows, row_width, n, 512, c.eps);
        try k.castRows(m.s, kv_f + 512 * 4, row_width, rows + 512 * 2, row_width, n, 64);
        try m.k.mla.rope(m.s, .{ .x = rows + 512 * 2, .x_row = row_width, .x_head = 0, .out = rows + 512 * 2, .o_row = row_width, .o_head = 0 }, n, 1, pos, a.inv_freq, 1.0);
        try k.absorbQ(m.s, q, a.wk, qa, n, H);
        const keys = p0 + n;
        const win: usize = c.window;
        const nch: usize = if (a.sliding) (win + 511) / 512 else (keys + 511) / 512;
        const part: cuda.mla.Partials = .{ .po = try m.zeros(nch * n * H * 512 * 4), .pm = try m.zeros(nch * n * H * 4), .pl = try m.zeros(nch * n * H * 4) };
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.sparse.nope + c.sparse.rope)));
        if (a.sliding) {
            const tok_h = try m.gpa.alloc(i32, n * win);
            defer m.gpa.free(tok_h);
            const cnt_h = try m.gpa.alloc(i32, n);
            defer m.gpa.free(cnt_h);
            @memset(tok_h, 0);
            for (0..n) |r| {
                const p = p0 + r;
                const first = if (p + 1 >= win) p + 1 - win else 0;
                cnt_h[r] = @intCast(p + 1 - first);
                for (first..p + 1, 0..) |key, j| tok_h[r * win + j] = @intCast(key);
            }
            try hy4.sparseChunks(&m.k.mla, m.s, qa, cache, try m.up(std.mem.sliceAsBytes(tok_h)), try m.up(std.mem.sliceAsBytes(cnt_h)), win, part, n, H, scale);
        } else try m.k.mla.chunks(m.s, qa, cache, pos, part, n, H, nch, scale);
        const ol = try m.zeros(n * H * 512 * 2);
        try k.mergeSink(m.s, part, a.sink, ol, n, H, nch);
        const o = try m.zeros(n * H * 256 * 2);
        try k.expandV(m.s, ol, a.wv, o, n, H);
        const g = try m.zeros(n * H * 256 * 4);
        try m.k.dense.rows(m.s, x, hd, a.gate, g, H * 256, n, H * 256, hd);
        try k.gateRows(m.s, o, g, n * H * 256);
        const proj = try m.zeros(n * hd * 4);
        try m.k.dense.rows(m.s, o, H * 256, a.o, proj, hd, n, hd, H * 256);
        try m.sum(proj, n * hd);
        const out = try m.zeros(n * hd * 2);
        try k.rmsNorm(m.s, proj, hd, a.post_norm, out, hd, n, hd, c.eps);
        return out;
    }

    /// x [n][d] bf16 through `t` on the picks (`per` a row) into bf16 [n][d] weighted by `wts`.
    fn grouped(m: *Model, t_: cuda.experts.Layer, x: u64, picks: u64, wts: u64, n: usize, per: usize, skip: i32) !u64 {
        var t = t_;
        if (skip >= 0) t.experts += 1; // the other ranks' picks, planned as one more expert and skipped
        const pairs = n * per;
        const tile = cuda.experts.promptTile(.nvfp4);
        const items = cuda.grouped.maxItems(pairs, t.experts, tile);
        const wide = pairs > cuda.grouped.small;
        const plan: cuda.grouped.Plan = .{ .members = try m.zeros(pairs * 4), .items = try m.zeros(items * 12), .counts = try m.zeros(8), .rank = try m.zeros((if (wide) pairs else 1) * 4), .hist = try m.zeros((if (wide) (pairs + 1023) / 1024 * t.experts else 1) * 4) };
        try m.k.router.route(m.s, picks, pairs, t.experts, tile, plan);
        const act = try m.zeros(pairs * t.width * 2);
        const y = try m.zeros(pairs * t.dims * 2);
        const g: cuda.experts.Grouped = .{ .nvfp4 = &m.k.x };
        try g.prompt(m.s, .up, .{ .x = x, .stride = t.dims, .slots = per }, t, plan, act, items, pairs, tile, skip);
        try g.prompt(m.s, .down_bf16, .{ .x = act, .stride = t.width }, t, plan, y, items, pairs, tile, skip);
        const out = try m.zeros(n * t.dims * 2);
        try m.k.glue.combine(m.s, y, .bf16, wts, out, .bf16, n, per, t.dims);
        return out;
    }

    /// One expert for every row at weight 1 (the shared expert, the dense MLP).
    fn one(m: *Model, t: cuda.experts.Layer, x: u64, n: usize) !u64 {
        const ones = try m.gpa.alloc(f32, n);
        defer m.gpa.free(ones);
        @memset(ones, 1);
        return m.grouped(t, x, try m.zeros(n * 4), try m.up(std.mem.sliceAsBytes(ones)), n, 1, -1);
    }

    fn mlp(m: *Model, p: *const Mlp, x: u64, n: usize) !u64 {
        const c = m.c;
        const hd: usize = c.hidden;
        const out = try m.zeros(n * hd * 2);
        switch (p.*) {
            .dense => |dm| {
                const y = try m.zeros(n * hd * 4);
                try m.k.k.addb(m.s, y, try m.one(dm.t, x, n), n * hd);
                try m.sum(y, n * hd);
                try m.k.k.rmsNorm(m.s, y, hd, dm.norm, out, hd, n, hd, c.eps);
            },
            .moe => |e| {
                const E: usize = c.experts;
                const K: usize = c.topk;
                const lat: usize = c.latent;
                const logits = try m.zeros(n * E * 4);
                try m.k.dense.rows(m.s, x, hd, e.router, logits, E, n, E, hd);
                const picks = try m.zeros(n * K * 4);
                const wts = try m.zeros(n * K * 4);
                try m.k.glue.topk(m.s, logits, e.bias, .{ .experts = E, .topk = K, .shared = false, .normalize = true, .scale = c.route_scale, .score = true }, picks, wts, n);
                const xl_f = try m.zeros(n * lat * 4);
                try m.k.dense.rows(m.s, x, hd, e.down, xl_f, lat, n, lat, hd);
                const xl = try m.zeros(n * lat * 2);
                try m.k.k.castRows(m.s, xl_f, lat, xl, lat, n, lat);
                const own = e.routed.experts;
                if (m.world > 1) try m.k.k.own(m.s, picks, wts, n * K, m.rank * own, own);
                const combined = try m.grouped(e.routed, xl, picks, wts, n, K, if (m.world > 1) @intCast(own) else -1);
                const routed = try m.zeros(n * hd * 4);
                try m.k.dense.rows(m.s, combined, lat, e.up, routed, hd, n, hd, lat);
                try m.k.k.addb(m.s, routed, try m.one(e.shared, x, n), n * hd);
                try m.sum(routed, n * hd);
                try m.k.k.rmsNorm(m.s, routed, hd, e.norm, out, hd, n, hd, c.eps);
            },
        }
        return out;
    }

    /// Engram's residual into the streams of rows p0..: each row's n-grams from the tokens before it.
    fn engramAdd(m: *Model, e_: ?*const Engram, p0: usize, n: usize) !void {
        const hd: usize = m.c.hidden;
        const S: usize = m.c.hc;
        const res = try m.zeros(n * S * hd * 4);
        if (e_) |e| try m.engramResidual(e, p0, n, res);
        try m.sum(res, n * S * hd);
        try m.k.k.add16(m.s, m.x, res, n * S * hd);
    }

    /// Engram's residual [n][S][hidden] fp32 of rows p0.. on the rank that holds the layer's table.
    fn engramResidual(m: *Model, e: *const Engram, p0: usize, n: usize, res: u64) !void {
        const hd: usize = m.c.hidden;
        const S: usize = m.c.hc;
        const hash = try m.gpa.alloc(i32, n * engram.heads);
        defer m.gpa.free(hash);
        const toks = m.tokens.items;
        for (0..n) |r| {
            const p = p0 + r;
            var comp: [engram.max_ngram]u32 = undefined;
            for (0..engram.max_ngram) |o| {
                const tok: i64 = if (p >= o) toks[p - o] else @as(i64, m.eg.pad);
                const look: usize = if (tok >= 0 and tok < @as(i64, @intCast(e.ids.len))) @intCast(tok) else m.eg.pad;
                comp[o] = @intCast(e.ids[look]);
            }
            var out: [engram.heads]u32 = undefined;
            engram.hash(comp, e.mult, &e.moduli, &out);
            for (out, 0..) |v, h| hash[r * engram.heads + h] = @intCast(v);
        }
        const rows = try m.zeros(n * engram.heads * 96 * 2);
        try m.k.k.engramRows(m.s, try m.up(std.mem.sliceAsBytes(hash)), e.offsets, e.table, e.row_norm, eps_engram, rows, n);
        const ew = engram.heads * 96;
        const value = try m.zeros(n * hd * 4);
        try m.k.dense.rows(m.s, rows, ew, e.value, value, hd, n, hd, ew);
        try m.k.k.bias(m.s, value, hd, e.value_bias, n, hd);
        const key = try m.zeros(n * S * hd * 4);
        for (0..S) |s| {
            try m.k.dense.rows(m.s, rows, ew, e.key[s], key + s * hd * 4, S * hd, n, hd, ew);
            try m.k.k.bias(m.s, key + s * hd * 4, S * hd, e.key_bias[s], n, hd);
        }
        try m.k.k.engramMix(m.s, key, value, m.x, e.mix, eps_engram, res, n, hd);
    }
};

fn bf16of(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn loadLayer(l: Loader, c: config.Text, i: usize, heads: [engram.heads]u32, engram_here: bool) !Layer {
    const p = "model.layers.{d}.";
    const sliding = c.kinds[i] == .sliding;
    const H_all: usize = c.heads(@intCast(i)).q;
    const H = H_all / l.world;
    const h0 = l.rank * H; // this rank's heads h0..h0+H
    const freqs = cuda.mla.ropeFreqs(if (sliding) sliding_theta else sparse_theta);
    // kv_b [H][192 + 256][512] split into wk [H][192][512] and wv [H][256][512]
    const kvb = (try l.tensor(p ++ "self_attn.kv_b_proj.weight", .{i})).bytes;
    const wk = try l.gpa.alloc(u8, H * 192 * 1024);
    defer l.gpa.free(wk);
    const wv = try l.gpa.alloc(u8, H * 256 * 1024);
    defer l.gpa.free(wv);
    for (0..H) |h| {
        @memcpy(wk[h * 192 * 1024 ..][0 .. 192 * 1024], kvb[(h0 + h) * 448 * 1024 ..][0 .. 192 * 1024]);
        @memcpy(wv[h * 256 * 1024 ..][0 .. 256 * 1024], kvb[((h0 + h) * 448 + 192) * 1024 ..][0 .. 256 * 1024]);
    }
    const a = p ++ "self_attn.";
    const hd: usize = c.hidden;
    const rows = H * 256 * 2; // a head slice's bytes a hidden column of q_b's and the gate's outputs
    const o_all = (try l.tensor(a ++ "o_proj.weight", .{i})).bytes;
    const o = try l.gpa.alloc(u8, hd * rows);
    defer l.gpa.free(o);
    for (0..hd) |r| @memcpy(o[r * rows ..][0..rows], o_all[r * H_all * 512 + h0 * 512 ..][0..rows]);
    const sink = (try l.tensor(a ++ "learnable_sink_param", .{i})).bytes[h0 * 4 ..][0 .. H * 4];
    const attn: Attn = .{ .q_a = try l.w(a ++ "q_a_proj.weight", .{i}), .q_a_norm = try l.w(a ++ "q_a_layernorm.weight", .{i}), .q_b = try l.up((try l.tensor(a ++ "q_b_proj.weight", .{i})).bytes[h0 * 256 * c.q_rank * 2 ..][0 .. H * 256 * c.q_rank * 2]), .kv_a = try l.w(a ++ "kv_a_proj_with_mqa.weight", .{i}), .kv_a_norm = try l.w(a ++ "kv_a_layernorm.weight", .{i}), .wk = try l.up(wk), .wv = try l.up(wv), .sink = try l.up(sink), .gate = try l.up((try l.tensor(a ++ "linear_gate.weight", .{i})).bytes[h0 * 256 * hd * 2 ..][0 .. H * 256 * hd * 2]), .o = try l.up(o), .post_norm = try l.w(a ++ "post_proj_norm.weight", .{i}), .inv_freq = try l.up(std.mem.sliceAsBytes(&freqs)), .heads = H, .sliding = sliding };
    const mp = p ++ "mlp.";
    const di = c.dense_inter / l.world;
    const si = c.expert_inter * c.shared / l.world;
    const eo = c.experts / l.world;
    const mlp_: Mlp = if (!c.moe[i]) .{ .dense = .{ .t = try l.experts(mp, .{i}, true, 0, 1, c.dense_inter, l.rank * di, di, hd, 0), .norm = try l.w(mp ++ "post_fc2_norm.weight", .{i}) } } else .{ .moe = .{
        .router = try l.w(mp ++ "gate.weight", .{i}),
        .bias = try l.w(mp ++ "gate.e_score_correction_bias", .{i}),
        .down = try l.w(mp ++ "routed_expert_down_proj.weight", .{i}),
        .up = try l.w(mp ++ "routed_expert_up_proj.weight", .{i}),
        .routed = try l.experts(mp ++ "experts.", .{i}, false, l.rank * eo, eo, c.expert_inter, 0, c.expert_inter, c.latent, c.swiglu_limit),
        .shared = try l.experts(mp ++ "shared_experts.", .{i}, true, 0, 1, c.expert_inter * c.shared, l.rank * si, si, hd, 0),
        .norm = try l.w(mp ++ "combine_output_norm.weight", .{i}),
    } };
    return .{ .attn = attn, .hc_attn = try l.hc(p ++ "hc_attn_layer.hc_pre.hc", .{i}), .hc_ffn = try l.hc(p ++ "hc_mlp_layer.hc_pre.hc", .{i}), .in_norm = try l.w(p ++ "input_layernorm.weight", .{i}), .post_norm = try l.w(p ++ "post_attention_layernorm.weight", .{i}), .mlp = mlp_, .engram = if (engram_here) try loadEngram(l, i, heads) else null };
}

fn loadEngram(l: Loader, i: usize, moduli: [engram.heads]u32) !Engram {
    const p = "model.layers.{d}.engram_layer.";
    var offsets: [engram.heads]i64 = undefined;
    var off: i64 = 0;
    for (moduli, 0..) |md, h| {
        offsets[h] = off;
        off += md;
    }
    const tw = try l.tensor(p ++ "multi_head_embedding.embedding.weight", .{i});
    const table: hy4.EngramTable = if (tw.dtype == .f8_e4m3) .{ .fp8 = try l.up(tw.bytes), .scales = try l.w(p ++ "multi_head_embedding.embedding.weight_scale_inv", .{i}) } else .{ .bf16 = try l.up(tw.bytes) };
    var e: Engram = .{
        .ids = std.mem.bytesAsSlice(i32, (try l.tensor(p ++ "lookup_table", .{i})).bytes),
        .offsets = try l.up(std.mem.sliceAsBytes(&offsets)),
        .moduli = moduli,
        .mult = engram.multipliers(@intCast(i + 1)) orelse return error.NoEngramMultipliers,
        .table = table,
        .row_norm = try l.f32s(p ++ "lookup_per_head_norms.{d}.weight", .{i}, true, engram.heads),
        .value = try l.w(p ++ "value_proj.weight", .{i}),
        .value_bias = try l.w(p ++ "value_proj.bias", .{i}),
        .key = undefined,
        .key_bias = undefined,
        .mix = .{ .norm_key = try l.f32s(p ++ "norm_key.{d}.weight", .{i}, true, 4), .norm_query = try l.f32s(p ++ "norm_query.{d}.weight", .{i}, true, 4), .gamma = try l.f32s(p ++ "unified_pre_gamma", .{i}, false, 1), .hc_scale = try l.f32s(p ++ "unified_per_hc_scales.{d}", .{i}, true, 4) },
    };
    for (0..4) |s| {
        e.key[s] = try l.w(p ++ "key_projs.{d}.weight", .{ i, s });
        e.key_bias[s] = try l.w(p ++ "key_projs.{d}.bias", .{ i, s });
    }
    return e;
}
