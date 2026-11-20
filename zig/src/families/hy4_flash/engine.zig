//! Hy4-flash on one GPU: every layer's weights on the device, a latent cache a layer, a chunk of rows a step.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("cuda.zig");
const config = @import("config.zig");
const engram = @import("engram.zig");

const X = cuda.experts.Nvfp4Experts;

const Bufs = std.ArrayList(cuda.DeviceBuffer);

/// A latent cache row: the 512 normed latent values, then the 64 roped ones (bf16).
pub const row_width = 576;
/// The rows a step takes at most when only its last row's token is wanted (longer prompts go in chunks).
pub const max_step_rows = 2048;
/// A verify block's rows: the last token and DFly's drafts.
pub const verify_rows = 6;
const hc_eps: f32 = 1e-6;
const hc_magnitude: f32 = 2.0;
/// Engram's nn.RMSNorm(eps=None): the reference matches float32's eps (hy4-real-check).
const eps_engram: f32 = std.math.floatEps(f32);

/// The shared kernels the engine launches beside the family's own.
pub const Kernels = struct {
    k: hy4.Kernels,
    mla: cuda.mla.Mla,
    dense: cuda.dense_rows.DenseRows,
    mod: cuda.Module,
    router: cuda.grouped.Router,
    x: X,
    glue: cuda.moe_glue.MoeGlue,
    lane: cuda.qmmf.Lane,
    packer: cuda.modelopt.Packer,
    embed: cuda.embed_rows.EmbedRows,
    /// GLM's top-512 pool selection (radix select, ties to the lower index), for the indexer's groups.
    sel_mod: cuda.Module,
    select: cuda.Function,
    select_wide: cuda.Function,

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer mod.unload();
        var sel_mod = try cuda.Module.load(d, cuda.kernels.glm_glue);
        errdefer sel_mod.unload();
        var lane = try cuda.qmmf.Lane.load(d, 12);
        errdefer lane.unload();
        try lane.loadPrompt(d);
        return .{ .k = try hy4.Kernels.load(d), .mla = try cuda.mla.Mla.load(d), .dense = try cuda.dense_rows.DenseRows.load(d), .mod = mod, .router = try cuda.grouped.Router.resolve(mod), .x = try X.load(d, sms), .glue = try cuda.moe_glue.MoeGlue.load(d), .lane = lane, .packer = try cuda.modelopt.Packer.load(d), .embed = try cuda.embed_rows.EmbedRows.load(d), .sel_mod = sel_mod, .select = try sel_mod.function("tf_glm_select_rows"), .select_wide = try sel_mod.function("tf_glm_select_rows_wide") };
    }

    pub fn unload(k: *Kernels) void {
        k.sel_mod.unload();
        k.embed.unload();
        k.packer.unload();
        k.lane.unload();
        k.glue.unload();
        k.x.unload();
        k.mod.unload();
        k.dense.unload();
        k.mla.unload();
        k.k.unload();
    }
};

const wt = @import("weights.zig");
const index = @import("index.zig");
const per_head = @import("heads.zig");
pub const Lin = wt.Lin;
pub const Attn = wt.Attn;
pub const Loader = wt.Loader;
pub const loadAttn = wt.loadAttn;
const Layer = wt.Layer;
const Mlp = wt.Mlp;
const Engram = wt.Engram;
const loadLayer = wt.loadLayer;
const bf16of = wt.bf16of;

/// The snapshot DFly reads: target layers, their token lists [layers][6][2048] and counts, the kept slot, validity.
pub const Snap = struct { layers: [8]usize, n: usize, tok: u64, cnt: u64, slot: u64, keys: usize = 0, kept: usize = 0 };

/// Graphs a chunk count holds: a decode row, a verify block, a DFly proposal (dense, or over the target's keys).
pub const graph_kinds = 4;
/// A captured step and where it leaves its result (logits, or DFly's draft ids) and its final-normed rows.
pub const Graph = struct { exec: cuda.graph.Exec, logits: u64, hn: u64, local: u64 };

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
    lm: Lin,
    weights: Bufs = .empty,
    /// A step's buffers: blocks kept across steps, handed out from `used` (block, bytes) and taken back at reset.
    scratch: Bufs = .empty,
    used: struct { block: usize = 0, at: usize = 0 } = .{},
    /// With `prof`: each part's synchronized time over the steps so far (ns): engram, attention, mlp, head.
    prof: ?std.Io = null,
    times: [8]i96 = @splat(0),
    /// The latent cache of each layer: `context` rows of row_width bf16.
    cache: []u64,
    context: usize,
    /// The streams [4][hidden] bf16 of the rows at hand, and every token so far (Engram reads the two before).
    x: u64 = 0,
    tokens: std.ArrayList(i32) = .empty,
    /// A step's inputs on the device: its streams' rows, its first position, 1.0 a row (the one-expert weights).
    pos: u64 = 0,
    /// pos[0] + 3: the indexer's selection then sees each row's own group, complete or not.
    pos3: u64 = 0,
    /// The embedding table on the device and the step's token ids (the streams start as copies of their rows).
    table: u64 = 0,
    ids: u64 = 0,
    /// The last step's final-normed rows (before the head) and a kept copy of one of them (DFly's target hidden).
    hn: u64 = 0,
    anchor: u64 = 0,
    /// The last step's logits of this rank's vocabulary slice [rows][vocab / world] (checks gather them).
    local: u64 = 0,
    ones: u64 = 0,
    /// Decode steps replayed as CUDA graphs, one a count of the full layers' 512-key chunks (null: not captured).
    graphs: []?Graph = &.{},
    use_graphs: bool = true,
    /// false: sparse layers attend densely past the recall budget too (checks the indexer against it).
    use_index: bool = true,
    /// DFly's view of the target's indexer: the last step's recalled lists for its draft layers' target layers.
    snap: ?Snap = null,
    /// A check's copy of one layer's recalled groups ([rows][512] int64 at `probe`; 0: none).
    probe: u64 = 0,
    probe_layer: usize = 0,
    at_layer: usize = 0,
    /// Tensor parallel: this rank's heads, experts and Engram tables; partial sums meet in `group`.
    group: ?*const cuda.tp_group.Group = null,
    rank: usize = 0,
    world: usize = 1,

    pub fn sum(m: *Model, p: u64, count: usize) !void {
        const g = m.group orelse return;
        const f = g.fast orelse return g.allReduceSum(m.s, p, count); // two ranks: RoCE writes (openRoce) where it fits
        if (g.world != 2 or !f.fits(count * 4)) return g.allReduceSum(m.s, p, count);
        const both = try m.buf(2 * count * 4);
        try g.allGather(m.s, p, both, count * 4);
        try m.k.k.add32(m.s, p, both + (1 - m.rank) * count * 4, count); // the other rank's partial: a + b == b + a
    }

    /// Every layer of `c` from `ck` onto the device; `heads` the Engram head sizes by layer id (config.json).
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const Kernels, ck: *core.checkpoint.Checkpoint, cfg: config.Config, heads: []const [engram.heads]u32, context: usize, group: ?*const cuda.tp_group.Group) !Model {
        const c = cfg.text;
        const rank = if (group) |g| g.rank else 0;
        const world = if (group) |g| g.world else 1;
        if (c.sparse.q % world != 0 or c.sliding.q % world != 0 or c.experts % world != 0) return error.TensorParallelSplit;
        var m: Model = .{ .gpa = gpa, .d = d, .s = s, .k = k, .c = c, .eg = cfg.engram, .layers = try gpa.alloc(Layer, c.layers), .embed = undefined, .norm = 0, .head = undefined, .lm = undefined, .cache = try gpa.alloc(u64, c.layers), .context = context, .group = group, .rank = rank, .world = world };
        errdefer m.close();
        const l: Loader = .{ .gpa = gpa, .d = d, .ck = ck, .bufs = &m.weights, .rank = rank, .world = world, .s = s, .packer = &k.packer, .context = context };
        m.embed = (try l.tensor("model.embed_tokens.weight", .{})).bytes;
        m.norm = try l.w("model.norm.weight", .{});
        m.head = try l.hc("model.hc_head.hc_head", .{});
        // the head's rows by vocabulary slice: each rank's logits, then its argmax pairs gathered
        const lm = try l.tensor("lm_head.weight", .{});
        if (c.vocab % world != 0) return error.TensorParallelSplit;
        const vl = c.vocab / world;
        const per: usize = if (lm.dtype == .bf16) 2 else 4;
        const mine = lm.bytes[rank * vl * c.hidden * per ..][0 .. vl * c.hidden * per];
        m.lm = if (lm.dtype == .bf16) try l.lin(mine, vl, c.hidden) else blk: {
            const h = try gpa.alloc(u16, vl * c.hidden);
            defer gpa.free(h);
            for (h, 0..) |*o, j| o.* = bf16of(std.mem.bytesAsValue(f32, mine[j * 4 ..][0..4]).*);
            break :blk try l.lin(std.mem.sliceAsBytes(h), vl, c.hidden);
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
            ck.release(); // the layer's file pages are on the device now: give them back
            std.log.info("hy4: layer {d} of {d} on the GPU", .{ i + 1, c.layers });
        }
        m.table = try l.up(m.embed);
        m.ids = try l.zeros(context * 4);
        m.anchor = try l.zeros(c.hidden * 2);
        m.pos = try l.zeros(4);
        m.pos3 = try l.zeros(4);
        const ones = try gpa.alloc(f32, context);
        defer gpa.free(ones);
        @memset(ones, 1);
        m.ones = try l.up(std.mem.sliceAsBytes(ones));
        m.graphs = try gpa.alloc(?Graph, (context / 512 + 2) * graph_kinds);
        @memset(m.graphs, null);
        return m;
    }

    pub fn close(m: *Model) void {
        for (m.graphs) |*g| if (g.*) |*x| x.exec.deinit();
        m.gpa.free(m.graphs);
        for (m.scratch.items) |*b| b.free();
        m.scratch.deinit(m.gpa);
        for (m.weights.items) |*b| b.free();
        m.weights.deinit(m.gpa);
        m.tokens.deinit(m.gpa);
        m.gpa.free(m.layers);
        m.gpa.free(m.cache);
    }

    pub fn reset(m: *Model) void {
        m.used = .{};
    }

    /// `n` bytes of the step's scratch, 256-aligned, from its blocks (a new one of 256 MiB or more if none holds them).
    pub fn take(m: *Model, n: usize) !cuda.DeviceBuffer {
        const len = std.mem.alignForward(usize, @max(n, 1), 256);
        while (m.used.block < m.scratch.items.len) : (m.used = .{ .block = m.used.block + 1 }) {
            const b = m.scratch.items[m.used.block];
            if (m.used.at + len <= b.len) {
                defer m.used.at += len;
                return .{ .d = m.d, .ptr = b.ptr + m.used.at, .len = n, .borrowed = true };
            }
        }
        var b = try cuda.DeviceBuffer.alloc(m.d, @max(len, 256 << 20));
        errdefer b.free();
        try m.scratch.append(m.gpa, b);
        m.used = .{ .block = m.scratch.items.len - 1, .at = len };
        return .{ .d = m.d, .ptr = b.ptr, .len = n, .borrowed = true };
    }

    pub fn zeros(m: *Model, n: usize) !u64 {
        const b = try m.take(n);
        try b.fill8(0, m.s.handle);
        return b.ptr;
    }

    /// Scratch every byte of which the launch that takes it writes (no clear).
    pub fn buf(m: *Model, n: usize) !u64 {
        return (try m.take(n)).ptr;
    }

    /// y [rows][w.n] fp32 = x [rows][w.k] bf16 (rows `ldx` apart) times a lane weight.
    pub fn proj(m: *Model, w: Lin, x: u64, ldx: usize, y: u64, rows: usize) !void {
        const pb = m.k.lane.partBytes(rows, w);
        try m.k.lane.matmulF32(m.s, x, ldx, rows, w, y, if (pb > 0) (try m.take(pb)).ptr else 0);
    }

    /// Profiling: the stream's work so far charged to part `p` (a synchronize when on).
    fn mark(m: *Model, p: usize, t: *i96) !void {
        const io = m.prof orelse return;
        try m.s.synchronize();
        const now = std.Io.Clock.awake.now(io).toNanoseconds();
        m.times[p] += now - t.*;
        t.* = now;
    }

    /// Rows `ids` at tokens.len..: each row's greedy token (or the `last`'s) into `out`, `logits` their full logits.
    pub fn step(m: *Model, ids: []const i32, last: bool, out: []i32, logits: ?[]f32) !void {
        // a long prompt goes in chunks: a step's scratch stays bounded whatever the prompt's length
        if (last and ids.len > max_step_rows) {
            var at: usize = 0;
            while (ids.len - at > max_step_rows) : (at += max_step_rows) try m.step(ids[at..][0..max_step_rows], true, out, null);
            return m.step(ids[at..], true, out, logits);
        }
        defer m.reset();
        const n = ids.len;
        const p0 = m.tokens.items.len;
        if (p0 + n > m.context) return error.PastContext;
        try m.tokens.appendSlice(m.gpa, ids);
        errdefer m.tokens.shrinkRetainingCapacity(p0);
        try m.inputs(ids, p0);
        const rows = if (last) 1 else n;
        const kind: ?usize = if (n == 1 and last) 0 else if (n == verify_rows and !last) 1 else null;
        const toks = if (kind != null and m.use_graphs and m.prof == null and logits == null) try m.replay(kind.?, p0, n, last) else try m.compute(n, p0, last);
        if (m.snap) |*sn| sn.keys = p0 + n; // the step keepRow names next: its lists stand when it ran the indexer
        try m.s.synchronize();
        try cuda.memory.download(m.d, toks, std.mem.sliceAsBytes(out[0..rows]));
        if (logits) |full| try m.gatherLogits(rows, full);
    }

    /// The last step's logits of every rank, each row's vocabulary slices in rank order, into `full`.
    fn gatherLogits(m: *Model, rows: usize, full: []f32) !void {
        const vl = m.c.vocab / m.world;
        const all = try m.gpa.alloc(f32, m.world * rows * vl);
        defer m.gpa.free(all);
        if (m.group) |g| {
            const dev = try m.buf(all.len * 4);
            try g.allGather(m.s, m.local, dev, rows * vl * 4);
            try m.s.synchronize();
            try cuda.memory.download(m.d, dev, std.mem.sliceAsBytes(all));
        } else try cuda.memory.download(m.d, m.local, std.mem.sliceAsBytes(all));
        for (0..m.world) |w| for (0..rows) |r| @memcpy(full[r * m.c.vocab + w * vl ..][0..vl], all[(w * rows + r) * vl ..][0..vl]);
    }

    /// The head: this rank's vocabulary slice into `local`, each row's token over all ranks into `ids`.
    pub fn headIds(m: *Model, hn: u64, rows: usize, ids: u64) !void {
        const vl = m.c.vocab / m.world;
        m.local = try m.buf(rows * vl * 4);
        try m.proj(m.lm, hn, m.c.hidden, m.local, rows);
        const pairs = try m.buf(rows * 8);
        try m.k.k.argmaxLocal(m.s, m.local, rows, vl, m.rank * vl, pairs);
        if (m.group) |g| {
            const all = try m.buf(m.world * rows * 8);
            try g.allGather(m.s, pairs, all, rows * 8);
            try m.k.k.argmaxPick(m.s, all, m.world, rows, ids);
        } else try m.k.k.argmaxPick(m.s, pairs, 1, rows, ids);
    }

    /// Row `r` of the last step's final-normed rows into `anchor` (the step's buffers are still in place).
    pub fn keepRow(m: *Model, r: usize) !void {
        const hd: usize = m.c.hidden;
        if (m.snap) |*sn| {
            sn.kept = r;
            const h = [1]i32{@intCast(r)};
            try (cuda.DeviceBuffer{ .d = m.d, .ptr = sn.slot, .len = 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&h));
        }
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = m.anchor, .len = hd * 2, .borrowed = true }).copyFrom(0, m.hn + r * hd * 2, hd * 2, m.s.handle);
    }

    /// The graph under `key`, captured from `ctx.run(m)` on its first use (its scratch from the step's blocks).
    pub fn graphed(m: *Model, key: usize, ctx: anytype) !Graph {
        if (m.graphs[key] == null) {
            if (m.scratch.items.len == 0) _ = try ctx.run(m); // the scratch's blocks exist before a capture
            try m.s.synchronize();
            m.reset();
            try cuda.graph.beginCapture(m.s, .relaxed);
            const out = ctx.run(m) catch |e| {
                var g = cuda.graph.endCapture(m.s) catch return e;
                g.deinit();
                return e;
            };
            var g = try cuda.graph.endCapture(m.s);
            defer g.deinit();
            const exec = try g.instantiate();
            try exec.upload(m.s);
            m.graphs[key] = .{ .exec = exec, .logits = out, .hn = m.hn, .local = m.local };
        }
        const g = m.graphs[key].?;
        try g.exec.launchOn(m.s);
        m.hn = g.hn;
        m.local = g.local;
        return g;
    }

    /// The step's inputs into their device buffers: the streams' rows, the position, each Engram layer's hash ids.
    fn inputs(m: *Model, ids: []const i32, p0: usize) !void {
        const n = ids.len;
        const view = struct {
            fn of(mm: *Model, p: u64, len: usize) cuda.DeviceBuffer {
                return .{ .d = mm.d, .ptr = p, .len = len, .borrowed = true };
            }
        }.of;
        try view(m, m.ids, n * 4).upload(0, std.mem.sliceAsBytes(ids));
        const pos_h = [2]i32{ @intCast(p0), @intCast(p0 + 3) };
        try view(m, m.pos, 4).upload(0, std.mem.sliceAsBytes(pos_h[0..1]));
        try view(m, m.pos3, 4).upload(0, std.mem.sliceAsBytes(pos_h[1..2]));
        const hash = try m.gpa.alloc(i32, n * engram.heads);
        defer m.gpa.free(hash);
        const toks = m.tokens.items;
        for (m.layers) |*L| if (L.engram) |*e| {
            for (0..n) |r| {
                const p = p0 + r;
                var comp: [engram.max_ngram]u32 = undefined;
                for (0..engram.max_ngram) |o| {
                    const tok: i64 = if (p >= o) toks[p - o] else @as(i64, m.eg.pad);
                    const look: usize = if (tok >= 0 and tok < @as(i64, @intCast(e.ids.len))) @intCast(tok) else m.eg.pad;
                    comp[o] = @intCast(e.ids[look]);
                }
                var hv: [engram.heads]u32 = undefined;
                engram.hash(comp, e.mult, &e.moduli, &hv);
                for (hv, 0..) |v, h| hash[r * engram.heads + h] = @intCast(v);
            }
            try view(m, e.hash, hash.len * 4).upload(0, std.mem.sliceAsBytes(hash));
        };
    }

    /// The step's graph (a decode row or a verify block) at this many full-layer chunks.
    fn replay(m: *Model, kind: usize, p0: usize, n: usize, last: bool) !u64 {
        const Run = struct {
            p0: usize,
            n: usize,
            last: bool,
            pub fn run(r: @This(), mm: *Model) !u64 {
                return mm.compute(r.n, r.p0, r.last);
            }
        };
        const nch = (p0 + n + 511) / 512;
        return (try m.graphed(nch * graph_kinds + kind, Run{ .p0 = p0, .n = n, .last = last })).logits;
    }

    /// Rows p0..p0+n through every layer from the input buffers: the logits' device rows (every row, or the last).
    fn compute(m: *Model, n: usize, p0: usize, last: bool) !u64 {
        m.reset();
        const c = m.c;
        const hd: usize = c.hidden;
        m.x = try m.buf(n * c.hc * hd * 2);
        try m.k.embed.bf16Copies(m.s, m.ids, m.table, m.x, n, hd, c.hc);
        var stage: []const u8 = "embed";
        var at: usize = 0;
        errdefer std.log.err("hy4 step: failed in layer {d}'s {s} ({d} rows at {d})", .{ at, stage, n, p0 });
        var t: i96 = if (m.prof) |io| std.Io.Clock.awake.now(io).toNanoseconds() else 0;
        try m.mark(3, &t);
        const kept = m.used; // a layer's scratch is taken back at its end: a step holds one layer's at a time
        for (m.layers, 0..) |*L, i| {
            m.used = kept;
            at = i;
            m.at_layer = i;
            stage = "engram";
            if (c.engram[i]) try m.engramAdd(if (L.engram) |*e| e else null, n);
            try m.mark(0, &t);
            stage = "attention";
            const h = try m.buf(n * hd * 2);
            const post = try m.buf(n * c.hc * 4);
            try m.k.k.hcPre(m.s, m.x, L.hc_attn, h, post, n, hd, c.eps, hc_eps, hc_magnitude);
            const hn = try m.buf(n * hd * 2);
            try m.k.k.rmsNorm16(m.s, h, hd, L.in_norm, hn, hd, n, hd, c.eps);
            try m.k.k.hcPost(m.s, m.x, try m.attend(&L.attn, m.cache[i], hn, m.pos, p0, n), post, n, hd);
            try m.mark(1, &t);
            stage = "mlp";
            const h2 = try m.buf(n * hd * 2);
            const post2 = try m.buf(n * c.hc * 4);
            try m.k.k.hcPre(m.s, m.x, L.hc_ffn, h2, post2, n, hd, c.eps, hc_eps, hc_magnitude);
            const hn2 = try m.buf(n * hd * 2);
            try m.k.k.rmsNorm16(m.s, h2, hd, L.post_norm, hn2, hd, n, hd, c.eps);
            try m.k.k.hcPost(m.s, m.x, try m.mlp(&L.mlp, hn2, n), post2, n, hd);
            try m.mark(2, &t);
        }
        stage = "head";
        const rows = if (last) 1 else n;
        const xs = m.x + (n - rows) * c.hc * hd * 2;
        const h = try m.buf(rows * hd * 2);
        try m.k.k.hcHead(m.s, xs, m.head, h, rows, hd, c.eps, hc_eps);
        const hn = try m.buf(rows * hd * 2);
        try m.k.k.rmsNorm16(m.s, h, hd, m.norm, hn, hd, rows, hd, c.eps);
        const toks = try m.buf(rows * 4);
        try m.headIds(hn, rows, toks);
        try m.mark(3, &t);
        m.hn = hn;
        return toks;
    }

    /// Rows p0..: latent rows into `cache`, then MLA over it (sliding: the last `window` keys a row as a token list).
    pub fn attend(m: *Model, a: *const Attn, cache: u64, x: u64, pos: u64, p0: usize, n: usize) !u64 {
        return m.attendList(a, cache, x, pos, p0, n, null);
    }

    /// A token list each row reads in place of the layer's own choice (DFly over the target's recalled keys).
    pub const List = struct { tok: u64, cnt: u64, w: usize };
    pub fn attendList(m: *Model, a: *const Attn, cache: u64, x: u64, pos: u64, p0: usize, n: usize, list: ?List) !u64 {
        const c = m.c;
        const hd: usize = c.hidden;
        const H = a.heads;
        const k = &m.k.k;
        const qr_n: usize = c.q_rank;
        const qa_f = try m.buf(n * qr_n * 4);
        try m.proj(a.q_a, x, hd, qa_f, n);
        const qr = try m.buf(n * qr_n * 2);
        try k.rmsNorm(m.s, qa_f, qr_n, a.q_a_norm, qr, qr_n, n, qr_n, c.eps);
        const q_f = try m.buf(n * H * 256 * 4);
        try m.proj(a.q_b, qr, qr_n, q_f, n);
        const q = try m.buf(n * H * 256 * 2);
        try k.castRows(m.s, q_f, H * 256, q, H * 256, n, H * 256);
        var t: i96 = if (m.prof) |io| std.Io.Clock.awake.now(io).toNanoseconds() else 0;
        const qa = try m.buf(n * H * row_width * 2);
        try m.k.mla.rope(m.s, .{ .x = q + 192 * 2, .x_row = H * 256, .x_head = 256, .out = qa + 512 * 2, .o_row = H * row_width, .o_head = row_width }, n, H, pos, a.inv_freq, 1.0);
        const kv_f = try m.buf(n * row_width * 4);
        try m.proj(a.kv_a, x, hd, kv_f, n);
        const rows = try m.buf(n * row_width * 2); // put into the cache at the device position (graphs replay it)
        try k.rmsNorm(m.s, kv_f, row_width, a.kv_a_norm, rows, row_width, n, 512, c.eps);
        try k.castRows(m.s, kv_f + 512 * 4, row_width, rows + 512 * 2, row_width, n, 64);
        try m.k.mla.rope(m.s, .{ .x = rows + 512 * 2, .x_row = row_width, .x_head = 0, .out = rows + 512 * 2, .o_row = row_width, .o_head = 0 }, n, 1, pos, a.inv_freq, 1.0);
        try k.put(m.s, cache, rows, pos, n, row_width);
        if (n >= per_head.min_rows) try per_head.absorb(m, a, q, qa, n) else try k.absorbQ(m.s, q, a.wk, qa, n, H);
        try m.mark(4, &t);
        const keys = p0 + n;
        const win: usize = c.window;
        const recall: usize = c.index_topk; // the indexer runs once the keys outgrow it
        if (a.idx) |ix| try index.keys(m, ix, x, pos, a.inv_freq, n);
        const indexed = a.idx != null and keys > recall and m.use_index;
        const nch: usize = if (list) |li| (li.w + 511) / 512 else if (a.sliding) (win + 511) / 512 else if (indexed) recall / 512 else (keys + 511) / 512;
        const part: cuda.mla.Partials = .{ .po = try m.zeros(nch * n * H * 512 * 4), .pm = try m.zeros(nch * n * H * 4), .pl = try m.zeros(nch * n * H * 4) };
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.sparse.nope + c.sparse.rope)));
        if (list) |li| {
            try hy4.sparseChunks(&m.k.mla, m.s, qa, cache, li.tok, li.cnt, li.w, part, n, H, scale);
        } else if (a.sliding) {
            const tok = try m.buf(n * win * 4);
            const cnt = try m.buf(n * 4);
            try k.window(m.s, pos, tok, cnt, n, win);
            try hy4.sparseChunks(&m.k.mla, m.s, qa, cache, tok, cnt, win, part, n, H, scale);
        } else if (indexed) {
            const tok = try m.buf(n * recall * 4);
            const cnt = try m.buf(n * 4);
            try index.recalled(m, a.idx.?, x, qr, pos, a.inv_freq, (keys + 511) / 512 * 128, n, tok, cnt);
            try hy4.sparseChunks(&m.k.mla, m.s, qa, cache, tok, cnt, recall, part, n, H, scale);
        } else try m.k.mla.chunks(m.s, qa, cache, pos, part, n, H, nch, scale);
        try m.mark(5, &t);
        const ol = try m.buf(n * H * 512 * 2);
        try k.mergeSink(m.s, part, a.sink, ol, n, H, nch);
        const o = try m.buf(n * H * 256 * 2);
        if (n >= per_head.min_rows) try per_head.expand(m, a, ol, o, n) else try k.expandV(m.s, ol, a.wv, o, n, H);
        try m.mark(7, &t);
        const g = try m.buf(n * H * 256 * 4);
        try m.proj(a.gate, x, hd, g, n);
        try k.gateRows(m.s, o, g, n * H * 256);
        try m.mark(6, &t);
        const po = try m.buf(n * hd * 4);
        try m.proj(a.o, o, H * 256, po, n);
        try m.sum(po, n * hd);
        const out = try m.buf(n * hd * 2);
        try k.rmsNorm(m.s, po, hd, a.post_norm, out, hd, n, hd, c.eps);
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
        const act = try m.buf(pairs * t.width * 2);
        const y = try m.zeros(pairs * t.dims * 2);
        const g: cuda.experts.Grouped = .{ .nvfp4 = &m.k.x };
        try g.prompt(m.s, .up, .{ .x = x, .stride = t.dims, .slots = per }, t, plan, act, items, pairs, tile, skip);
        try g.prompt(m.s, .down_bf16, .{ .x = act, .stride = t.width }, t, plan, y, items, pairs, tile, skip);
        const out = try m.buf(n * t.dims * 2);
        try m.k.glue.combine(m.s, y, .bf16, wts, out, .bf16, n, per, t.dims);
        return out;
    }

    /// One expert for every row at weight 1 (the shared expert, the dense MLP).
    fn one(m: *Model, t: cuda.experts.Layer, x: u64, n: usize) !u64 {
        return m.grouped(t, x, try m.zeros(n * 4), m.ones, n, 1, -1);
    }

    fn mlp(m: *Model, p: *const Mlp, x: u64, n: usize) !u64 {
        const c = m.c;
        const hd: usize = c.hidden;
        const out = try m.buf(n * hd * 2);
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
                const logits = try m.buf(n * E * 4);
                try m.proj(e.router, x, hd, logits, n);
                const picks = try m.buf(n * K * 4);
                const wts = try m.buf(n * K * 4);
                try m.k.glue.topk(m.s, logits, e.bias, .{ .experts = E, .topk = K, .shared = false, .normalize = true, .scale = c.route_scale, .score = true }, picks, wts, n);
                const xl_f = try m.buf(n * lat * 4);
                try m.proj(e.down, x, hd, xl_f, n);
                const xl = try m.buf(n * lat * 2);
                try m.k.k.castRows(m.s, xl_f, lat, xl, lat, n, lat);
                const own = e.routed.experts;
                if (m.world > 1) try m.k.k.own(m.s, picks, wts, n * K, m.rank * own, own);
                const combined = try m.grouped(e.routed, xl, picks, wts, n, K, if (m.world > 1) @intCast(own) else -1);
                const routed = try m.buf(n * hd * 4);
                try m.proj(e.up, combined, lat, routed, n);
                try m.k.k.addb(m.s, routed, try m.one(e.shared, x, n), n * hd);
                try m.sum(routed, n * hd);
                try m.k.k.rmsNorm(m.s, routed, hd, e.norm, out, hd, n, hd, c.eps);
            },
        }
        return out;
    }

    /// Engram's residual into the streams of rows p0..: each row's n-grams from the tokens before it.
    fn engramAdd(m: *Model, e_: ?*const Engram, n: usize) !void {
        const hd: usize = m.c.hidden;
        const S: usize = m.c.hc;
        const res = try m.zeros(n * S * hd * 4);
        if (e_) |e| try m.engramResidual(e, n, res);
        try m.sum(res, n * S * hd);
        try m.k.k.add16(m.s, m.x, res, n * S * hd);
    }

    /// Engram's residual [n][S][hidden] fp32 of the step's rows on the rank that holds the layer's table.
    fn engramResidual(m: *Model, e: *const Engram, n: usize, res: u64) !void {
        const hd: usize = m.c.hidden;
        const S: usize = m.c.hc;
        const rows = try m.buf(n * engram.heads * 96 * 2);
        try m.k.k.engramRows(m.s, e.hash, e.offsets, e.table, e.row_norm, eps_engram, rows, n);
        const ew = engram.heads * 96;
        const kv = try m.buf(n * (S + 1) * hd * 4); // each row [k0 k1 k2 k3 v] fp32, on the tensor cores
        try m.proj(e.proj, rows, ew, kv, n);
        try m.k.k.bias(m.s, kv, (S + 1) * hd, e.bias, n, (S + 1) * hd);
        try m.k.k.engramMix(m.s, kv, kv + S * hd * 4, m.x, e.mix, eps_engram, res, n, hd);
    }
};
