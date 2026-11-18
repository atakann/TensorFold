//! hy4-real-check (GPU): the NVFP4 checkpoint's first 6 layers against transformers on them dequantized to bf16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const hy4 = @import("hy4_flash");
const t = @import("hy4_checks.zig");

const engram = hy4.engram;
const X = cuda.experts.Nvfp4Experts;
const Ctx = t.Ctx;
const hidden = t.hidden;
const streams = t.streams;
const eps = t.eps;
/// Engram's nn.RMSNorm(eps=None): the bf16 reference matches float32's eps, not bf16's (2^-7: rel err 1.10).
const eps_engram: f32 = std.math.floatEps(f32);

const kinds = [_]t.Layer{
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 64, .theta = 1e4, .sliding = true },
    .{ .heads = 64, .theta = 1e4, .sliding = true },
    .{ .heads = 64, .theta = 1e4, .sliding = true },
    .{ .heads = 32, .theta = 1e7, .sliding = false },
};
const engram_layer = 1;
const experts = 512;
const topk = 16;
const latent = 1536;
const shared_inter = 1536;
const dense_inter = 13824;
const pad_id = 120002;
/// The checkpoint's folder (its config.json holds the Engram head sizes).
var dir: []const u8 = "";

/// The grouped expert path: the plan kernels, NVFP4 experts and the shared routing and combine.
const Moe = struct {
    mod: cuda.Module,
    router: cuda.grouped.Router,
    x: X,
    glue: cuda.moe_glue.MoeGlue,

    fn load(gpu: check.Gpu) !Moe {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        return .{ .mod = mod, .router = try cuda.grouped.Router.resolve(mod), .x = try X.load(gpu.d, sms), .glue = try cuda.moe_glue.MoeGlue.load(gpu.d) };
    }

    fn unload(m: *Moe) void {
        m.glue.unload();
        m.x.unload();
        m.mod.unload();
    }
};

/// `count` NVFP4 SwiGLU experts under `pre` ("{pre}{e}." or "{pre}" for one) packed as X lays them out.
fn table(c: *Ctx, comptime pre: []const u8, args: anytype, comptime single: bool, count: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
    const gpa = c.gpu.gpa;
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const up = try gpa.alloc(u32, count * up_len);
    defer gpa.free(up);
    const down = try gpa.alloc(u32, count * down_len);
    defer gpa.free(down);
    const us = try gpa.alloc(f32, count * 2);
    defer gpa.free(us);
    const ds = try gpa.alloc(f32, count);
    defer gpa.free(ds);
    for (0..count) |e| {
        var q: [3][2][]const u8 = undefined;
        var g: [3]f32 = undefined;
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const name = if (single) pre ++ proj ++ "." else pre ++ "{d}." ++ proj ++ ".";
            const xa = if (single) args else args ++ .{e};
            q[m][0] = (try c.tensor(name ++ "weight", xa)).bytes;
            q[m][1] = (try c.tensor(name ++ "weight_scale", xa)).bytes;
            g[m] = std.mem.bytesAsValue(f32, (try c.tensor(name ++ "weight_scale_2", xa)).bytes[0..4]).*;
        }
        X.packGateUp(up[e * up_len ..][0..up_len], q[0], q[1], ni, d);
        X.packOne(down[e * down_len ..][0..down_len], q[2][0], q[2][1], d, ni);
        us[e * 2] = g[0];
        us[e * 2 + 1] = g[1];
        ds[e] = g[2];
    }
    return .{ .format = .nvfp4, .up = try c.up(std.mem.sliceAsBytes(up)), .down = try c.up(std.mem.sliceAsBytes(down)), .up_scale = try c.up(std.mem.sliceAsBytes(us)), .down_scale = try c.up(std.mem.sliceAsBytes(ds)), .width = ni, .dims = d, .experts = count, .limit = limit };
}

/// x [rows][d] bf16 through `l` on the picks (`per` a row) into bf16 [rows][d] weighted by `wts`.
fn grouped(c: *Ctx, m: *const Moe, l: cuda.experts.Layer, x: u64, picks: u64, wts: u64, rows: usize, per: usize) !u64 {
    const pairs = rows * per;
    const tile = cuda.experts.promptTile(.nvfp4);
    const items = cuda.grouped.maxItems(pairs, l.experts, tile);
    const wide = pairs > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try c.zeros(pairs * 4),
        .items = try c.zeros(items * 12),
        .counts = try c.zeros(8),
        .rank = try c.zeros((if (wide) pairs else 1) * 4),
        .hist = try c.zeros((if (wide) (pairs + 1023) / 1024 * l.experts else 1) * 4),
    };
    try m.router.route(c.s, picks, pairs, l.experts, tile, plan);
    const act = try c.zeros(pairs * l.width * 2);
    const y = try c.zeros(pairs * l.dims * 2);
    const g: cuda.experts.Grouped = .{ .nvfp4 = &m.x };
    try g.prompt(c.s, .up, .{ .x = x, .stride = l.dims, .slots = per }, l, plan, act, items, pairs, tile, -1);
    try g.prompt(c.s, .down_bf16, .{ .x = act, .stride = l.width }, l, plan, y, items, pairs, tile, -1);
    const out = try c.zeros(rows * l.dims * 2);
    try m.glue.combine(c.s, y, .bf16, wts, out, .bf16, rows, per, l.dims);
    return out;
}

/// One expert for every row at weight 1 (the shared expert, layer 0's dense MLP).
fn one(c: *Ctx, m: *const Moe, l: cuda.experts.Layer, x: u64, rows: usize) !u64 {
    const ones = try c.gpu.gpa.alloc(f32, rows);
    defer c.gpu.gpa.free(ones);
    @memset(ones, 1);
    return grouped(c, m, l, x, try c.zeros(rows * 4), try c.up(std.mem.sliceAsBytes(ones)), rows, 1);
}

fn moe(c: *Ctx, m: *const Moe, i: usize, x: u64, rows: usize) !u64 {
    const p = "model.layers.{d}.mlp.";
    const logits = try c.zeros(rows * experts * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "gate.weight", .{i}), logits, experts, rows, experts, hidden);
    const picks = try c.zeros(rows * topk * 4);
    const wts = try c.zeros(rows * topk * 4);
    try m.glue.topk(c.s, logits, try c.w(p ++ "gate.e_score_correction_bias", .{i}), .{ .experts = experts, .topk = topk, .shared = false, .normalize = true, .scale = 3.998, .score = true }, picks, wts, rows);
    const xl_f = try c.zeros(rows * latent * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "routed_expert_down_proj.weight", .{i}), xl_f, latent, rows, latent, hidden);
    const xl = try c.zeros(rows * latent * 2);
    try c.k.castRows(c.s, xl_f, latent, xl, latent, rows, latent);
    const routed_l = try table(c, p ++ "experts.", .{i}, false, experts, latent, latent, 10.0);
    const combined = try grouped(c, m, routed_l, xl, picks, wts, rows, topk);
    const routed = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, combined, latent, try c.w(p ++ "routed_expert_up_proj.weight", .{i}), routed, hidden, rows, hidden, latent);
    const shared_l = try table(c, p ++ "shared_experts.", .{i}, true, 1, shared_inter, hidden, 0);
    try c.k.addb(c.s, routed, try one(c, m, shared_l, x, rows), rows * hidden);
    const out = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm(c.s, routed, hidden, try c.w(p ++ "combine_output_norm.weight", .{i}), out, hidden, rows, hidden, eps);
    return out;
}

fn denseMlp(c: *Ctx, m: *const Moe, x: u64, rows: usize) !u64 {
    const l = try table(c, "model.layers.0.mlp.", .{}, true, 1, dense_inter, hidden, 0);
    const y = try one(c, m, l, x, rows);
    const out = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm16(c.s, y, hidden, try c.w("model.layers.0.mlp.post_fc2_norm.weight", .{}), out, hidden, rows, hidden, eps);
    return out;
}

/// The layer's hash-head table sizes from config.json's engram_layer_head_vocab_sizes (2-grams, then 3-grams).
fn headSizes(c: *Ctx, layer_id: usize) ![engram.heads]u32 {
    const gpa = c.gpu.gpa;
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(c.gpu.io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const tc = if (root.get("text_config")) |v| v.object else root;
    var key: [8]u8 = undefined;
    const orders = tc.get("engram_layer_head_vocab_sizes").?.object.get(try std.fmt.bufPrint(&key, "{d}", .{layer_id})).?.array.items;
    var out: [engram.heads]u32 = undefined;
    var n: usize = 0;
    for (orders) |o| for (o.array.items) |v| {
        out[n] = @intCast(v.integer);
        n += 1;
    };
    if (n != engram.heads) return error.EngramHeads;
    return out;
}

/// Engram on the stored FP8 rows, the hash ids from engram.zig's release multipliers checked against the reference.
fn engramAdd(c: *Ctx, ref: *core.checkpoint.Checkpoint, i: usize, x: u64, ids: []align(1) const i32) !void {
    const gpa = c.gpu.gpa;
    const rows = ids.len;
    const p = "model.layers.{d}.engram_layer.";
    const want: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try ref.get("engram.1.hash")).bytes);
    const table_ids: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try c.tensor(p ++ "lookup_table", .{i})).bytes);
    const mult = engram.multipliers(@intCast(i + 1)) orelse return error.NoMultipliers;
    const moduli = try headSizes(c, i + 1);
    var offsets: [engram.heads]i64 = undefined;
    var off: i64 = 0;
    for (moduli, 0..) |m, h| {
        offsets[h] = off;
        off += m;
    }
    const hash = try gpa.alloc(i32, rows * engram.heads);
    defer gpa.free(hash);
    for (0..rows) |r| {
        var comp: [engram.max_ngram]u32 = undefined;
        for (0..engram.max_ngram) |o| {
            const tok: i32 = if (r >= o) ids[r - o] else pad_id;
            const look: usize = if (tok < table_ids.len) @intCast(tok) else pad_id;
            comp[o] = @intCast(table_ids[look]);
        }
        var out: [engram.heads]u32 = undefined;
        engram.hash(comp, mult, &moduli, &out);
        for (out, 0..) |v, h| hash[r * engram.heads + h] = @intCast(v);
    }
    var diff: usize = 0;
    for (hash, want) |a, b| diff += @intFromBool(a != b);
    try check.expect(diff == 0, "hy4 real engram hash ids: {d} of {d} differ from the reference", .{ diff, hash.len });
    check.pass("hy4 real engram hash ids of layer {d}: {d} as the reference", .{ i, hash.len });
    const e = try c.zeros(rows * engram.heads * 96 * 2);
    const tbl: hy4.EngramTable = .{ .fp8 = try c.w(p ++ "multi_head_embedding.embedding.weight", .{i}), .scales = try c.w(p ++ "multi_head_embedding.embedding.weight_scale_inv", .{i}) };
    try c.k.engramRows(c.s, try c.up(std.mem.sliceAsBytes(hash)), try c.up(std.mem.sliceAsBytes(&offsets)), tbl, try c.wf32(p ++ "lookup_per_head_norms.{d}.weight", .{i}, engram.heads), eps_engram, e, rows);
    const ew = engram.heads * 96;
    check.pass("hy4 real: Engram's normed rows: rel err {e:.3}", .{try t.relErr(c, e, false, try t.refF32(ref, "engram.1.e", .{}))});
    const value = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, e, ew, try c.w(p ++ "value_proj.weight", .{i}), value, hidden, rows, hidden, ew);
    try c.k.bias(c.s, value, hidden, try c.w(p ++ "value_proj.bias", .{i}), rows, hidden);
    check.pass("hy4 real: Engram's value: rel err {e:.3}", .{try t.relErr(c, value, true, try t.refF32(ref, "engram.1.value", .{}))});
    const key = try c.zeros(rows * streams * hidden * 4);
    for (0..streams) |s| {
        try c.dense.rows(c.s, e, ew, try c.w(p ++ "key_projs.{d}.weight", .{ i, s }), key + s * hidden * 4, streams * hidden, rows, hidden, ew);
        try c.k.bias(c.s, key + s * hidden * 4, streams * hidden, try c.w(p ++ "key_projs.{d}.bias", .{ i, s }), rows, hidden);
    }
    const mix: hy4.EngramMix = .{
        .norm_key = try c.wf32(p ++ "norm_key.{d}.weight", .{i}, streams),
        .norm_query = try c.wf32(p ++ "norm_query.{d}.weight", .{i}, streams),
        .gamma = try c.wf32One(p ++ "unified_pre_gamma", .{i}),
        .hc_scale = try c.wf32(p ++ "unified_per_hc_scales.{d}", .{i}, streams),
    };
    const res = try c.zeros(rows * streams * hidden * 4);
    try c.k.engramMix(c.s, key, value, x, mix, eps_engram, res, rows, hidden);
    check.pass("hy4 real: Engram's residual alone: rel err {e:.3}", .{try t.relErr(c, res, true, try t.refF32(ref, "engram.1.out", .{}))});
    try c.k.add16(c.s, x, res, rows * streams * hidden);
}

fn forward(c: *Ctx, m: *const Moe, ref: *core.checkpoint.Checkpoint) !void {
    const gpa = c.gpu.gpa;
    const ids: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try ref.get("input_ids")).bytes);
    const rows = ids.len;
    const emb = (try c.tensor("model.embed_tokens.weight", .{})).bytes;
    const x0 = try gpa.alloc(u8, rows * streams * hidden * 2);
    defer gpa.free(x0);
    for (ids, 0..) |id, r| for (0..streams) |s| {
        @memcpy(x0[(r * streams + s) * hidden * 2 ..][0 .. hidden * 2], emb[@as(usize, @intCast(id)) * hidden * 2 ..][0 .. hidden * 2]);
    };
    var st = try cuda.DeviceBuffer.fromHost(c.gpu.d, x0);
    defer st.free();
    const x = st.ptr;
    const pre = "model.layers.{d}.";
    {
        // the MoE module alone on the reference's own input
        defer c.reset();
        const in32 = try t.refF32(ref, "mlp.1.in", .{});
        const in16 = try gpa.alloc(u16, in32.len);
        defer gpa.free(in16);
        for (in16, in32) |*o, v| o.* = t.tobf16(v);
        const out = try moe(c, m, 1, try c.up(std.mem.sliceAsBytes(in16)), rows);
        check.pass("hy4 real: layer 1's MoE alone: rel err {e:.3}", .{try t.relErr(c, out, false, try t.refF32(ref, "mlp.1.out", .{}))});
    }
    for (0..kinds.len) |i| {
        defer c.reset();
        if (i == engram_layer) try engramAdd(c, ref, i, x, ids);
        const h = try c.zeros(rows * hidden * 2);
        const post = try c.zeros(rows * streams * 4);
        try c.k.hcPre(c.s, x, try c.hc(pre ++ "hc_attn_layer.hc_pre.hc", .{i}), h, post, rows, hidden, eps, 1e-6, 2.0);
        const hn = try c.zeros(rows * hidden * 2);
        try c.k.rmsNorm16(c.s, h, hidden, try c.w(pre ++ "input_layernorm.weight", .{i}), hn, hidden, rows, hidden, eps);
        try c.k.hcPost(c.s, x, try t.attend(c, i, hn, rows), post, rows, hidden);
        const h2 = try c.zeros(rows * hidden * 2);
        const post2 = try c.zeros(rows * streams * 4);
        try c.k.hcPre(c.s, x, try c.hc(pre ++ "hc_mlp_layer.hc_pre.hc", .{i}), h2, post2, rows, hidden, eps, 1e-6, 2.0);
        const hn2 = try c.zeros(rows * hidden * 2);
        try c.k.rmsNorm16(c.s, h2, hidden, try c.w(pre ++ "post_attention_layernorm.weight", .{i}), hn2, hidden, rows, hidden, eps);
        try c.k.hcPost(c.s, x, if (i == 0) try denseMlp(c, m, hn2, rows) else try moe(c, m, i, hn2, rows), post2, rows, hidden);
        if (i + 1 == kinds.len) break; // the reference's last hidden state is the final norm's output
        const rel = try t.relErr(c, x, false, try t.refF32(ref, "hidden.{d}", .{i + 1}));
        check.pass("hy4 real: streams after layer {d} ({s}): rel err {e:.3}", .{ i, if (kinds[i].sliding) "sliding" else "sparse", rel });
        try check.expect(rel < 5e-2, "hy4 real: streams after layer {d}: rel err {e}", .{ i, rel });
    }
    defer c.reset();
    const h = try c.zeros(rows * hidden * 2);
    try c.k.hcHead(c.s, x, try c.hc("model.hc_head.hc_head", .{}), h, rows, hidden, eps, 1e-6);
    const hn = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm16(c.s, h, hidden, try c.w("model.norm.weight", .{}), hn, hidden, rows, hidden, eps);
    const frel = try t.relErr(c, hn, false, try t.refF32(ref, "hidden.{d}", .{kinds.len}));
    check.pass("hy4 real: final norm after layer 5 and the head: rel err {e:.3}", .{frel});
    try check.expect(frel < 5e-2, "hy4 real: final norm: rel err {e}", .{frel});
    const lm = try c.tensor("lm_head.weight", .{});
    const vocab = lm.dim(0);
    const lm_w = if (lm.dtype == .bf16) try c.up(lm.bytes) else blk: {
        const lm16 = try gpa.alloc(u16, vocab * hidden);
        defer gpa.free(lm16);
        for (lm16, 0..) |*o, j| o.* = t.tobf16(std.mem.bytesAsValue(f32, lm.bytes[j * 4 ..][0..4]).*);
        break :blk try c.up(std.mem.sliceAsBytes(lm16));
    };
    const logits = try c.zeros(rows * vocab * 4);
    try c.dense.rows(c.s, hn, hidden, lm_w, logits, vocab, rows, vocab, hidden);
    const want = try t.refF32(ref, "logits", .{});
    const lrel = try t.relErr(c, logits, true, want);
    const got = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(got);
    try cuda.memory.download(c.gpu.d, logits, std.mem.sliceAsBytes(got));
    var same: usize = 0;
    for (0..rows) |r| same += @intFromBool(std.mem.indexOfMax(f32, got[r * vocab ..][0..vocab]) == t.argmax(want[r * vocab ..][0..vocab]));
    check.pass("hy4 real: logits rel err {e:.3}, argmax {d}/{d} as the reference", .{ lrel, same, rows });
    try check.expect(same * 10 >= rows * 9, "hy4 real: argmax {d}/{d}", .{ same, rows });
}

/// hy4-real-check REAL6 REF: the checkpoint's first 6 layers, then real_reference.py's dump.
pub fn run(gpu: check.Gpu, rest: []const []const u8) !void {
    if (rest.len != 2) return error.UnknownCommand;
    dir = rest[0];
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var k = try hy4.Kernels.load(gpu.d);
    defer k.unload();
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();
    var dense = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer dense.unload();
    var m = try Moe.load(gpu);
    defer m.unload();
    var ck = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, rest[0]);
    defer ck.close();
    var ref = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, rest[1]);
    defer ref.close();
    var c: Ctx = .{ .gpu = gpu, .s = s, .k = &k, .mla = &mla, .dense = &dense, .ck = &ck, .kinds = &kinds };
    defer c.deinit();
    return forward(&c, &m, &ref);
}
