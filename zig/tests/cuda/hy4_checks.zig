//! hy4-attn-check and hy4-forward-check (GPU): a tiny Hy4-flash checkpoint against the transformers reference.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const hy4 = @import("hy4_flash");
const engram = hy4.engram;

pub const hidden = 3072;
pub const streams = 4;
pub const eps: f32 = 1e-5;
/// nn.RMSNorm(eps=None) in Engram takes its dtype's eps: the reference runs in float32.
const eps32: f32 = std.math.floatEps(f32);

pub const Layer = struct { heads: usize, theta: f64, sliding: bool };
const layers = [_]Layer{
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 64, .theta = 1e4, .sliding = true },
};
const engram_layer = 1;
const experts = 32;
const topk = 4;
const latent = 1536;
const dense_inter = 13824;

const Bufs = std.ArrayList(cuda.DeviceBuffer);

/// The kernels, the checkpoint and the buffers of the step at hand (freed by `reset`).
pub const Ctx = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    k: *const hy4.Kernels,
    mla: *const cuda.mla.Mla,
    dense: *const cuda.dense_rows.DenseRows,
    ck: *core.checkpoint.Checkpoint,
    kinds: []const Layer = &layers,
    bufs: Bufs = .empty,

    pub fn reset(c: *Ctx) void {
        for (c.bufs.items) |*b| b.free();
        c.bufs.clearRetainingCapacity();
    }

    pub fn deinit(c: *Ctx) void {
        c.reset();
        c.bufs.deinit(c.gpu.gpa);
    }

    pub fn keep(c: *Ctx, b: cuda.DeviceBuffer) !u64 {
        try c.bufs.append(c.gpu.gpa, b);
        return b.ptr;
    }

    /// Cleared on the check's stream, so a clear never lands after a launch that wrote the buffer.
    pub fn zeros(c: *Ctx, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(c.gpu.d, n);
        errdefer b.free();
        try b.fill8(0, c.s.handle);
        return c.keep(b);
    }

    pub fn up(c: *Ctx, bytes: []const u8) !u64 {
        return c.keep(try cuda.DeviceBuffer.fromHost(c.gpu.d, bytes));
    }

    pub fn tensor(c: *Ctx, comptime fmt: []const u8, args: anytype) !core.checkpoint.Tensor {
        var buf: [200]u8 = undefined;
        return c.ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    }

    pub fn w(c: *Ctx, comptime fmt: []const u8, args: anytype) !u64 {
        return c.up((try c.tensor(fmt, args)).bytes);
    }

    /// `count` bf16 tensors (the last format argument their index), one after the other as fp32 on the device.
    pub fn wf32(c: *Ctx, comptime fmt: []const u8, args: anytype, count: usize) !u64 {
        var all: std.ArrayList(f32) = .empty;
        defer all.deinit(c.gpu.gpa);
        for (0..count) |j| {
            const t = try c.tensor(fmt, args ++ .{j});
            const v: []align(1) const u16 = std.mem.bytesAsSlice(u16, t.bytes);
            for (v) |b| try all.append(c.gpu.gpa, bf16f(b));
        }
        return c.up(std.mem.sliceAsBytes(all.items));
    }

    pub fn wf32One(c: *Ctx, comptime fmt: []const u8, args: anytype) !u64 {
        const v: []align(1) const u16 = std.mem.bytesAsSlice(u16, (try c.tensor(fmt, args)).bytes);
        const f = try c.gpu.gpa.alloc(f32, v.len);
        defer c.gpu.gpa.free(f);
        for (f, v) |*o, b| o.* = bf16f(b);
        return c.up(std.mem.sliceAsBytes(f));
    }

    pub fn hc(c: *Ctx, comptime prefix: []const u8, args: anytype) !hy4.Hc {
        return .{ .fn_ = try c.w(prefix ++ "_fn", args), .base = try c.w(prefix ++ "_base", args), .scale = try c.w(prefix ++ "_scale", args) };
    }
};

pub fn bf16f(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

pub fn tobf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// One layer's attention module: x [rows][hidden] bf16 in, the module's output [rows][hidden] bf16 back.
pub fn attend(c: *Ctx, i: usize, x: u64, rows: usize) !u64 {
    const L = c.kinds[i];
    const H = L.heads;
    const k = c.k;
    const p = "model.layers.{d}.self_attn.";
    // q: q_a, its norm, q_b; the rope on the last 64 of each head into qa's columns 512..
    const qa_f = try c.zeros(rows * 1024 * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "q_a_proj.weight", .{i}), qa_f, 1024, rows, 1024, hidden);
    const qr = try c.zeros(rows * 1024 * 2);
    try k.rmsNorm(c.s, qa_f, 1024, try c.w(p ++ "q_a_layernorm.weight", .{i}), qr, 1024, rows, 1024, eps);
    const q_f = try c.zeros(rows * H * 256 * 4);
    try c.dense.rows(c.s, qr, 1024, try c.w(p ++ "q_b_proj.weight", .{i}), q_f, H * 256, rows, H * 256, 1024);
    const q = try c.zeros(rows * H * 256 * 2);
    try k.castRows(c.s, q_f, H * 256, q, H * 256, rows, H * 256);
    const freqs = cuda.mla.ropeFreqs(L.theta);
    const inv_freq = try c.up(std.mem.sliceAsBytes(&freqs));
    const pos = try c.zeros(4);
    const qa = try c.zeros(rows * H * 576 * 2);
    try c.mla.rope(c.s, .{ .x = q + 192 * 2, .x_row = H * 256, .x_head = 256, .out = qa + 512 * 2, .o_row = H * 576, .o_head = 576 }, rows, H, pos, inv_freq, 1.0);
    // the latent cache rows: kv_a's 512 normed, then its 64 roped
    const kv_f = try c.zeros(rows * 576 * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "kv_a_proj_with_mqa.weight", .{i}), kv_f, 576, rows, 576, hidden);
    const lc = try c.zeros(rows * 576 * 2);
    try k.rmsNorm(c.s, kv_f, 576, try c.w(p ++ "kv_a_layernorm.weight", .{i}), lc, 576, rows, 512, eps);
    try k.castRows(c.s, kv_f + 512 * 4, 576, lc + 512 * 2, 576, rows, 64);
    try c.mla.rope(c.s, .{ .x = lc + 512 * 2, .x_row = 576, .x_head = 0, .out = lc + 512 * 2, .o_row = 576, .o_head = 0 }, rows, 1, pos, inv_freq, 1.0);
    // kv_b split into wk [H][192][512] and wv [H][256][512]
    const kvb = (try c.tensor(p ++ "kv_b_proj.weight", .{i})).bytes;
    const gpa = c.gpu.gpa;
    const wk_h = try gpa.alloc(u8, H * 192 * 512 * 2);
    defer gpa.free(wk_h);
    const wv_h = try gpa.alloc(u8, H * 256 * 512 * 2);
    defer gpa.free(wv_h);
    for (0..H) |h| {
        @memcpy(wk_h[h * 192 * 1024 ..][0 .. 192 * 1024], kvb[h * 448 * 1024 ..][0 .. 192 * 1024]);
        @memcpy(wv_h[h * 256 * 1024 ..][0 .. 256 * 1024], kvb[(h * 448 + 192) * 1024 ..][0 .. 256 * 1024]);
    }
    try k.absorbQ(c.s, q, try c.up(wk_h), qa, rows, H);
    // attention: dense causal chunks, or the last 512 keys a row as a token list
    const nch: usize = if (L.sliding) 1 else (rows + 511) / 512;
    const part: cuda.mla.Partials = .{ .po = try c.zeros(nch * rows * H * 512 * 4), .pm = try c.zeros(nch * rows * H * 4), .pl = try c.zeros(nch * rows * H * 4) };
    const scale: f32 = 1.0 / 16.0;
    if (L.sliding) {
        const tok_h = try gpa.alloc(i32, rows * 512);
        defer gpa.free(tok_h);
        const cnt_h = try gpa.alloc(i32, rows);
        defer gpa.free(cnt_h);
        @memset(tok_h, 0);
        for (0..rows) |r| {
            const first = if (r >= 511) r - 511 else 0;
            cnt_h[r] = @intCast(r + 1 - first);
            for (first..r + 1, 0..) |key, j| tok_h[r * 512 + j] = @intCast(key);
        }
        try hy4.sparseChunks(c.mla, c.s, qa, lc, try c.up(std.mem.sliceAsBytes(tok_h)), try c.up(std.mem.sliceAsBytes(cnt_h)), 512, part, rows, H, scale);
    } else try c.mla.chunks(c.s, qa, lc, pos, part, rows, H, nch, scale);
    const ol = try c.zeros(rows * H * 512 * 2);
    try k.mergeSink(c.s, part, try c.w(p ++ "learnable_sink_param", .{i}), ol, rows, H, nch);
    const o = try c.zeros(rows * H * 256 * 2);
    try k.expandV(c.s, ol, try c.up(wv_h), o, rows, H);
    const g = try c.zeros(rows * H * 256 * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "linear_gate.weight", .{i}), g, H * 256, rows, H * 256, hidden);
    try k.gateRows(c.s, o, g, rows * H * 256);
    const proj = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, o, H * 256, try c.w(p ++ "o_proj.weight", .{i}), proj, hidden, rows, hidden, H * 256);
    const out = try c.zeros(rows * hidden * 2);
    try k.rmsNorm(c.s, proj, hidden, try c.w(p ++ "post_proj_norm.weight", .{i}), out, hidden, rows, hidden, eps);
    return out;
}

/// gate and up of x [rows][k] into gu [rows][2 inter] fp32, then SwiGLU into bf16 [rows][inter].
pub fn swigluRows(c: *Ctx, x: u64, kdim: usize, gate: u64, upw: u64, rows: usize, inter: usize, limit: f32) !u64 {
    const gu = try c.zeros(rows * 2 * inter * 4);
    try c.dense.rows(c.s, x, kdim, gate, gu, 2 * inter, rows, inter, kdim);
    try c.dense.rows(c.s, x, kdim, upw, gu + inter * 4, 2 * inter, rows, inter, kdim);
    const act = try c.zeros(rows * inter * 2);
    try c.k.swiglu(c.s, gu, act, rows, inter, limit);
    return act;
}

/// Layer 0's dense MLP with its post_fc2_norm: x [rows][hidden] bf16 in and out.
fn denseMlp(c: *Ctx, x: u64, rows: usize) !u64 {
    const p = "model.layers.0.mlp.";
    const act = try swigluRows(c, x, hidden, try c.w(p ++ "gate_proj.weight", .{}), try c.w(p ++ "up_proj.weight", .{}), rows, dense_inter, 0);
    const down = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, act, dense_inter, try c.w(p ++ "down_proj.weight", .{}), down, hidden, rows, hidden, dense_inter);
    const out = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm(c.s, down, hidden, try c.w(p ++ "post_fc2_norm.weight", .{}), out, hidden, rows, hidden, eps);
    return out;
}

/// The latent MoE: routing on the device, the experts one at a time over their rows (gathered on the host).
fn moe(c: *Ctx, i: usize, x: u64, rows: usize) !u64 {
    const p = "model.layers.{d}.mlp.";
    const gpa = c.gpu.gpa;
    const logits = try c.zeros(rows * experts * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "gate.weight", .{i}), logits, experts, rows, experts, hidden);
    const ids = try c.zeros(rows * topk * 4);
    const wts = try c.zeros(rows * topk * 4);
    try c.k.route(c.s, logits, try c.w(p ++ "gate.e_score_correction_bias", .{i}), ids, wts, rows, experts, topk, 3.998);
    const xl_f = try c.zeros(rows * latent * 4);
    try c.dense.rows(c.s, x, hidden, try c.w(p ++ "routed_expert_down_proj.weight", .{i}), xl_f, latent, rows, latent, hidden);
    const xl = try c.zeros(rows * latent * 2);
    try c.k.castRows(c.s, xl_f, latent, xl, latent, rows, latent);
    try c.s.synchronize();
    const ids_h = try gpa.alloc(i32, rows * topk);
    defer gpa.free(ids_h);
    const wts_h = try gpa.alloc(f32, rows * topk);
    defer gpa.free(wts_h);
    const xl_h = try gpa.alloc(u16, rows * latent);
    defer gpa.free(xl_h);
    try cuda.memory.download(c.gpu.d, ids, std.mem.sliceAsBytes(ids_h));
    try cuda.memory.download(c.gpu.d, wts, std.mem.sliceAsBytes(wts_h));
    try cuda.memory.download(c.gpu.d, xl, std.mem.sliceAsBytes(xl_h));
    const acc = try gpa.alloc(f32, rows * latent);
    defer gpa.free(acc);
    @memset(acc, 0);
    const gate_up = try c.w(p ++ "experts.gate_up_proj", .{i});
    const down = try c.w(p ++ "experts.down_proj", .{i});
    const sel = try gpa.alloc(u32, rows * topk);
    defer gpa.free(sel);
    const xin = try gpa.alloc(u16, rows * topk * latent);
    defer gpa.free(xin);
    const y = try gpa.alloc(f32, rows * topk * latent);
    defer gpa.free(y);
    for (0..experts) |e| {
        var n: usize = 0;
        for (ids_h, 0..) |id, j| if (id == @as(i32, @intCast(e))) {
            sel[n] = @intCast(j);
            @memcpy(xin[n * latent ..][0..latent], xl_h[(j / topk) * latent ..][0..latent]);
            n += 1;
        };
        if (n == 0) continue;
        const xe = try c.up(std.mem.sliceAsBytes(xin[0 .. n * latent]));
        const gu = try c.zeros(n * 2 * latent * 4);
        try c.dense.rows(c.s, xe, latent, gate_up + e * 2 * latent * latent * 2, gu, 2 * latent, n, 2 * latent, latent);
        const act = try c.zeros(n * latent * 2);
        try c.k.swiglu(c.s, gu, act, n, latent, 10.0);
        const ye = try c.zeros(n * latent * 4);
        try c.dense.rows(c.s, act, latent, down + e * latent * latent * 2, ye, latent, n, latent, latent);
        try c.s.synchronize();
        try cuda.memory.download(c.gpu.d, ye, std.mem.sliceAsBytes(y[0 .. n * latent]));
        for (sel[0..n], 0..) |j, m| {
            const r = j / topk;
            for (0..latent) |d| acc[r * latent + d] += wts_h[j] * y[m * latent + d];
        }
    }
    const acc_d = try c.up(std.mem.sliceAsBytes(acc));
    const routed_in = try c.zeros(rows * latent * 2);
    try c.k.castRows(c.s, acc_d, latent, routed_in, latent, rows, latent);
    const routed = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, routed_in, latent, try c.w(p ++ "routed_expert_up_proj.weight", .{i}), routed, hidden, rows, hidden, latent);
    const sp = "model.layers.{d}.mlp.shared_experts.";
    const act = try swigluRows(c, x, hidden, try c.w(sp ++ "gate_proj.weight", .{i}), try c.w(sp ++ "up_proj.weight", .{i}), rows, latent, 0);
    const shared = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, act, latent, try c.w(sp ++ "down_proj.weight", .{i}), shared, hidden, rows, hidden, latent);
    try c.k.add32(c.s, routed, shared, rows * hidden);
    const out = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm(c.s, routed, hidden, try c.w(p ++ "combine_output_norm.weight", .{i}), out, hidden, rows, hidden, eps);
    return out;
}

/// Engram's residual added into the streams, its hash ids from engram.zig checked against the reference's first.
fn engramAdd(c: *Ctx, fwd: *core.checkpoint.Checkpoint, i: usize, x: u64, ids: []align(1) const i32, rows: usize) !void {
    const gpa = c.gpu.gpa;
    const p = "model.layers.{d}.engram_layer.";
    var nb: [64]u8 = undefined;
    const mult_t: []align(1) const u64 = std.mem.bytesAsSlice(u64, (try fwd.get(try std.fmt.bufPrint(&nb, "engram.{d}.mult", .{i}))).bytes);
    const mod_t: []align(1) const i64 = std.mem.bytesAsSlice(i64, (try fwd.get(try std.fmt.bufPrint(&nb, "engram.{d}.moduli", .{i}))).bytes);
    const want: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try fwd.get(try std.fmt.bufPrint(&nb, "engram.{d}.hash", .{i}))).bytes);
    const table: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try c.tensor(p ++ "lookup_table", .{i})).bytes);
    var moduli: [engram.heads]u32 = undefined;
    var offsets: [engram.heads]i64 = undefined;
    var off: i64 = 0;
    for (mod_t, 0..) |m, h| {
        moduli[h] = @intCast(m);
        offsets[h] = off;
        off += m;
    }
    var mult: [engram.max_ngram]u64 = undefined;
    for (&mult, 0..) |*m, j| m.* = mult_t[j];
    const pad: i32 = 120002;
    const hash = try gpa.alloc(i32, rows * engram.heads);
    defer gpa.free(hash);
    for (0..rows) |r| {
        var comp: [engram.max_ngram]u32 = undefined;
        for (0..engram.max_ngram) |o| {
            const tok: i32 = if (r >= o) ids[r - o] else pad;
            const look: usize = if (tok < table.len) @intCast(tok) else @intCast(pad);
            comp[o] = @intCast(table[look]);
        }
        var out: [engram.heads]u32 = undefined;
        engram.hash(comp, mult, &moduli, &out);
        for (out, 0..) |v, h| hash[r * engram.heads + h] = @intCast(v);
    }
    var diff: usize = 0;
    for (hash, want) |a, b| diff += @intFromBool(a != b);
    try check.expect(diff == 0, "hy4 engram hash ids: {d} of {d} differ from the reference", .{ diff, hash.len });
    check.pass("hy4 engram hash ids of layer {d}: {d} as the reference", .{ i, hash.len });
    const e = try c.zeros(rows * engram.heads * 96 * 2);
    const table16 = try c.w(p ++ "multi_head_embedding.embedding.weight", .{i});
    try c.k.engramRows(c.s, try c.up(std.mem.sliceAsBytes(hash)), try c.up(std.mem.sliceAsBytes(&offsets)), .{ .bf16 = table16 }, try c.wf32(p ++ "lookup_per_head_norms.{d}.weight", .{i}, engram.heads), eps32, e, rows);
    const ew = engram.heads * 96;
    const value = try c.zeros(rows * hidden * 4);
    try c.dense.rows(c.s, e, ew, try c.w(p ++ "value_proj.weight", .{i}), value, hidden, rows, hidden, ew);
    try c.k.bias(c.s, value, hidden, try c.w(p ++ "value_proj.bias", .{i}), rows, hidden);
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
    try c.k.engramMix(c.s, key, value, x, mix, eps32, res, rows, hidden);
    try c.k.add16(c.s, x, res, rows * streams * hidden);
}

/// Relative error of bf16 (or fp32) values against the reference's fp32.
pub fn relErr(c: *Ctx, p: u64, f32s: bool, want: []align(1) const f32) !f64 {
    try c.s.synchronize();
    const n = want.len;
    const bytes = try c.gpu.gpa.alloc(u8, n * @as(usize, if (f32s) 4 else 2));
    defer c.gpu.gpa.free(bytes);
    try cuda.memory.download(c.gpu.d, p, bytes);
    var dd: f64 = 0;
    var ww: f64 = 0;
    for (want, 0..) |wv, j| {
        const g: f32 = if (f32s) std.mem.bytesAsValue(f32, bytes[j * 4 ..][0..4]).* else bf16f(std.mem.readInt(u16, bytes[j * 2 ..][0..2], .little));
        const e = @as(f64, g) - wv;
        dd += e * e;
        ww += @as(f64, wv) * wv;
    }
    return @sqrt(dd / ww);
}

pub fn refF32(ref: *core.checkpoint.Checkpoint, comptime fmt: []const u8, args: anytype) ![]align(1) const f32 {
    var buf: [64]u8 = undefined;
    return std.mem.bytesAsSlice(f32, (try ref.get(try std.fmt.bufPrint(&buf, fmt, args))).bytes);
}

pub fn argmax(v: []align(1) const f32) usize {
    var best: usize = 0;
    for (v, 0..) |x, j| if (x > v[best]) {
        best = j;
    };
    return best;
}

/// The whole tiny model on the GPU: streams after each layer, the final norm and the logits against the reference.
fn forwardCheck(c: *Ctx, ref: *core.checkpoint.Checkpoint, fwd: *core.checkpoint.Checkpoint) !void {
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
    for (0..layers.len) |i| {
        defer c.reset();
        if (i == engram_layer) try engramAdd(c, fwd, i, x, ids, rows);
        const h = try c.zeros(rows * hidden * 2);
        const post = try c.zeros(rows * streams * 4);
        try c.k.hcPre(c.s, x, try c.hc(pre ++ "hc_attn_layer.hc_pre.hc", .{i}), h, post, rows, hidden, eps, 1e-6, 2.0);
        const hn = try c.zeros(rows * hidden * 2);
        try c.k.rmsNorm16(c.s, h, hidden, try c.w(pre ++ "input_layernorm.weight", .{i}), hn, hidden, rows, hidden, eps);
        try c.k.hcPost(c.s, x, try attend(c, i, hn, rows), post, rows, hidden);
        const h2 = try c.zeros(rows * hidden * 2);
        const post2 = try c.zeros(rows * streams * 4);
        try c.k.hcPre(c.s, x, try c.hc(pre ++ "hc_mlp_layer.hc_pre.hc", .{i}), h2, post2, rows, hidden, eps, 1e-6, 2.0);
        const hn2 = try c.zeros(rows * hidden * 2);
        try c.k.rmsNorm16(c.s, h2, hidden, try c.w(pre ++ "post_attention_layernorm.weight", .{i}), hn2, hidden, rows, hidden, eps);
        try c.k.hcPost(c.s, x, if (i == 0) try denseMlp(c, hn2, rows) else try moe(c, i, hn2, rows), post2, rows, hidden);
        if (i + 1 < layers.len) {
            const rel = try relErr(c, x, false, try refF32(ref, "hidden.{d}", .{i + 1}));
            try check.expect(rel < 3e-2, "hy4 forward: streams after layer {d}: rel err {e}", .{ i, rel });
            check.pass("hy4 forward: streams after layer {d}: rel err {e:.3}", .{ i, rel });
        }
    }
    defer c.reset();
    const h = try c.zeros(rows * hidden * 2);
    try c.k.hcHead(c.s, x, try c.hc("model.hc_head.hc_head", .{}), h, rows, hidden, eps, 1e-6);
    const hn = try c.zeros(rows * hidden * 2);
    try c.k.rmsNorm16(c.s, h, hidden, try c.w("model.norm.weight", .{}), hn, hidden, rows, hidden, eps);
    const rel = try relErr(c, hn, false, try refF32(ref, "hidden.{d}", .{layers.len}));
    try check.expect(rel < 3e-2, "hy4 forward: final norm: rel err {e}", .{rel});
    check.pass("hy4 forward: final norm after the head: rel err {e:.3}", .{rel});
    // lm_head is fp32 in the checkpoint; dense_rows reads bf16 weights
    const lm = (try c.tensor("lm_head.weight", .{})).bytes;
    const vocab = lm.len / 4 / hidden;
    const lm16 = try gpa.alloc(u16, vocab * hidden);
    defer gpa.free(lm16);
    for (lm16, 0..) |*o, j| o.* = tobf16(std.mem.bytesAsValue(f32, lm[j * 4 ..][0..4]).*);
    const logits = try c.zeros(rows * vocab * 4);
    try c.dense.rows(c.s, hn, hidden, try c.up(std.mem.sliceAsBytes(lm16)), logits, vocab, rows, vocab, hidden);
    const want = try refF32(ref, "logits", .{});
    const lrel = try relErr(c, logits, true, want);
    const got = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(got);
    try cuda.memory.download(c.gpu.d, logits, std.mem.sliceAsBytes(got));
    var same: usize = 0;
    for (0..rows) |r| same += @intFromBool(std.mem.indexOfMax(f32, got[r * vocab ..][0..vocab]) == argmax(want[r * vocab ..][0..vocab]));
    try check.expect(lrel < 3e-2, "hy4 forward: logits: rel err {e}, argmax {d}/{d}", .{ lrel, same, rows });
    check.pass("hy4 forward: logits: rel err {e:.3}, argmax {d}/{d} as the reference", .{ lrel, same, rows });
}

/// The attention modules alone on the reference's own inputs.
fn attnCheck(c: *Ctx, ref: *core.checkpoint.Checkpoint) !void {
    const rows = 700;
    for (layers, 0..) |L, i| {
        defer c.reset();
        var buf: [16]u8 = undefined;
        const x = try c.up((try ref.get(try std.fmt.bufPrint(&buf, "x.{d}", .{i}))).bytes);
        const out = try attend(c, i, x, rows);
        const rel = try relErr(c, out, false, try refF32(ref, "out.{d}", .{i}));
        const kind = if (L.sliding) "sliding" else "sparse";
        try check.expect(rel < 1e-2, "hy4 attention layer {d} ({s}, {d} heads): rel err {e}", .{ i, kind, L.heads, rel });
        check.pass("hy4 attention layer {d} ({s}, {d} heads): rel err {e:.3}", .{ i, kind, L.heads, rel });
    }
}

/// The hy4- commands: hy4-attn-check TINY ATTN and hy4-forward-check TINY FWD (the tiny checkpoint, then dumps).
pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const []const u8) !void {
    if (std.mem.eql(u8, cmd, "hy4-real-check")) return @import("hy4_real_check.zig").run(gpu, rest);
    const attn = std.mem.eql(u8, cmd, "hy4-attn-check");
    if (!(attn or std.mem.eql(u8, cmd, "hy4-forward-check")) or rest.len != 2) {
        std.debug.print("usage: hy4-attn-check TINY ATTN | hy4-forward-check TINY FWD   a tiny Hy4-flash against the reference\n", .{});
        return error.UnknownCommand;
    }
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var k = try hy4.Kernels.load(gpu.d);
    defer k.unload();
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();
    var dense = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer dense.unload();
    var ck = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, rest[0]);
    defer ck.close();
    var dump = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, rest[1]);
    defer dump.close();
    var c: Ctx = .{ .gpu = gpu, .s = s, .k = &k, .mla = &mla, .dense = &dense, .ck = &ck };
    defer c.deinit();
    if (attn) return attnCheck(&c, &dump);
    // the reference's hidden streams and logits sit beside the tiny checkpoint
    var ref = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, rest[0]);
    defer ref.close();
    try ref.add(rest[0], "reference.safetensors");
    return forwardCheck(&c, &ref, &dump);
}
