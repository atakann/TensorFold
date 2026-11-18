//! hy4-attn-check (GPU): a tiny Hy4-flash checkpoint's attention modules against the reference's outputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const hy4 = @import("hy4_flash");

const hidden = 3072;
const rows = 700;
const eps: f32 = 1e-5;

const Layer = struct { heads: usize, theta: f64, sliding: bool };
const layers = [_]Layer{
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 32, .theta = 1e7, .sliding = false },
    .{ .heads = 64, .theta = 1e4, .sliding = true },
};

const Bufs = std.ArrayList(cuda.DeviceBuffer);

/// The check's stream: buffers are cleared on it, so a clear never lands after a launch that wrote the buffer.
var clear_on: ?cuda.Stream = null;

fn keep(gpu: check.Gpu, bufs: *Bufs, b: cuda.DeviceBuffer) !u64 {
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

fn zeros(gpu: check.Gpu, bufs: *Bufs, n: usize) !u64 {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer b.free();
    try b.fill8(0, clear_on.?.handle);
    return keep(gpu, bufs, b);
}

fn up(gpu: check.Gpu, bufs: *Bufs, bytes: []const u8) !u64 {
    return keep(gpu, bufs, try cuda.DeviceBuffer.fromHost(gpu.d, bytes));
}

fn get(ck: *core.checkpoint.Checkpoint, comptime fmt: []const u8, args: anytype) !core.checkpoint.Tensor {
    var buf: [200]u8 = undefined;
    return ck.get(try std.fmt.bufPrint(&buf, fmt, args));
}

fn bf16f(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}


/// One layer's attention module on the GPU: projections by dense_rows, MLA by mla.cu and hy4_flash.cu.
fn layer(gpu: check.Gpu, s: cuda.Stream, k: *const hy4.Kernels, mla: *const cuda.mla.Mla, dense: *const cuda.dense_rows.DenseRows, ck: *core.checkpoint.Checkpoint, ref: *core.checkpoint.Checkpoint, i: usize) !void {
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const L = layers[i];
    const H = L.heads;
    const p = "model.layers.{d}.self_attn.";
    const x = try up(gpu, &bufs, (try get(ref, "x.{d}", .{i})).bytes);
    const w = struct {
        fn of(g: check.Gpu, b: *Bufs, c: *core.checkpoint.Checkpoint, comptime leaf: []const u8, li: usize) !u64 {
            return up(g, b, (try get(c, p ++ leaf, .{li})).bytes);
        }
    }.of;
    // q: q_a, its norm, q_b; the rope on the last 64 of each head into qa's columns 512..
    const qa_f = try zeros(gpu, &bufs, rows * 1024 * 4);
    try dense.rows(s, x, hidden, try w(gpu, &bufs, ck, "q_a_proj.weight", i), qa_f, 1024, rows, 1024, hidden);
    const qr = try zeros(gpu, &bufs, rows * 1024 * 2);
    try k.rmsNorm(s, qa_f, 1024, try w(gpu, &bufs, ck, "q_a_layernorm.weight", i), qr, 1024, rows, 1024, eps);
    const q_f = try zeros(gpu, &bufs, rows * H * 256 * 4);
    try dense.rows(s, qr, 1024, try w(gpu, &bufs, ck, "q_b_proj.weight", i), q_f, H * 256, rows, H * 256, 1024);
    const q = try zeros(gpu, &bufs, rows * H * 256 * 2);
    try k.castRows(s, q_f, H * 256, q, H * 256, rows, H * 256);
    const freqs = cuda.mla.ropeFreqs(L.theta);
    const inv_freq = try up(gpu, &bufs, std.mem.sliceAsBytes(&freqs));
    const pos = try zeros(gpu, &bufs, 4);
    const qa = try zeros(gpu, &bufs, rows * H * 576 * 2);
    try mla.rope(s, .{ .x = q + 192 * 2, .x_row = H * 256, .x_head = 256, .out = qa + 512 * 2, .o_row = H * 576, .o_head = 576 }, rows, H, pos, inv_freq, 1.0);
    // the latent cache rows: kv_a's 512 normed, then its 64 roped
    const kv_f = try zeros(gpu, &bufs, rows * 576 * 4);
    try dense.rows(s, x, hidden, try w(gpu, &bufs, ck, "kv_a_proj_with_mqa.weight", i), kv_f, 576, rows, 576, hidden);
    const lc = try zeros(gpu, &bufs, rows * 576 * 2);
    try k.rmsNorm(s, kv_f, 576, try w(gpu, &bufs, ck, "kv_a_layernorm.weight", i), lc, 576, rows, 512, eps);
    try k.castRows(s, kv_f + 512 * 4, 576, lc + 512 * 2, 576, rows, 64);
    try mla.rope(s, .{ .x = lc + 512 * 2, .x_row = 576, .x_head = 0, .out = lc + 512 * 2, .o_row = 576, .o_head = 0 }, rows, 1, pos, inv_freq, 1.0);
    // kv_b split into wk [H][192][512] and wv [H][256][512]
    const kvb = (try get(ck, p ++ "kv_b_proj.weight", .{i})).bytes;
    const wk_h = try gpu.gpa.alloc(u8, H * 192 * 512 * 2);
    defer gpu.gpa.free(wk_h);
    const wv_h = try gpu.gpa.alloc(u8, H * 256 * 512 * 2);
    defer gpu.gpa.free(wv_h);
    for (0..H) |h| {
        @memcpy(wk_h[h * 192 * 1024 ..][0 .. 192 * 1024], kvb[h * 448 * 1024 ..][0 .. 192 * 1024]);
        @memcpy(wv_h[h * 256 * 1024 ..][0 .. 256 * 1024], kvb[(h * 448 + 192) * 1024 ..][0 .. 256 * 1024]);
    }
    try k.absorbQ(s, q, try up(gpu, &bufs, wk_h), qa, rows, H);
    // attention: dense causal chunks, or the last 512 keys a row as a token list
    const nch: usize = if (L.sliding) 1 else (rows + 511) / 512;
    const part: cuda.mla.Partials = .{ .po = try zeros(gpu, &bufs, nch * rows * H * 512 * 4), .pm = try zeros(gpu, &bufs, nch * rows * H * 4), .pl = try zeros(gpu, &bufs, nch * rows * H * 4) };
    const scale: f32 = 1.0 / 16.0;
    if (L.sliding) {
        const tok_h = try gpu.gpa.alloc(i32, rows * 512);
        defer gpu.gpa.free(tok_h);
        const cnt_h = try gpu.gpa.alloc(i32, rows);
        defer gpu.gpa.free(cnt_h);
        @memset(tok_h, 0);
        for (0..rows) |r| {
            const first = if (r >= 511) r - 511 else 0;
            cnt_h[r] = @intCast(r + 1 - first);
            for (first..r + 1, 0..) |key, j| tok_h[r * 512 + j] = @intCast(key);
        }
        try hy4.sparseChunks(mla, s, qa, lc, try up(gpu, &bufs, std.mem.sliceAsBytes(tok_h)), try up(gpu, &bufs, std.mem.sliceAsBytes(cnt_h)), 512, part, rows, H, scale);
    } else try mla.chunks(s, qa, lc, pos, part, rows, H, nch, scale);
    const ol = try zeros(gpu, &bufs, rows * H * 512 * 2);
    try k.mergeSink(s, part, try w(gpu, &bufs, ck, "learnable_sink_param", i), ol, rows, H, nch);
    const o = try zeros(gpu, &bufs, rows * H * 256 * 2);
    try k.expandV(s, ol, try up(gpu, &bufs, wv_h), o, rows, H);
    const g = try zeros(gpu, &bufs, rows * H * 256 * 4);
    try dense.rows(s, x, hidden, try w(gpu, &bufs, ck, "linear_gate.weight", i), g, H * 256, rows, H * 256, hidden);
    try k.gateRows(s, o, g, rows * H * 256);
    const proj = try zeros(gpu, &bufs, rows * hidden * 4);
    try dense.rows(s, o, H * 256, try w(gpu, &bufs, ck, "o_proj.weight", i), proj, hidden, rows, hidden, H * 256);
    const out = try zeros(gpu, &bufs, rows * hidden * 2);
    try k.rmsNorm(s, proj, hidden, try w(gpu, &bufs, ck, "post_proj_norm.weight", i), out, hidden, rows, hidden, eps);
    try s.synchronize();
    const got = try gpu.gpa.alloc(u16, rows * hidden);
    defer gpu.gpa.free(got);
    try cuda.memory.download(gpu.d, out, std.mem.sliceAsBytes(got));
    const want: []align(1) const f32 = std.mem.bytesAsSlice(f32, (try get(ref, "out.{d}", .{i})).bytes);
    var dd: f64 = 0;
    var ww: f64 = 0;
    for (got, want) |gv, wv| {
        const e = @as(f64, bf16f(gv)) - wv;
        dd += e * e;
        ww += @as(f64, wv) * wv;
    }
    const rel = @sqrt(dd / ww);
    try check.expect(rel < 1e-2, "hy4 attention layer {d} ({s}, {d} heads): rel err {e}", .{ i, if (L.sliding) "sliding" else "sparse", H, rel });
    check.pass("hy4 attention layer {d} ({s}, {d} heads): rel err {e:.3}", .{ i, if (L.sliding) "sliding" else "sparse", H, rel });
}

/// The hy4- commands: hy4-attn-check TINY REF (the tiny checkpoint, the reference's per-layer inputs and outputs).
pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const []const u8) !void {
    if (!std.mem.eql(u8, cmd, "hy4-attn-check") or rest.len != 2) {
        std.debug.print("usage: hy4-attn-check TINY REF   a tiny Hy4-flash checkpoint's attention against the reference\n", .{});
        return error.UnknownCommand;
    }
    return attnCheck(gpu, rest[0], rest[1]);
}

fn attnCheck(gpu: check.Gpu, tiny: []const u8, ref_dir: []const u8) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    clear_on = s;
    var k = try hy4.Kernels.load(gpu.d);
    defer k.unload();
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();
    var dense = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer dense.unload();
    var ck = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, tiny);
    defer ck.close();
    var ref = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, ref_dir);
    defer ref.close();
    for (0..layers.len) |i| try layer(gpu, s, &k, &mla, &dense, &ck, &ref, i);
}
