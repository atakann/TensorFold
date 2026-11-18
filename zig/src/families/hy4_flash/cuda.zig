//! Hy4-flash on CUDA: its kernel image and mla.cu, launched for attention, hyper-connections and Engram.
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Args = cuda.launch.Args;
const Function = cuda.Function;

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// hy4_flash.cu's kernels.
pub const Kernels = struct {
    mod: cuda.Module,
    absorb: Function,
    expand: Function,
    merge: Function,
    rmsnorm: Function,
    cast: Function,
    gate: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_flash);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .absorb = try mod.function("tf_hy4_absorb_q_192"),
            .expand = try mod.function("tf_hy4_expand_v_256"),
            .merge = try mod.function("tf_hy4_merge_sink"),
            .rmsnorm = try mod.function("tf_hy4_rmsnorm"),
            .cast = try mod.function("tf_hy4_cast"),
            .gate = try mod.function("tf_hy4_gate"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// q [rows][heads][256] (nope 192 first) and wk [heads][192][512] into qa [rows][heads][576]'s latent columns.
    pub fn absorbQ(k: *const Kernels, s: Stream, q: u64, wk: u64, qa: u64, rows: usize, heads: usize) !void {
        const rb: usize = if (rows >= 64) 64 else rows;
        var a: Args = .{};
        inline for (.{ q, wk, qa, i64of(rows), i64of(heads), i64of(rb), @as(i64, 256), @as(i64, 576) }) |v| a.add(v);
        try cuda.launch.launch(k.absorb, .{ .grid = .{ .x = @intCast(heads), .y = 512 / 32, .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// ol [rows][heads][512] and wv [heads][256][512] into out [rows][heads][256].
    pub fn expandV(k: *const Kernels, s: Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize) !void {
        const rb: usize = if (rows >= 64) 64 else rows;
        var a: Args = .{};
        inline for (.{ ol, wv, out, i64of(rows), i64of(heads), i64of(rb) }) |v| a.add(v);
        try cuda.launch.launch(k.expand, .{ .grid = .{ .x = @intCast(heads), .y = 256 / 16, .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The chunks' partials merged with the per-head sink (fp32 [heads]) into out [rows][heads][512] bf16.
    pub fn mergeSink(k: *const Kernels, s: Stream, p: cuda.mla.Partials, sink: u64, out: u64, rows: usize, heads: usize, nch: usize) !void {
        var a: Args = .{};
        inline for (.{ p.po, p.pm, p.pl, sink, out, i64of(rows), i64of(heads), @as(c_int, @intCast(nch)) }) |v| a.add(v);
        try cuda.launch.launch(k.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// out [rows] rows os apart = bf16 RMS norm of fp32 x rows xs apart, n wide, times the bf16 weight.
    pub fn rmsNorm(k: *const Kernels, s: Stream, x: u64, xs: usize, w: u64, out: u64, os: usize, rows: usize, n: usize, eps: f32) !void {
        var a: Args = .{};
        inline for (.{ x, i64of(xs), w, out, i64of(os), i64of(n), eps }) |v| a.add(v);
        try cuda.launch.launch(k.rmsnorm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn castRows(k: *const Kernels, s: Stream, x: u64, xs: usize, out: u64, os: usize, rows: usize, n: usize) !void {
        var a: Args = .{};
        inline for (.{ x, i64of(xs), out, i64of(os), i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.cast, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn gateRows(k: *const Kernels, s: Stream, o: u64, g: u64, n: usize) !void {
        var a: Args = .{};
        inline for (.{ o, g, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.gate, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// mla.cu's chunk tiles over token lists (`tok` [rows][w] i32, `cnt` [rows]): sliding layers' last 512 keys.
pub fn sparseChunks(m: *const cuda.mla.Mla, s: Stream, qa: u64, cache: u64, tok: u64, cnt: u64, w: usize, p: cuda.mla.Partials, rows: usize, heads: usize, scale: f32) !void {
    const nch = (w + 511) / 512;
    var a: Args = .{};
    inline for (.{ qa, cache, @as(u64, 0), tok, cnt, p.po, p.pm, p.pl, i64of(rows), i64of(heads), i64of(w), scale, @as(c_int, 1), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
    try cuda.launch.launch(m.chunks_fn, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, cuda.mla.heads_tile), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = cuda.mla.chunksSmem(cuda.mla.heads_tile) }, s, &a);
}
