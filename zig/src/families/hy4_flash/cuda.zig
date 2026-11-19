//! Hy4-flash on CUDA: its kernel image and mla.cu, launched for attention, hyper-connections and Engram.
const std = @import("std");
const cuda = @import("cuda");

/// The host side of Engram's hashes.
pub const engram = @import("engram.zig");
/// The model on one GPU: weights, latent caches, a chunk of rows a step.
pub const engine = @import("engine.zig");
/// The DFly drafter on the engine's caches.
pub const dfly = @import("dfly.zig");
pub const config = @import("config.zig");

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
    norm16: Function,
    bias_fn: Function,
    add16_fn: Function,
    add32_fn: Function,
    addb_fn: Function,
    own_fn: Function,
    window_fn: Function,
    put_fn: Function,
    fuse_fn: Function,
    amax_local: Function,
    amax_pick: Function,
    idx_pool: Function,
    idx_scores: Function,
    idx_tokens: Function,
    dfly_tokens: Function,
    swiglu_fn: Function,
    route_fn: Function,
    hc_pre: Function,
    hc_post: Function,
    hc_head: Function,
    engram_rows: Function,
    engram_mix: Function,

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
            .norm16 = try mod.function("tf_hy4_rmsnorm16"),
            .bias_fn = try mod.function("tf_hy4_bias"),
            .add16_fn = try mod.function("tf_hy4_add16"),
            .add32_fn = try mod.function("tf_hy4_add32"),
            .addb_fn = try mod.function("tf_hy4_addb"),
            .own_fn = try mod.function("tf_hy4_own"),
            .window_fn = try mod.function("tf_hy4_window"),
            .put_fn = try mod.function("tf_hy4_put"),
            .fuse_fn = try mod.function("tf_hy4_fuse"),
            .amax_local = try mod.function("tf_hy4_argmax_local"),
            .amax_pick = try mod.function("tf_hy4_argmax_pick"),
            .idx_pool = try mod.function("tf_hy4_idx_pool"),
            .idx_scores = try mod.function("tf_hy4_idx_scores"),
            .idx_tokens = try mod.function("tf_hy4_idx_tokens"),
            .dfly_tokens = try mod.function("tf_hy4_dfly_tokens"),
            .swiglu_fn = try mod.function("tf_hy4_swiglu"),
            .route_fn = try mod.function("tf_hy4_route"),
            .hc_pre = try mod.function("tf_hy4_hc_pre"),
            .hc_post = try mod.function("tf_hy4_hc_post"),
            .hc_head = try mod.function("tf_hy4_hc_head"),
            .engram_rows = try mod.function("tf_hy4_engram_rows"),
            .engram_mix = try mod.function("tf_hy4_engram_mix"),
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


    /// rmsNorm on bf16 rows: out = w bf16(x rsqrt(mean(x^2) + eps)).
    pub fn rmsNorm16(k: *const Kernels, s: Stream, x: u64, xs: usize, w: u64, out: u64, os: usize, rows: usize, n: usize, eps: f32) !void {
        var a: Args = .{};
        inline for (.{ x, i64of(xs), w, out, i64of(os), i64of(n), eps }) |v| a.add(v);
        try cuda.launch.launch(k.norm16, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out [rows] fp32 rows os apart += bias (bf16 [n]).
    pub fn bias(k: *const Kernels, s: Stream, out: u64, os: usize, b: u64, rows: usize, n: usize) !void {
        var a: Args = .{};
        inline for (.{ out, i64of(os), b, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.bias_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// x bf16 += f fp32 over n values.
    pub fn add16(k: *const Kernels, s: Stream, x: u64, f: u64, n: usize) !void {
        var a: Args = .{};
        inline for (.{ x, f, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.add16_fn, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// a fp32 += b fp32 over n values.
    pub fn add32(k: *const Kernels, s: Stream, x: u64, f: u64, n: usize) !void {
        var a: Args = .{};
        inline for (.{ x, f, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.add32_fn, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Expert parallel picks: this rank's experts lo..lo+count renumbered from 0, the others to `count` at weight 0.
    pub fn own(k: *const Kernels, s: Stream, picks: u64, wts: u64, n: usize, lo: usize, count: usize) !void {
        var a: Args = .{};
        inline for (.{ picks, wts, i64of(n), @as(c_int, @intCast(lo)), @as(c_int, @intCast(count)) }) |v| a.add(v);
        try cuda.launch.launch(k.own_fn, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Each row's last `win` keys (rows at device pos[0] + r) as a token list [rows][win] and counts [rows].
    pub fn window(k: *const Kernels, s: Stream, pos: u64, tok: u64, cnt: u64, rows: usize, win: usize) !void {
        var a: Args = .{};
        inline for (.{ pos, tok, cnt, i64of(win) }) |v| a.add(v);
        try cuda.launch.launch(k.window_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// rows [n][w] bf16 into `cache` at rows pos[0].. (device position).
    pub fn put(k: *const Kernels, s: Stream, cache: u64, rows: u64, pos: u64, n: usize, w: usize) !void {
        var a: Args = .{};
        inline for (.{ cache, rows, pos, i64of(w) }) |v| a.add(v);
        try cuda.launch.launch(k.put_fn, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// x bf16 = x + gate fused (DFly's fusion of the anchor row with the target's hidden state).
    pub fn fuse(k: *const Kernels, s: Stream, x: u64, fused: u64, gate: u64, n: usize) !void {
        var a: Args = .{};
        inline for (.{ x, fused, gate, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.fuse_fn, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Each row's first maximum of a vocabulary slice's n logits as (value, index + offset) pairs [rows].
    pub fn argmaxLocal(k: *const Kernels, s: Stream, logits: u64, rows: usize, n: usize, offset: usize, out: u64) !void {
        var a: Args = .{};
        inline for (.{ logits, i64of(n), i64of(offset), out }) |v| a.add(v);
        try cuda.launch.launch(k.amax_local, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// Each row's token over the ranks' pairs [world][rows], int32 into `ids`.
    pub fn argmaxPick(k: *const Kernels, s: Stream, all: u64, world: usize, rows: usize, ids: u64) !void {
        var a: Args = .{};
        inline for (.{ all, @as(c_int, @intCast(world)), i64of(rows), ids }) |v| a.add(v);
        try cuda.launch.launch(k.amax_pick, .{ .grid = .{ .x = cdiv(rows, 128) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The indexer's group keys: each of `rows` rows (at pos[0]..) that closes its group pools it into gk.
    pub fn idxPool(k: *const Kernels, s: Stream, tk: u64, gk: u64, pos: u64, w: IdxPool, rows: usize, eps: f32) !void {
        var a: Args = .{};
        inline for (.{ tk, gk, pos, w.wv, w.wg, w.ape, w.norm, w.inv_freq, eps }) |v| a.add(v);
        try cuda.launch.launch(k.idx_pool, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The indexer's scores [rows][np] over the group keys (own group +inf, later ones -inf).
    pub fn idxScores(k: *const Kernels, s: Stream, q: u64, w: u64, gk: u64, pos: u64, out: u64, rows: usize, np: usize, scale: f32) !void {
        var a: Args = .{};
        inline for (.{ q, w, gk, pos, out, i64of(np), scale }) |v| a.add(v);
        try cuda.launch.launch(k.idx_scores, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(np, 32) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The recalled groups (int64 [rows][512]) as token lists [rows][2048] and counts [rows].
    pub fn idxTokens(k: *const Kernels, s: Stream, sel: u64, pos: u64, tok: u64, cnt: u64, rows: usize) !void {
        var a: Args = .{};
        inline for (.{ sel, pos, tok, cnt }) |v| a.add(v);
        try cuda.launch.launch(k.idx_tokens, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, s, &a);
    }

    /// DFly's token lists [rows][w]: snapshot slot slot[0]'s keys before pos[0], then the block's rows causally.
    pub fn dflyTokens(k: *const Kernels, s: Stream, snap_tok: u64, snap_cnt: u64, slot: u64, pos: u64, tok: u64, cnt: u64, rows: usize, w: usize) !void {
        var a: Args = .{};
        inline for (.{ snap_tok, snap_cnt, slot, pos, tok, cnt, i64of(w) }) |v| a.add(v);
        try cuda.launch.launch(k.dfly_tokens, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// a fp32 += b bf16 over n values.
    pub fn addb(k: *const Kernels, s: Stream, x: u64, b: u64, n: usize) !void {
        var a: Args = .{};
        inline for (.{ x, b, i64of(n) }) |v| a.add(v);
        try cuda.launch.launch(k.addb_fn, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out [rows][I] bf16 = silu(gate) up from gu [rows][2I] fp32 (gate first), clamped at limit (0: none).
    pub fn swiglu(k: *const Kernels, s: Stream, gu: u64, out: u64, rows: usize, inter: usize, limit: f32) !void {
        var a: Args = .{};
        inline for (.{ gu, out, i64of(inter), limit }) |v| a.add(v);
        try cuda.launch.launch(k.swiglu_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Sigmoid routing: ids i32 and weights fp32 [rows][topk] from logits fp32 [rows][experts] and the fp32 bias.
    pub fn route(k: *const Kernels, s: Stream, logits: u64, b: u64, ids: u64, wts: u64, rows: usize, experts: usize, topk: usize, scale: f32) !void {
        var a: Args = .{};
        inline for (.{ logits, b, ids, wts, i64of(rows), @as(c_int, @intCast(experts)), @as(c_int, @intCast(topk)), scale }) |v| a.add(v);
        try cuda.launch.launch(k.route_fn, .{ .grid = .{ .x = cdiv(rows, 128) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The hyper-connection's input side: out [rows][d] bf16 and post [rows][4] fp32 from the 4 streams.
    pub fn hcPre(k: *const Kernels, s: Stream, x: u64, w: Hc, out: u64, post: u64, rows: usize, d: usize, eps: f32, hc_eps: f32, magnitude: f32) !void {
        var a: Args = .{};
        inline for (.{ x, w.fn_, w.base, w.scale, out, post, i64of(d), eps, hc_eps, magnitude }) |v| a.add(v);
        try cuda.launch.launch(k.hc_pre, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The streams += post times the sublayer's output f [rows][d] bf16.
    pub fn hcPost(k: *const Kernels, s: Stream, x: u64, f: u64, post: u64, rows: usize, d: usize) !void {
        var a: Args = .{};
        inline for (.{ x, f, post, i64of(d) }) |v| a.add(v);
        try cuda.launch.launch(k.hc_post, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The head's mix of the 4 streams into out [rows][d] bf16.
    pub fn hcHead(k: *const Kernels, s: Stream, x: u64, w: Hc, out: u64, rows: usize, d: usize, eps: f32, hc_eps: f32) !void {
        var a: Args = .{};
        inline for (.{ x, w.fn_, w.base, w.scale, out, i64of(d), eps, hc_eps }) |v| a.add(v);
        try cuda.launch.launch(k.hc_head, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Engram's normed rows e [rows][16][96] bf16 from hash ids, an fp8 table with row scales or a bf16 table.
    pub fn engramRows(k: *const Kernels, s: Stream, hash: u64, offsets: u64, t: EngramTable, norm_w: u64, eps: f32, e: u64, rows: usize) !void {
        var a: Args = .{};
        inline for (.{ hash, offsets, t.fp8, t.scales, t.bf16, norm_w, eps, e, i64of(rows) }) |v| a.add(v);
        try cuda.launch.launch(k.engram_rows, .{ .grid = .{ .x = @intCast(rows), .y = 16 }, .block = .{ .x = 96 } }, s, &a);
    }

    /// Engram's gated residual out [rows][4][d] fp32 from key [rows][4][d], value [rows][d] and the streams.
    pub fn engramMix(k: *const Kernels, s: Stream, key: u64, value: u64, x: u64, w: EngramMix, eps: f32, out: u64, rows: usize, d: usize) !void {
        var a: Args = .{};
        inline for (.{ key, value, x, w.norm_key, w.norm_query, w.gamma, w.hc_scale, eps, out, i64of(d) }) |v| a.add(v);
        try cuda.launch.launch(k.engram_mix, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// The indexer's pooling weights: value and gate projections and intra_group_ape (fp32), the key norm, rope freqs.
pub const IdxPool = struct { wv: u64, wg: u64, ape: u64, norm: u64, inv_freq: u64 };
/// A hyper-connection's fp32 weights: fn [m][4 d], base [m], scale (2 values, the head's 1).
pub const Hc = struct { fn_: u64, base: u64, scale: u64 };
/// An Engram table: fp8 rows with bf16 row scales, or bf16 rows (fp8 and scales 0).
pub const EngramTable = struct { fp8: u64 = 0, scales: u64 = 0, bf16: u64 = 0 };
/// Engram's per-stream norms and scales as fp32: norm_key, norm_query and hc_scale [4][d], gamma [d].
pub const EngramMix = struct { norm_key: u64, norm_query: u64, gamma: u64, hc_scale: u64 };

/// mla.cu's chunk tiles over token lists (`tok` [rows][w] i32, `cnt` [rows]): sliding layers' last 512 keys.
pub fn sparseChunks(m: *const cuda.mla.Mla, s: Stream, qa: u64, cache: u64, tok: u64, cnt: u64, w: usize, p: cuda.mla.Partials, rows: usize, heads: usize, scale: f32) !void {
    const nch = (w + 511) / 512;
    var a: Args = .{};
    inline for (.{ qa, cache, @as(u64, 0), tok, cnt, p.po, p.pm, p.pl, i64of(rows), i64of(heads), i64of(w), scale, @as(c_int, 1), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
    try cuda.launch.launch(m.chunks_fn, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, cuda.mla.heads_tile), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = cuda.mla.chunksSmem(cuda.mla.heads_tile) }, s, &a);
}
