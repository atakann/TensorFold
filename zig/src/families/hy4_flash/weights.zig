//! Hy4-flash's weights on the device: the checkpoint's tensors cut to this rank and laid out for the engine.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("cuda.zig");
const config = @import("config.zig");
const engram = @import("engram.zig");

const X = cuda.experts.Nvfp4Experts;
const Tensor = core.checkpoint.Tensor;
const Bufs = std.ArrayList(cuda.DeviceBuffer);
const sparse_theta: f64 = 1e7;
const sliding_theta: f64 = 1e4;

/// A bf16 projection in the lane matmul's layout (rows padded to 128).
pub const Lin = cuda.qmmf.Weight;
pub const Attn = struct { q_a: Lin, q_a_norm: u64, q_b: Lin, kv_a: Lin, kv_a_norm: u64, wk: u64, wv: u64, sink: u64, gate: Lin, o: Lin, post_norm: u64, inv_freq: u64, heads: usize, sliding: bool, idx: ?Indexer = null };
/// A sparse layer's compressed indexer (replicated on every rank) and its caches: token keys, pooled group keys.
pub const Indexer = struct { q: Lin, hw: Lin, tk: Lin, tk_norm: u64, wv: u64, wg: u64, ape: u64, k_norm: u64, tk_cache: u64, gk_cache: u64 };
pub const Moe = struct { router: Lin, bias: u64, down: Lin, up: Lin, routed: cuda.experts.Layer, shared: cuda.experts.Layer, norm: u64 };
pub const Mlp = union(enum) { dense: struct { t: cuda.experts.Layer, norm: u64 }, moe: Moe };
pub const Engram = struct { ids: []align(1) const i32, hash: u64, offsets: u64, moduli: [engram.heads]u32, mult: [engram.max_ngram]u64, table: hy4.EngramTable, row_norm: u64, value: u64, value_bias: u64, key: [4]u64, key_bias: [4]u64, mix: hy4.EngramMix };
pub const Layer = struct { attn: Attn, hc_attn: hy4.Hc, hc_ffn: hy4.Hc, in_norm: u64, post_norm: u64, mlp: Mlp, engram: ?Engram };

/// Device buffers that live as long as the model, and the checkpoint they came from.
pub const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    bufs: *Bufs,
    rank: usize = 0,
    world: usize = 1,
    s: cuda.Stream,
    packer: *const cuda.modelopt.Packer,
    context: usize,

    pub fn zeros(l: Loader, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(l.d, n);
        errdefer b.free();
        try b.fill8(0, null);
        try l.bufs.append(l.gpa, b);
        return b.ptr;
    }

    /// bf16 [n][k] into the lane matmul's layout on the device.
    pub fn lin(l: Loader, bytes: []const u8, n: usize, k: usize) !Lin {
        if (bytes.len != n * k * 2) return error.UnexpectedTensor;
        var src = try cuda.DeviceBuffer.fromHost(l.d, bytes);
        defer src.free();
        var out = try cuda.DeviceBuffer.alloc(l.d, cuda.nvfp4.padded(n) * k * 2);
        errdefer out.free();
        try l.packer.bf16Lane(l.s, src.ptr, out.ptr, n, k);
        try l.s.synchronize();
        try l.bufs.append(l.gpa, out);
        return cuda.modelopt.bf16Weight(out.ptr, n, k);
    }

    pub fn linOf(l: Loader, comptime fmt: []const u8, args: anytype) !Lin {
        const t = try l.tensor(fmt, args);
        return l.lin(t.bytes, t.dim(0), t.dim(1));
    }

    pub fn tensor(l: Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [200]u8 = undefined;
        return l.ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    }

    pub fn up(l: Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(l.d, bytes);
        try l.bufs.append(l.gpa, b);
        return b.ptr;
    }

    pub fn w(l: Loader, comptime fmt: []const u8, args: anytype) !u64 {
        return l.up((try l.tensor(fmt, args)).bytes);
    }

    /// bf16 tensors (or one) as fp32 on the device, one after the other.
    pub fn f32s(l: Loader, comptime fmt: []const u8, args: anytype, comptime indexed: bool, count: usize) !u64 {
        var all: std.ArrayList(f32) = .empty;
        defer all.deinit(l.gpa);
        for (0..count) |j| {
            const t = try l.tensor(fmt, if (indexed) args ++ .{j} else args);
            for (std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes))) |b| try all.append(l.gpa, @bitCast(@as(u32, b) << 16));
        }
        return l.up(std.mem.sliceAsBytes(all.items));
    }

    pub fn hc(l: Loader, comptime prefix: []const u8, args: anytype) !hy4.Hc {
        return .{ .fn_ = try l.w(prefix ++ "_fn", args), .base = try l.w(prefix ++ "_base", args), .scale = try l.w(prefix ++ "_scale", args) };
    }

    /// NVFP4 SwiGLU experts e0..e0+count (or one) packed as X lays them out, cut to intermediate rows lo..lo+ni.
    pub fn experts(l: Loader, comptime pre: []const u8, args: anytype, comptime single: bool, e0: usize, count: usize, total: usize, lo: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
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

pub fn bf16of(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// One MLA attention under `a` (its heads cut to this rank's), the target's layers' and DFly's alike.
pub fn loadAttn(l: Loader, comptime a: []const u8, args: anytype, H_all: usize, sliding: bool, q_rank: usize, hd: usize) !Attn {
    const H = H_all / l.world;
    const h0 = l.rank * H; // this rank's heads h0..h0+H
    const freqs = cuda.mla.ropeFreqs(if (sliding) sliding_theta else sparse_theta);
    // kv_b [H][192 + 256][512] split into wk [H][192][512] and wv [H][256][512]
    const kvb = (try l.tensor(a ++ "kv_b_proj.weight", args)).bytes;
    const wk = try l.gpa.alloc(u8, H * 192 * 1024);
    defer l.gpa.free(wk);
    const wv = try l.gpa.alloc(u8, H * 256 * 1024);
    defer l.gpa.free(wv);
    for (0..H) |h| {
        @memcpy(wk[h * 192 * 1024 ..][0 .. 192 * 1024], kvb[(h0 + h) * 448 * 1024 ..][0 .. 192 * 1024]);
        @memcpy(wv[h * 256 * 1024 ..][0 .. 256 * 1024], kvb[((h0 + h) * 448 + 192) * 1024 ..][0 .. 256 * 1024]);
    }
    const rows = H * 256 * 2; // a head slice's bytes a hidden column of q_b's and the gate's outputs
    const o_all = (try l.tensor(a ++ "o_proj.weight", args)).bytes;
    const o = try l.gpa.alloc(u8, hd * rows);
    defer l.gpa.free(o);
    for (0..hd) |r| @memcpy(o[r * rows ..][0..rows], o_all[r * H_all * 512 + h0 * 512 ..][0..rows]);
    const st = try l.tensor(a ++ "learnable_sink_param", args); // fp32 in the target, bf16 in DFly
    var sink: [128]f32 = undefined;
    for (0..H) |h| sink[h] = if (st.dtype == .f32) std.mem.bytesAsValue(f32, st.bytes[(h0 + h) * 4 ..][0..4]).* else @bitCast(@as(u32, std.mem.readInt(u16, st.bytes[(h0 + h) * 2 ..][0..2], .little)) << 16);
    return .{ .q_a = try l.linOf(a ++ "q_a_proj.weight", args), .q_a_norm = try l.w(a ++ "q_a_layernorm.weight", args), .q_b = try l.lin((try l.tensor(a ++ "q_b_proj.weight", args)).bytes[h0 * 256 * q_rank * 2 ..][0 .. H * 256 * q_rank * 2], H * 256, q_rank), .kv_a = try l.linOf(a ++ "kv_a_proj_with_mqa.weight", args), .kv_a_norm = try l.w(a ++ "kv_a_layernorm.weight", args), .wk = try l.up(wk), .wv = try l.up(wv), .sink = try l.up(std.mem.sliceAsBytes(sink[0..H])), .gate = try l.lin((try l.tensor(a ++ "linear_gate.weight", args)).bytes[h0 * 256 * hd * 2 ..][0 .. H * 256 * hd * 2], H * 256, hd), .o = try l.lin(o, hd, H * 256), .post_norm = try l.w(a ++ "post_proj_norm.weight", args), .inv_freq = try l.up(std.mem.sliceAsBytes(&freqs)), .heads = H, .sliding = sliding };
}

pub fn loadLayer(l: Loader, c: config.Text, i: usize, heads: [engram.heads]u32, engram_here: bool) !Layer {
    const p = "model.layers.{d}.";
    const sliding = c.kinds[i] == .sliding;
    const H_all: usize = c.heads(@intCast(i)).q;
    var attn = try loadAttn(l, p ++ "self_attn.", .{i}, H_all, sliding, c.q_rank, c.hidden);
    if (!sliding) attn.idx = try loadIndexer(l, i);
    const hd: usize = c.hidden;
    const mp = p ++ "mlp.";
    const di = c.dense_inter / l.world;
    const si = c.expert_inter * c.shared / l.world;
    const eo = c.experts / l.world;
    const mlp_: Mlp = if (!c.moe[i]) .{ .dense = .{ .t = try l.experts(mp, .{i}, true, 0, 1, c.dense_inter, l.rank * di, di, hd, 0), .norm = try l.w(mp ++ "post_fc2_norm.weight", .{i}) } } else .{ .moe = .{
        .router = try l.linOf(mp ++ "gate.weight", .{i}),
        .bias = try l.w(mp ++ "gate.e_score_correction_bias", .{i}),
        .down = try l.linOf(mp ++ "routed_expert_down_proj.weight", .{i}),
        .up = try l.linOf(mp ++ "routed_expert_up_proj.weight", .{i}),
        .routed = try l.experts(mp ++ "experts.", .{i}, false, l.rank * eo, eo, c.expert_inter, 0, c.expert_inter, c.latent, c.swiglu_limit),
        .shared = try l.experts(mp ++ "shared_experts.", .{i}, true, 0, 1, c.expert_inter * c.shared, l.rank * si, si, hd, 0),
        .norm = try l.w(mp ++ "combine_output_norm.weight", .{i}),
    } };
    return .{ .attn = attn, .hc_attn = try l.hc(p ++ "hc_attn_layer.hc_pre.hc", .{i}), .hc_ffn = try l.hc(p ++ "hc_mlp_layer.hc_pre.hc", .{i}), .in_norm = try l.w(p ++ "input_layernorm.weight", .{i}), .post_norm = try l.w(p ++ "post_attention_layernorm.weight", .{i}), .mlp = mlp_, .engram = if (engram_here) try loadEngram(l, i, heads) else null };
}

pub fn loadEngram(l: Loader, i: usize, moduli: [engram.heads]u32) !Engram {
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
        .hash = try l.zeros(l.context * engram.heads * 4),
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

/// The compressed indexer of sparse layer `i`, with caches for `l.context` tokens.
fn loadIndexer(l: Loader, i: usize) !Indexer {
    const p = "model.layers.{d}.self_attn.compressed_indexer.";
    return .{
        .q = try l.linOf(p ++ "query_projection.weight", .{i}),
        .hw = try l.linOf(p ++ "head_weight_projection.weight", .{i}),
        .tk = try l.linOf(p ++ "token_key_projection.weight", .{i}),
        .tk_norm = try l.w(p ++ "token_key_norm.weight", .{i}),
        .wv = try l.w(p ++ "key_value_projection.weight", .{i}),
        .wg = try l.w(p ++ "key_gate_projection.weight", .{i}),
        .ape = try l.w(p ++ "intra_group_ape", .{i}),
        .k_norm = try l.w(p ++ "key_norm.weight", .{i}),
        .tk_cache = try l.zeros(l.context * 128 * 2),
        .gk_cache = try l.zeros(l.context / 4 * 128 * 2),
    };
}
