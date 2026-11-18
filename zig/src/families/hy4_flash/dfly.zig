//! Hy4-flash's DFly drafter: 5 MLA layers on the target's own KV caches, a block of the last token and masks.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");

const Model = engine.Model;
const Lin = engine.Lin;

/// Draft tokens a round: the block is the last sampled token and this many masks.
pub const drafts = 5;
const width = drafts + 1;

const Layer = struct { attn: engine.Attn, in_norm: u64, post_norm: u64, gate_up: Lin, down: Lin, inter: usize, kv: usize };

pub const Draft = struct {
    layers: []Layer,
    norm: u64,
    enorm: u64,
    hnorm: u64,
    eh: Lin,
    fuse_gate: u64,
    c_hnorm: u64,
    c_enorm: u64,
    c_gate_up: Lin,
    c_down: Lin,
    mask: i32,
    pos: u64,

    /// The checkpoint's model.dfly_layers.* onto the device: attention heads, the FFN's width cut to this rank's.
    pub fn load(l: engine.Loader, gpa: std.mem.Allocator, kv_layers: []const u32, mask: u32, q_rank: usize, hd: usize) !Draft {
        const p = "model.dfly_layers.";
        const layers = try gpa.alloc(Layer, kv_layers.len);
        errdefer gpa.free(layers);
        for (layers, kv_layers, 0..) |*L, kv, i| {
            const gate = try l.tensor(p ++ "{d}.mlp.gate_proj.weight", .{i});
            const inter_all = gate.dim(0);
            const inter = inter_all / l.world;
            L.* = .{
                .attn = try engine.loadAttn(l, p ++ "{d}.self_attn.", .{i}, (try l.tensor(p ++ "{d}.self_attn.learnable_sink_param", .{i})).dim(0), false, q_rank, hd),
                .in_norm = try l.w(p ++ "{d}.input_layernorm.weight", .{i}),
                .post_norm = try l.w(p ++ "{d}.post_attention_layernorm.weight", .{i}),
                .gate_up = try gateUp(l, gate.bytes, (try l.tensor(p ++ "{d}.mlp.up_proj.weight", .{i})).bytes, inter_all, l.rank * inter, inter, hd),
                .down = try cols(l, (try l.tensor(p ++ "{d}.mlp.down_proj.weight", .{i})).bytes, hd, inter_all, l.rank * inter, inter),
                .inter = inter,
                .kv = kv,
            };
        }
        const cg = try l.tensor(p ++ "hidden_correction.gate_proj.weight", .{});
        return .{
            .layers = layers,
            .norm = try l.w(p ++ "final_norm.weight", .{}),
            .enorm = try l.w(p ++ "mtp_fusion.enorm.weight", .{}),
            .hnorm = try l.w(p ++ "mtp_fusion.hnorm.weight", .{}),
            .eh = try l.linOf(p ++ "mtp_fusion.eh_proj.weight", .{}),
            .fuse_gate = try l.w(p ++ "mtp_fusion.gate", .{}),
            .c_hnorm = try l.w(p ++ "hidden_correction.hidden_norm.weight", .{}),
            .c_enorm = try l.w(p ++ "hidden_correction.embed_norm.weight", .{}),
            .c_gate_up = try gateUp(l, cg.bytes, (try l.tensor(p ++ "hidden_correction.up_proj.weight", .{})).bytes, cg.dim(0), 0, cg.dim(0), cg.dim(1)),
            .c_down = try l.linOf(p ++ "hidden_correction.down_proj.weight", .{}),
            .mask = @intCast(mask),
            .pos = try l.zeros(4),
        };
    }

    /// Gate rows lo..lo+n then up rows lo..lo+n as one lane weight: its outputs [gate | up], as swiglu reads them.
    fn gateUp(l: engine.Loader, gate: []const u8, up: []const u8, all: usize, lo: usize, n: usize, k: usize) !Lin {
        _ = all;
        const both = try l.gpa.alloc(u8, 2 * n * k * 2);
        defer l.gpa.free(both);
        @memcpy(both[0 .. n * k * 2], gate[lo * k * 2 ..][0 .. n * k * 2]);
        @memcpy(both[n * k * 2 ..], up[lo * k * 2 ..][0 .. n * k * 2]);
        return l.lin(both, 2 * n, k);
    }

    /// Columns lo..lo+n of a [rows][all] weight.
    fn cols(l: engine.Loader, w: []const u8, rows: usize, all: usize, lo: usize, n: usize) !Lin {
        const out = try l.gpa.alloc(u8, rows * n * 2);
        defer l.gpa.free(out);
        for (0..rows) |r| @memcpy(out[r * n * 2 ..][0 .. n * 2], w[(r * all + lo) * 2 ..][0 .. n * 2]);
        return l.lin(out, rows, n);
    }

    /// `drafts` tokens after `t` (next at tokens.len) from m.anchor; the block's rows go into the target's caches.
    pub fn propose(d: *Draft, m: *Model, t: i32, out: *[drafts]i32) !void {
        m.reset();
        defer m.reset();
        const c = m.c;
        const hd: usize = c.hidden;
        const P = m.tokens.items.len;
        if (P + width > m.context) return error.PastContext;
        const s = m.s;
        const k = &m.k.k;
        var ids: [width]i32 = @splat(d.mask);
        ids[0] = t;
        const x = try m.buf(width * hd * 2);
        try d.embedRows(m, &ids, x);
        const pos_h = [1]i32{@intCast(P)};
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = d.pos, .len = 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&pos_h));
        // the anchor row fused with the target's hidden state
        const cat = try m.buf(2 * hd * 2);
        try k.rmsNorm16(s, x, hd, d.enorm, cat, 2 * hd, 1, hd, c.eps);
        try k.rmsNorm16(s, m.anchor, hd, d.hnorm, cat + hd * 2, 2 * hd, 1, hd, c.eps);
        const fused = try m.buf(hd * 4);
        try m.proj(d.eh, cat, 2 * hd, fused, 1);
        try k.fuse(s, x, fused, d.fuse_gate, hd);
        // the layers on an fp32 residual
        const r = try m.zeros(width * hd * 4);
        try k.addb(s, r, x, width * hd);
        for (d.layers) |*L| {
            const hn = try m.buf(width * hd * 2);
            try k.rmsNorm(s, r, hd, L.in_norm, hn, hd, width, hd, c.eps);
            try k.addb(s, r, try m.attend(&L.attn, m.cache[L.kv], hn, d.pos, P, width), width * hd);
            const hn2 = try m.buf(width * hd * 2);
            try k.rmsNorm(s, r, hd, L.post_norm, hn2, hd, width, hd, c.eps);
            const gu = try m.buf(width * 2 * L.inter * 4);
            try m.proj(L.gate_up, hn2, hd, gu, width);
            const act = try m.buf(width * L.inter * 2);
            try k.swiglu(s, gu, act, width, L.inter, 0);
            const dn = try m.buf(width * hd * 4);
            try m.proj(L.down, act, L.inter, dn, width);
            try m.sum(dn, width * hd);
            try k.add32(s, r, dn, width * hd);
        }
        const hidden = try m.buf(width * hd * 2);
        try k.rmsNorm(s, r, hd, d.norm, hidden, hd, width, hd, c.eps);
        // one token at a time: each row corrected by the token before it, then the target's head
        const ci = d.c_down.k;
        const prev = try m.buf(hd * 2);
        const logits = try m.buf(c.vocab * 4);
        const row = try m.gpa.alloc(f32, c.vocab);
        defer m.gpa.free(row);
        var last = t;
        for (0..drafts) |j| {
            try d.embedRows(m, &.{last}, prev);
            const h = hidden + (j + 1) * hd * 2;
            try k.rmsNorm16(s, h, hd, d.c_hnorm, cat, 2 * hd, 1, hd, c.eps);
            try k.rmsNorm16(s, prev, hd, d.c_enorm, cat + hd * 2, 2 * hd, 1, hd, c.eps);
            const gu = try m.buf(2 * ci * 4);
            try m.proj(d.c_gate_up, cat, 2 * hd, gu, 1);
            const act = try m.buf(ci * 2);
            try k.swiglu(s, gu, act, 1, ci, 0);
            const corr = try m.zeros(hd * 4);
            try m.proj(d.c_down, act, ci, corr, 1);
            try k.addb(s, corr, h, hd);
            const ch = try m.buf(hd * 2);
            try k.castRows(s, corr, hd, ch, hd, 1, hd);
            try m.proj(m.lm, ch, hd, logits, 1);
            try s.synchronize();
            try cuda.memory.download(m.d, logits, std.mem.sliceAsBytes(row));
            last = @intCast(std.mem.indexOfMax(f32, row));
            out[j] = last;
        }
    }

    /// The tokens' embedding rows (bf16, the target's table) into `x`.
    fn embedRows(d: *Draft, m: *Model, ids: []const i32, x: u64) !void {
        _ = d;
        const hd: usize = m.c.hidden;
        const h = try m.gpa.alloc(u8, ids.len * hd * 2);
        defer m.gpa.free(h);
        for (ids, 0..) |id, r| @memcpy(h[r * hd * 2 ..][0 .. hd * 2], m.embed[@as(usize, @intCast(id)) * hd * 2 ..][0 .. hd * 2]);
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = x, .len = h.len, .borrowed = true }).upload(0, h);
    }
};
