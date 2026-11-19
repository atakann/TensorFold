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

/// A draft row's token list past the recall budget: the target's 2048 recalled keys and the block's rows.
const list_width = 2048 + width;

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
    /// The block's ids on the device (the last token, then masks) and the drafts a proposal leaves.
    ids: u64,
    out: u64,

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
            .ids = blk: {
                var ids: [width]i32 = @splat(@intCast(mask));
                ids[0] = 0;
                break :blk try l.up(std.mem.sliceAsBytes(&ids));
            },
            .out = try l.zeros(drafts * 4),
        };
    }

    /// Whether the draft reads the target's recalled keys: the kept row's step ran the indexer.
    fn overTarget(d: *const Draft, m: *const Model) bool {
        _ = d;
        const sn = m.snap orelse return false;
        return m.use_index and sn.keys > m.c.index_topk;
    }

    /// The snapshot of the target layers this draft reads (a block's lists for each), on the device.
    pub fn snap(d: *const Draft, l: engine.Loader) !engine.Snap {
        var sn: engine.Snap = .{ .layers = undefined, .n = d.layers.len, .tok = try l.zeros(d.layers.len * engine.verify_rows * 2048 * 4), .cnt = try l.zeros(d.layers.len * engine.verify_rows * 4), .slot = try l.zeros(4) };
        for (d.layers, 0..) |L, j| sn.layers[j] = L.kv;
        return sn;
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
        const P = m.tokens.items.len;
        if (P + width > m.context) return error.PastContext;
        const t_h = [1]i32{t};
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = d.ids, .len = 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&t_h));
        const pos_h = [1]i32{@intCast(P)};
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = d.pos, .len = 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&pos_h));
        const Run = struct {
            d: *Draft,
            P: usize,
            pub fn run(r: @This(), mm: *Model) !u64 {
                return r.d.body(mm, r.P);
            }
        };
        const nch = (P + width + 511) / 512;
        const over: usize = @intFromBool(d.overTarget(m));
        if (m.use_graphs) {
            _ = try m.graphed(nch * engine.graph_kinds + 2 + over, Run{ .d = d, .P = P });
        } else _ = try d.body(m, P);
        try m.s.synchronize();
        try cuda.memory.download(m.d, d.out, std.mem.sliceAsBytes(out));
        m.reset();
    }

    /// The proposal on the device alone (a graph replays it): the block's layers, then each draft's head and argmax.
    fn body(d: *Draft, m: *Model, P: usize) !u64 {
        m.reset();
        const c = m.c;
        const hd: usize = c.hidden;
        const s = m.s;
        const k = &m.k.k;
        const x = try m.buf(width * hd * 2);
        try m.k.embed.bf16(s, d.ids, m.table, x, width, hd);
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
        const over = d.overTarget(m);
        for (d.layers, 0..) |*L, j| {
            const hn = try m.buf(width * hd * 2);
            try k.rmsNorm(s, r, hd, L.in_norm, hn, hd, width, hd, c.eps);
            // past the recall budget: the target's recalled keys for the kept row, then the block (vLLM's reuse)
            var list: ?Model.List = null;
            if (over) {
                const sn = m.snap.?;
                const tok = try m.buf(width * list_width * 4);
                const cnt = try m.buf(width * 4);
                try k.dflyTokens(s, sn.tok + j * engine.verify_rows * 2048 * 4, sn.cnt + j * engine.verify_rows * 4, sn.slot, d.pos, tok, cnt, width, list_width);
                list = .{ .tok = tok, .cnt = cnt, .w = list_width };
            }
            try k.addb(s, r, try m.attendList(&L.attn, m.cache[L.kv], hn, d.pos, P, width, list), width * hd);
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
        // one token at a time: each row corrected by the token before it, then the target's head and argmax
        const ci = d.c_down.k;
        const prev = try m.buf(hd * 2);
        for (0..drafts) |j| {
            try m.k.embed.bf16(s, if (j == 0) d.ids else d.out + (j - 1) * 4, m.table, prev, 1, hd);
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
            try m.headIds(ch, 1, d.out + j * 4);
        }
        return d.out;
    }
};
