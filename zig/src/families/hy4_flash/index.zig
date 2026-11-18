//! Hy4-flash's compressed indexer on the engine's caches: token and group keys, then each row's recalled tokens.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const wt = @import("weights.zig");

const Model = engine.Model;

/// The indexer's caches for rows pos..: each row's token key, and the group key of each group a row closes.
pub fn keys(m: *Model, ix: wt.Indexer, x: u64, pos: u64, inv_freq: u64, n: usize) !void {
    const k = &m.k.k;
    const f = try m.buf(n * 128 * 4);
    try m.proj(ix.tk, x, m.c.hidden, f, n);
    const rows = try m.buf(n * 128 * 2);
    try k.rmsNorm(m.s, f, 128, ix.tk_norm, rows, 128, n, 128, m.c.eps);
    try k.put(m.s, ix.tk_cache, rows, pos, n, 128);
    try k.idxPool(m.s, ix.tk_cache, ix.gk_cache, pos, .{ .wv = ix.wv, .wg = ix.wg, .ape = ix.ape, .norm = ix.k_norm, .inv_freq = inv_freq }, n, m.c.eps);
}

/// Each row's recalled keys (its own group and the best 511 before it, scored over `np` groups) as token lists.
pub fn recalled(m: *Model, ix: wt.Indexer, x: u64, qr: u64, pos: u64, inv_freq: u64, np: usize, n: usize, tok: u64, cnt: u64) !void {
    const k = &m.k.k;
    const qf = try m.buf(n * 2048 * 4);
    try m.proj(ix.q, qr, m.c.q_rank, qf, n);
    const q = try m.buf(n * 2048 * 2);
    try k.castRows(m.s, qf, 2048, q, 2048, n, 2048);
    try m.k.mla.rope(m.s, .{ .x = q + 64 * 2, .x_row = 2048, .x_head = 128, .out = q + 64 * 2, .o_row = 2048, .o_head = 128 }, n, 16, pos, inv_freq, 1.0);
    const w = try m.buf(n * 16 * 4);
    try m.proj(ix.hw, x, m.c.hidden, w, n);
    const scores = try m.buf(n * np * 4);
    try k.idxScores(m.s, q, w, ix.gk_cache, pos, scores, n, np, 1.0 / @sqrt(16.0 * 128.0));
    const sel = try m.buf(n * 512 * 8);
    var a: cuda.Args = .{};
    inline for (.{ scores, sel, @as(i64, @intCast(np)), m.pos3, @as(c_int, 512) }) |v| a.add(v);
    const wide = n <= 4;
    try cuda.launch.launch(if (wide) m.k.select_wide else m.k.select, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = if (wide) 1024 else 128 } }, m.s, &a);
    if (m.probe != 0 and m.at_layer == m.probe_layer) try (cuda.DeviceBuffer{ .d = m.d, .ptr = m.probe, .len = n * 512 * 8, .borrowed = true }).copyFrom(0, sel, n * 512 * 8, m.s.handle);
    try k.idxTokens(m.s, sel, pos, tok, cnt, n);
}
