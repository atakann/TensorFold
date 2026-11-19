//! MLA's per-head absorb and expand as Lane GEMMs on the tensor cores for prompt passes (decode keeps mla.cuh's).
const engine = @import("engine.zig");
const wt = @import("weights.zig");

const Model = engine.Model;
const row_width = engine.row_width;

/// Rows from which a step takes these GEMMs: decode, verify and DFly rounds stay below it, in their graphs.
pub const min_rows = 64;

/// qa [n][H][row_width] bf16, its first 512 columns a head = q_nope [n][H][192] (q rows H * 256 apart) times wk.
pub fn absorb(m: *Model, a: *const wt.Attn, q: u64, qa: u64, n: usize) !void {
    const f = try m.buf(n * 512 * 4);
    for (0..a.heads) |h| {
        try m.proj(a.lins.wk[h], q + h * 256 * 2, a.heads * 256, f, n);
        try m.k.k.castRows(m.s, f, 512, qa + h * row_width * 2, a.heads * row_width, n, 512);
    }
}

/// o [n][H][256] bf16 = the merged latent rows ol [n][H][512] times each head's wv.
pub fn expand(m: *Model, a: *const wt.Attn, ol: u64, o: u64, n: usize) !void {
    const f = try m.buf(n * 256 * 4);
    for (0..a.heads) |h| {
        try m.proj(a.lins.wv[h], ol + h * 512 * 2, a.heads * 512, f, n);
        try m.k.k.castRows(m.s, f, 256, o + h * 256 * 2, a.heads * 256, n, 256);
    }
}
