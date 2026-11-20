//! BF16 projections as FP8 E4M3 with one scale a tensor, made at load: half the bytes a decode step reads.
const std = @import("std");

/// The largest finite E4M3 value.
pub const max: f32 = 448.0;

/// `x` as E4M3, rounded to nearest even and saturated at +-448 (no NaN code is written).
pub fn encode(x: f32) u8 {
    const sign: u8 = if (x < 0 or (x == 0 and std.math.signbit(x))) 0x80 else 0;
    const a = @abs(x);
    if (!(a < max)) return sign | 0x7E; // 448 and above, and NaN, saturate
    // subnormals: m x 2^-9 for m in 0..7; m 8 is the smallest normal, 0x08
    if (a < 0x1p-6) return sign | @as(u8, @intFromFloat(even(a * 0x1p9)));
    const bits: u32 = @bitCast(a);
    var ee: i32 = @as(i32, @intCast(bits >> 23)) - 127; // floor(log2 a), -6..8
    const mant = bits & 0x7FFFFF;
    var m: u32 = mant >> 20; // three mantissa bits, then round the 20 below to nearest even
    const rest = mant & 0xFFFFF;
    if (rest > 0x80000 or (rest == 0x80000 and m & 1 == 1)) m += 1;
    if (m == 8) {
        m = 0;
        ee += 1;
    }
    if (ee > 8 or (ee == 8 and m == 7)) return sign | 0x7E;
    return sign | @as(u8, @intCast((ee + 7) << 3)) | @as(u8, @intCast(m));
}

/// The E4M3 code's value.
pub fn decode(c: u8) f32 {
    const e: i32 = (c >> 3) & 0xF;
    const m: f32 = @floatFromInt(c & 7);
    const v = if (e == 0) m * 0x1p-9 else (1 + m / 8) * std.math.pow(f32, 2, @floatFromInt(e - 7));
    return if (c & 0x80 != 0) -v else v;
}

fn even(v: f32) f32 {
    const f = @floor(v);
    const d = v - f;
    if (d > 0.5 or (d == 0.5 and @mod(f, 2) == 1)) return f + 1;
    return f;
}

fn bf16(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b[i * 2 ..][0..2], .little)) << 16);
}

/// BF16 `bytes` [n][k] into E4M3 codes in `out` with scale amax / 448; returns the scale (value = code x scale).
pub fn quantize(bytes: []const u8, out: []u8) f32 {
    var amax: f32 = 0;
    for (0..out.len) |i| amax = @max(amax, @abs(bf16(bytes, i)));
    const scale = if (amax > 0) amax / max else 1;
    for (out, 0..) |*c, i| c.* = encode(bf16(bytes, i) / scale);
    return scale;
}

test "E4M3 codes of known values, rounding and saturation" {
    const t = std.testing;
    try t.expectEqual(@as(u8, 0x38), encode(1.0));
    try t.expectEqual(@as(u8, 0x30), encode(0.5));
    try t.expectEqual(@as(u8, 0xC0), encode(-2.0));
    try t.expectEqual(@as(u8, 0x7E), encode(448.0));
    try t.expectEqual(@as(u8, 0x7E), encode(1000.0));
    try t.expectEqual(@as(u8, 0x01), encode(0x1p-9));
    try t.expectEqual(@as(u8, 0x08), encode(0x1p-6));
    try t.expectEqual(@as(u8, 0x00), encode(0));
    try t.expectEqual(@as(u8, 0x38), encode(1.0625)); // halfway between 1 and 1.125: to even
    try t.expectEqual(@as(u8, 0x3A), encode(1.1875)); // halfway between 1.125 and 1.25: to even (1.25)
    for (0..256) |c| {
        const code: u8 = @intCast(c);
        if (code & 0x7F == 0x7F or code == 0x80) continue; // NaN, and -0 reads as 0
        try t.expectEqual(code, encode(decode(code)));
    }
}
