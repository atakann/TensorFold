//! config.json's parts the engine needs: the text config, each Engram layer's head sizes and the DFly keys.
const std = @import("std");
const config = @import("config.zig");
const engram = @import("engram.zig");

pub const Setup = struct {
    cfg: config.Config,
    /// The Engram head sizes of each Engram layer, in layer order.
    heads: std.ArrayList([engram.heads]u32) = .empty,
    /// DFly: the target layers whose caches the draft layers read (0 of them: no drafter), and the mask token.
    kv: [8]u32 = undefined,
    n_kv: usize = 0,
    mask: u32 = 0,

    pub fn deinit(s: *Setup, gpa: std.mem.Allocator) void {
        s.heads.deinit(gpa);
    }
};

/// The model folder's config.json; `why` names a refusal.
pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: *config.Why) !Setup {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var s: Setup = .{ .cfg = try config.parse(gpa, text, why) };
    errdefer s.deinit(gpa);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const tc = o.get("text_config").?.object;
    const sizes = tc.get("engram_layer_head_vocab_sizes").?.object;
    for (tc.get("engram_layer_ids").?.array.items) |id| {
        if (id.integer > s.cfg.text.layers) continue;
        var key: [8]u8 = undefined;
        var one: [engram.heads]u32 = undefined;
        var n: usize = 0;
        for (sizes.get(try std.fmt.bufPrint(&key, "{d}", .{id.integer})).?.array.items) |g| for (g.array.items) |v| {
            one[n] = @intCast(v.integer);
            n += 1;
        };
        try s.heads.append(gpa, one);
    }
    if (o.get("dfly_target_kv_layers")) |kv| {
        for (kv.array.items, 0..) |v, i| s.kv[i] = @intCast(v.integer);
        s.n_kv = kv.array.items.len;
        s.mask = @intCast(o.get("dfly_mask_token_id").?.integer);
    }
    return s;
}
