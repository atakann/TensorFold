//! hy4-engine-check (GPU): engine.zig on 6 layers: one step against the reference, chunks and decode against the step.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const hy4 = @import("hy4_flash");
const t = @import("hy4_checks.zig");

const engine = hy4.engine;

/// config.json's text config, and the Engram head sizes of each Engram layer in layer order.
fn readConfig(gpu: check.Gpu, dir: []const u8, heads: *std.ArrayList([hy4.engram.heads]u32)) !hy4.config.Config {
    const gpa = gpu.gpa;
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var why: hy4.config.Why = .{};
    const cfg = hy4.config.parse(gpa, text, &why) catch |e| {
        std.debug.print("config.json refused: {s}\n", .{why.text()});
        return e;
    };
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const tc = parsed.value.object.get("text_config").?.object;
    const ids = tc.get("engram_layer_ids").?.array.items;
    const sizes = tc.get("engram_layer_head_vocab_sizes").?.object;
    for (ids) |id| {
        if (id.integer > cfg.text.layers) continue;
        var key: [8]u8 = undefined;
        var one: [hy4.engram.heads]u32 = undefined;
        var n: usize = 0;
        for (sizes.get(try std.fmt.bufPrint(&key, "{d}", .{id.integer})).?.array.items) |o| for (o.array.items) |v| {
            one[n] = @intCast(v.integer);
            n += 1;
        };
        try heads.append(gpa, one);
    }
    return cfg;
}

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn relRows(got: []const f32, want: []align(1) const f32) f64 {
    var dd: f64 = 0;
    var ww: f64 = 0;
    for (got, want) |g, w| {
        dd += (@as(f64, g) - w) * (@as(f64, g) - w);
        ww += @as(f64, w) * w;
    }
    return @sqrt(dd / ww);
}

/// hy4-engine-check REAL6 REF
pub fn run(gpu: check.Gpu, rest: []const []const u8) !void {
    if (rest.len != 2) return error.UnknownCommand;
    const gpa = gpu.gpa;
    var heads: std.ArrayList([hy4.engram.heads]u32) = .empty;
    defer heads.deinit(gpa);
    const cfg = try readConfig(gpu, rest[0], &heads);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k = try engine.Kernels.load(gpu.d, sms);
    defer k.unload();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, rest[0]);
    defer ck.close();
    var ref = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, rest[1]);
    defer ref.close();
    // TF_TP_RANK of TF_TP_WORLD (NCCL_COMM_ID: rank 0's address): this rank's share, partial sums over NCCL
    const world = envInt("TF_TP_WORLD", 1);
    var group: ?cuda.tp_group.Group = if (world > 1) try cuda.tp_group.Group.open(gpu.d, envInt("TF_TP_RANK", 0), world) else null;
    defer if (group) |*g| g.close();
    const t0 = check.now(gpu.io);
    var m = try engine.Model.open(gpa, gpu.d, s, &k, &ck, cfg, heads.items, 1024, if (group) |*g| g else null);
    defer m.close();
    std.debug.print("  rank {d} of {d}: opened {d} layers in {d:.1} s\n", .{ m.rank, m.world, cfg.text.layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9 });
    const ids: []align(1) const i32 = std.mem.bytesAsSlice(i32, (try ref.get("input_ids")).bytes);
    const n = ids.len;
    const V: usize = cfg.text.vocab;
    const idv = try gpa.alloc(i32, n);
    defer gpa.free(idv);
    for (idv, ids) |*o, v| o.* = v;
    const all = try gpa.alloc(f32, n * V);
    defer gpa.free(all);
    try m.step(idv, false, all);
    const want = try t.refF32(&ref, "logits", .{});
    var same: usize = 0;
    for (0..n) |r| same += @intFromBool(std.mem.indexOfMax(f32, all[r * V ..][0..V]) == t.argmax(want[r * V ..][0..V]));
    const rel = relRows(all, want);
    check.pass("hy4 engine: one step of {d} rows: logits rel err {e:.3}, argmax {d}/{d} as the reference", .{ n, rel, same, n });
    try check.expect(rel < 5e-2 and same * 10 >= n * 9, "hy4 engine: one step: rel err {e}, argmax {d}/{d}", .{ rel, same, n });
    // the same prompt again: two chunks, then single decode rows, each row's logits against the one step's
    m.tokens.clearRetainingCapacity();
    const row = try gpa.alloc(f32, V);
    defer gpa.free(row);
    const cuts = [_]usize{ 40, n - 5 };
    var at: usize = 0;
    var worst: f64 = 0;
    var flips: usize = 0;
    for (0..cuts.len + 5) |j| {
        const end = if (j < cuts.len) cuts[j] else at + 1;
        try m.step(idv[at..end], true, row);
        const r = relRows(row, all[(end - 1) * V ..][0..V]);
        worst = @max(worst, r);
        flips += @intFromBool(std.mem.indexOfMax(f32, row) != std.mem.indexOfMax(f32, all[(end - 1) * V ..][0..V]));
        at = end;
    }
    check.pass("hy4 engine: chunks of 40 and 20, then 5 decode rows: worst rel err {e:.3} against the one step, {d} argmax flips", .{ worst, flips });
    try check.expect(worst < 1e-2 and flips == 0, "hy4 engine: chunked and decode rows differ: rel err {e}, {d} flips", .{ worst, flips });
}
