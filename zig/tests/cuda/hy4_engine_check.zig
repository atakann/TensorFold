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

/// config.json's DFly keys: the target layers whose caches the draft layers read, and the mask token.
fn dflyConfig(gpu: check.Gpu, dir: []const u8, kv: *[8]u32, mask: *u32) !usize {
    const gpa = gpu.gpa;
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    mask.* = @intCast(o.get("dfly_mask_token_id").?.integer);
    const items = o.get("dfly_target_kv_layers").?.array.items;
    for (items, 0..) |v, i| kv[i] = @intCast(v.integer);
    return items.len;
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
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

/// hy4-generate CKPT IDS N: a prompt (comma-separated ids in a file), then N greedy tokens; rank 0 prints them.
pub fn generate(gpu: check.Gpu, rest: []const []const u8) !void {
    if (rest.len != 3) return error.UnknownCommand;
    const gpa = gpu.gpa;
    var heads: std.ArrayList([hy4.engram.heads]u32) = .empty;
    defer heads.deinit(gpa);
    const cfg = try readConfig(gpu, rest[0], &heads);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[1], gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(i32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |v| try ids.append(gpa, try std.fmt.parseInt(i32, v, 10));
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k = try engine.Kernels.load(gpu.d, sms);
    defer k.unload();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, rest[0]);
    defer ck.close();
    const world = envInt("TF_TP_WORLD", 1);
    var group: ?cuda.tp_group.Group = if (world > 1) try cuda.tp_group.Group.open(gpu.d, envInt("TF_TP_RANK", 0), world) else null;
    defer if (group) |*g| g.close();
    const t0 = check.now(gpu.io);
    var m = try engine.Model.open(gpa, gpu.d, s, &k, &ck, cfg, heads.items, 8192, if (group) |*g| g else null);
    defer m.close();
    if (std.c.getenv("TF_HY4_PROF") != null) m.prof = gpu.io;
    if (std.c.getenv("TF_HY4_NOGRAPH") != null) m.use_graphs = false;
    const t1 = check.now(gpu.io);
    std.debug.print("  rank {d} of {d}: {d} layers in {d:.1} s\n", .{ m.rank, m.world, cfg.text.layers, @as(f64, @floatFromInt(t1 - t0)) / 1e9 });
    const V: usize = cfg.text.vocab;
    const D = hy4.dfly.drafts;
    const rows = try gpa.alloc(f32, (D + 1) * V);
    defer gpa.free(rows);
    // TF_HY4_DFLY=1: the checkpoint's DFly drafter proposes, one target step verifies the block
    var draft: ?hy4.dfly.Draft = null;
    if (std.c.getenv("TF_HY4_DFLY") != null) {
        var kv: [8]u32 = undefined;
        var mask: u32 = 0;
        const n_kv = try dflyConfig(gpu, rest[0], &kv, &mask);
        const l: engine.Loader = .{ .gpa = gpa, .d = gpu.d, .ck = &ck, .bufs = &m.weights, .rank = m.rank, .world = m.world, .s = s, .packer = &k.packer, .context = m.context };
        draft = try hy4.dfly.Draft.load(l, gpa, kv[0..n_kv], mask, cfg.text.q_rank, cfg.text.hidden);
    }
    try m.step(ids.items, true, rows);
    try m.keepRow(0);
    const t2 = check.now(gpu.io);
    var out: std.ArrayList(i32) = .empty;
    defer out.deinit(gpa);
    var next: i32 = @intCast(std.mem.indexOfMax(f32, rows[0..V]));
    var rounds: usize = 0;
    const eos: i32 = @intCast(cfg.eos);
    gen: while (out.items.len < steps) {
        try out.append(gpa, next);
        if (next == eos) break;
        if (draft) |*dr| {
            var d: [D]i32 = undefined;
            try dr.propose(&m, next, &d);
            const p0 = m.tokens.items.len;
            var block: [D + 1]i32 = undefined;
            block[0] = next;
            @memcpy(block[1..], &d);
            try m.step(&block, false, rows);
            var a: usize = 0;
            while (a < D and std.mem.indexOfMax(f32, rows[a * V ..][0..V]) == @as(usize, @intCast(d[a]))) a += 1;
            m.tokens.shrinkRetainingCapacity(p0 + 1 + a);
            try m.keepRow(a);
            rounds += 1;
            for (d[0..a]) |v| {
                try out.append(gpa, v);
                if (v == eos) break :gen;
            }
            next = @intCast(std.mem.indexOfMax(f32, rows[a * V ..][0..V]));
        } else {
            try m.step(&.{next}, true, rows);
            next = @intCast(std.mem.indexOfMax(f32, rows[0..V]));
        }
    }
    const t3 = check.now(gpu.io);
    if (m.rank != 0) return;
    if (m.prof != null) std.debug.print("  time (prompt and decode, ms): engram {d:.0}, attention {d:.0}, mlp {d:.0}, head and the rest {d:.0}\n", .{ ms(m.times[0]), ms(m.times[1]), ms(m.times[2]), ms(m.times[3]) });
    if (m.prof != null) std.debug.print("  inside attention (ms): rope, kv rows and absorb {d:.0}, chunks {d:.0}, merge and expand {d:.0}, gate {d:.0}\n", .{ ms(m.times[4]), ms(m.times[5]), ms(m.times[7]), ms(m.times[6]) });
    if (rounds > 0) std.debug.print("  DFly: {d} rounds, {d:.2} tokens a round\n", .{ rounds, @as(f64, @floatFromInt(out.items.len)) / @as(f64, @floatFromInt(rounds)) });
    std.debug.print("  prompt {d} tokens in {d:.2} s; {d} tokens in {d:.2} s: {d:.2} tok/s\nIDS ", .{ ids.items.len, @as(f64, @floatFromInt(t2 - t1)) / 1e9, out.items.len, @as(f64, @floatFromInt(t3 - t2)) / 1e9, @as(f64, @floatFromInt(out.items.len)) / (@as(f64, @floatFromInt(t3 - t2)) / 1e9) });
    for (out.items) |v| std.debug.print("{d},", .{v});
    std.debug.print("\n", .{});
}
