//! Hy4-flash behind the native CUDA registry: --tp N over processes (NCCL_COMM_ID); rank 0 serves, the rest follow.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const dfly = @import("dfly.zig");
const files = @import("files.zig");
const serve = @import("serve.zig");
const config = @import("config.zig");

pub const model_type = "hy_v4_flash";
pub const formats: []const []const u8 = &.{ "modelopt-nvfp4", "modelopt-mixed" };
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.max_step_rows;
/// A kept state is the caches through its position and nothing past it.
pub const snap_lookahead: u32 = 0;
/// DFly is the checkpoint's own drafter: --drafter loads nothing more.
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Hy4-flash drafts with the checkpoint's own DFly layers; serve it without --drafter";
pub const explains_open = true;
/// A request's verify reserve: the window past its prompt and reply.
pub const window_rows: u32 = serve.max_window;

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, room: ?Room = null, tp: u32 = 1, rank: u32 = 0, master: ?[]const u8 = null, master_port: u16 = 29551 };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    learn_bytes: usize = 0,
    target: *serve.Hy4,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

const Owned = struct {
    gpa: std.mem.Allocator,
    setup: files.Setup,
    /// Open while serving: Engram's token id tables are read from its mapped files.
    ck: core.checkpoint.Checkpoint,
    s: cuda.Stream,
    k: engine.Kernels,
    group: ?cuda.tp_group.Group,
    m: engine.Model,
    draft: ?dfly.Draft,
    link: ?serve.Link,
    hy: serve.Hy4,
};

/// A rank's share of the checkpoint: the experts, heads and Engram tables split, the rest whole (an estimate).
pub fn placementWeightBytes(_: std.mem.Allocator, io: std.Io, dir: []const u8, tp: u32, _: u32, _: bool) !u64 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) {
        total += (try d.statFile(io, e.name, .{})).size;
    };
    return total / @max(tp, 1);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const rank: usize = o.rank;
    const world: usize = @max(o.tp, 1);
    if (rank >= world) return error.BadRank;
    // TF_HY4_FP8=0 keeps the projections BF16 as stored; by default they load as FP8 E4M3 (fp8.zig)
    @import("weights.zig").fp8_projections = if (core.profile.lookup("TF_HY4_FP8")) |v| !std.mem.eql(u8, v, "0") else true;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    var why: config.Why = .{};
    own.setup = files.read(gpa, io, dir, &why) catch |e| {
        std.log.err("config.json refused: {s}", .{why.text()});
        return e;
    };
    errdefer own.setup.deinit(gpa);
    own.s = try cuda.Stream.init(ctx.d, true);
    errdefer own.s.deinit();
    const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
    own.k = try engine.Kernels.load(ctx.d, sms);
    errdefer own.k.unload();
    own.group = if (world > 1) try cuda.tp_group.Group.open(ctx.d, rank, world) else null;
    errdefer if (own.group) |*g| g.close();
    const group: ?*const cuda.tp_group.Group = if (own.group) |*g| g else null;
    // the cache holds the window, the verify reserve and DFly's block, in whole 512-key chunks
    const context = std.mem.alignForward(usize, o.context + 2 * serve.max_window, 512);
    own.ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    errdefer own.ck.close();
    own.m = try engine.Model.open(gpa, ctx.d, own.s, &own.k, &own.ck, own.setup.cfg, own.setup.heads.items, context, group);
    errdefer own.m.close();
    own.draft = null;
    if (o.drafts and own.setup.n_kv > 0) {
        const l: engine.Loader = .{ .gpa = gpa, .d = ctx.d, .ck = &own.ck, .bufs = &own.m.weights, .rank = rank, .world = world, .s = own.s, .packer = &own.k.packer, .context = context };
        const c = own.setup.cfg.text;
        own.draft = try dfly.Draft.load(l, gpa, own.setup.kv[0..own.setup.n_kv], own.setup.mask, c.q_rank, c.hidden);
        own.m.snap = try own.draft.?.snap(l);
    }
    own.link = if (group) |g| try serve.Link.init(gpa, ctx.d, g, context) else null;
    errdefer if (own.link) |*l| l.deinit(gpa);
    own.hy = try serve.Hy4.init(gpa, &own.m, if (own.draft) |*d| d else null, if (own.link) |*l| l else null);
    std.log.info("hy4: rank {d} of {d} loaded, {d}-token cache, DFly {s}", .{ rank, world, context, if (own.draft != null) "on" else "off" });
    return .{
        .backend = own.hy.backend(),
        .facts = own.hy.facts(),
        .rows = serve.max_window,
        .stream_bytes = 0,
        .free_streams = 1,
        .ctx = own,
        .deinit = release,
        .target = &own.hy,
        .follow = if (rank > 0) followRank0 else null,
    };
}

fn followRank0(p: *anyopaque) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(p));
    return serve.follow(own.gpa, &own.m, if (own.draft) |*d| d else null, &own.link.?);
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "Hy4-flash serves one stream: retry once the running request ends",
        error.StructuresUnsupported => "structured output is not built for Hy4-flash yet",
        error.CapturedKernels => "Hy4-flash runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BadRank => "--rank: this rank's number below --tp",
        error.NoCommId => "--tp above 1 needs NCCL_COMM_ID (rank 0's address:port) on every rank",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.hy.deinit();
    if (own.link) |*l| l.deinit(own.gpa);
    own.m.close();
    own.ck.close();
    if (own.group) |*g| g.close();
    own.k.unload();
    own.s.deinit();
    own.setup.deinit(own.gpa);
    own.gpa.destroy(own);
}
