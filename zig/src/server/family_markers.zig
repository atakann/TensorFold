//! The think markers a family's tokenizer names: Hy4-flash's beside the shared ones.
const reply_text = @import("reply_text.zig");

/// Hy4-flash's template opens the think block in the prompt. Its closer token decodes as `</think>` (hf_hy4.zig).
pub const hy4: reply_text.Markers = .{ .open = "", .close = "</think>" };

/// The token only Hy4-flash's tokenizer has.
const hy4_closer = "</think:opensource>";

/// The server's markers: MiniMax-M3's or Hy4-flash's when the tokenizer has their closer, else `current`.
pub fn pick(current: reply_text.Markers, minimax: bool, text: anytype) reply_text.Markers {
    if (minimax) return reply_text.mm_markers;
    if (text.tokenId(hy4_closer) != null) return hy4;
    return current;
}

/// The tag a reply's think block ends with: the markers' own, `</think>` beside Gemma 4's channel.
pub fn closeTag(m: reply_text.Markers) []const u8 {
    return if (reply_text.isChannel(m)) "</think>" else m.close;
}

test "Hy4-flash's closer is its own, the shared markers keep theirs" {
    const std = @import("std");
    try std.testing.expectEqualStrings("</think>", closeTag(hy4));
    try std.testing.expectEqualStrings("</think>", closeTag(reply_text.channel_markers));
    try std.testing.expectEqualStrings("</mm:think>", closeTag(reply_text.mm_markers));
}
