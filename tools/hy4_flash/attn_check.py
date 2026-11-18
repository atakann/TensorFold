"""Dev-time check of the MLA chunk body and a sink merge at Hy4-flash's shape against a float32 reference on the GPU."""

import sys

import torch
from torch.utils.cpp_extension import load

ext = load("hy4_attn_check", [sys.argv[1] + "/tools/hy4_flash/attn_check.cu"], extra_include_paths=[sys.argv[1] + "/zig/kernels/cuda"],
           extra_cuda_cflags=["-O3", "-arch=sm_121a", "--use_fast_math"], verbose=False)


def run(rows, heads, keys_n, sparse, with_sink, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qa = torch.randn(rows, heads, 576, device="cuda", generator=g).bfloat16()
    lc = torch.randn(keys_n, 576, device="cuda", generator=g).bfloat16()
    sink = (torch.randn(heads, device="cuda", generator=g) * 2 + 6).float()
    scale = 256 ** -0.5
    start = keys_n - rows
    if sparse:  # a top-k list a row (some -1 padding), as the indexer gives it
        w = 2048
        tok = torch.full((rows, w), -1, dtype=torch.int32, device="cuda")
        cnt = torch.zeros(rows, dtype=torch.int32, device="cuda")
        for r in range(rows):
            visible = start + r + 1
            pick = torch.randperm(visible, device="cuda", generator=g)[: min(visible, w - 7)].sort().values
            tok[r, : len(pick)] = pick.int()
            cnt[r] = len(pick)
        keys = tok.long()
        nch = (w + 511) // 512
    else:
        tok = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        keys = torch.arange(keys_n, device="cuda").expand(rows, -1).clone()
        keys[keys > (start + torch.arange(rows, device="cuda")).unsqueeze(1)] = -1
        nch = (keys_n + 511) // 512
    pos = torch.tensor([start], dtype=torch.int32, device="cuda")
    out = ext.attend(qa, lc, pos, tok, cnt, sink, scale, nch, sparse, with_sink).float()
    want = reference_masked(qa, lc, keys, sink if with_sink else None, scale)
    err = (out - want).abs().max().item()
    rel = ((out - want).norm() / want.norm()).item()
    no_sink = reference_masked(qa, lc, keys, None, scale)
    return err, rel, (out - no_sink).abs().max().item()


def reference_masked(qa, lc, keys, sink, scale):
    k = keys.clamp(min=0)
    s = torch.einsum("rhd,rkd->rhk", qa.float(), lc.float()[k]) * scale
    s = s.masked_fill((keys < 0).unsqueeze(1), float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink.view(1, -1, 1).expand(s.shape[0], -1, 1)], -1)
    p = torch.softmax(s, -1)[..., : keys.shape[1]]
    return torch.einsum("rhk,rkd->rhd", p, lc.float()[k][..., :512])


for heads in (32, 64):
    for sparse in (False, True):
        for with_sink in (True, False):
            # 3,000 keys: a sparse row picks 2,041 of up to 3,000 visible ones, a real subset
            err, rel, vs_plain = run(rows=8, heads=heads, keys_n=3000, sparse=sparse, with_sink=with_sink)
            print(f"heads {heads:2d} {'sparse' if sparse else 'dense '} sink {with_sink!s:5}: max abs err {err:.2e}, "
                  f"rel {rel:.2e}, vs the no-sink reference {vs_plain:.2e}")
