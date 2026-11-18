"""Dev-time check: the reference's hyper-connections (attn_hc, ffn_hc, hc_head) against the CUDA kernels in hc_check.cu."""

import sys

import torch
from safetensors.torch import load_file
from torch.utils.cpp_extension import load

src, tiny = sys.argv[1], sys.argv[2]
ext = load("hy4_hc_check", [src + "/tools/hy4_flash/hc_check.cu"], extra_cuda_cflags=["-O3", "-arch=sm_121a"])
from transformers import AutoModelForImageTextToText  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).cuda().eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long().cuda()[None]
text = model.model.language_model if hasattr(model.model, "language_model") else model.model
seen = []


def hook(name):
    def post(mod, args, out):
        seen.append((name, mod, args[0].detach(), out))

    return post


for i, layer in enumerate(text.layers):
    layer.attn_hc.register_forward_hook(hook(f"layer {i} attn_hc"))
    layer.ffn_hc.register_forward_hook(hook(f"layer {i} ffn_hc"))
text.hc_head.register_forward_hook(hook("hc_head"))
with torch.no_grad():
    model(input_ids=ids, use_cache=False)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


eps = text.config.rms_norm_eps
for name, mod, x, out in seen:
    xb = x[0].to(torch.bfloat16).contiguous()
    if name == "hc_head":
        got = ext.head(xb, mod.hc_fn.float().contiguous(), mod.hc_base.float().contiguous(), mod.hc_scale.float().contiguous(), eps, mod.eps)
        print(f"{name:16s}: out rel err {rel(got, out[0]):.2e}")
        continue
    got, post = ext.pre(xb, mod.fn.float().contiguous(), mod.base.float().contiguous(), mod.scale.float().contiguous(),
                        eps, mod.hc_eps, mod.hc_post_magnitude)
    want_post, want_out = out
    print(f"{name:16s}: out rel err {rel(got, want_out[0]):.2e}, post rel err {rel(post, want_post[0]):.2e}")
