"""Dev-time check: the reference's MoE router against tf_glm_topk; the same experts and weights for every row."""

import sys

import torch
from safetensors.torch import load_file
from torch.utils.cpp_extension import load

src, tiny = sys.argv[1], sys.argv[2]
ext = load("hy4_moe_check", [src + "/tools/hy4_flash/moe_check.cu"], extra_include_paths=[src + "/zig/kernels/cuda"],
           extra_cuda_cflags=["-O3", "-arch=sm_121a"])
from transformers import AutoModelForImageTextToText  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).cuda().eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long().cuda()[None]
text = model.model.language_model if hasattr(model.model, "language_model") else model.model
seen = []
for i, layer in enumerate(text.layers):
    if hasattr(layer.mlp, "gate"):
        layer.mlp.gate.register_forward_hook(lambda mod, args, out, i=i: seen.append((i, mod, out)))
with torch.no_grad():
    model(input_ids=ids, use_cache=False)
for i, gate, (logits, want_w, want_i) in seen:
    pick, wts = ext.topk(logits.float().contiguous(), gate.e_score_correction_bias.float().contiguous(), gate.top_k,
                         gate.routed_scaling_factor)
    got_i, got_w = pick[:, : gate.top_k].long(), wts[:, : gate.top_k]
    order_got, order_want = got_i.sort(-1), want_i.sort(-1)
    same = (order_got.values == order_want.values).all(-1)
    w_err = (got_w.gather(1, order_got.indices) - want_w.gather(1, order_want.indices)).abs().max().item()
    print(f"layer {i}: same experts on {int(same.sum())}/{len(same)} rows, max weight err {w_err:.2e}, "
          f"shared slot id {int(pick[0, gate.top_k])} weight {float(wts[0, gate.top_k])}")
