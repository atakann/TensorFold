"""The reference's attention inputs (bf16) and outputs (float32) of each layer of the tiny checkpoint, for hy4-attn-check."""

import os
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

tiny, out = sys.argv[1], sys.argv[2]
model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).cuda().eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long().cuda()[None]
text = model.model.language_model if hasattr(model.model, "language_model") else model.model
dump = {}
for i, layer in enumerate(text.layers):
    def pre(mod, args, kwargs, i=i):
        dump[f"x.{i}"] = (args[0] if args else kwargs["hidden_states"])[0].bfloat16().contiguous()

    def post(mod, args, kwargs, o, i=i):
        dump[f"out.{i}"] = o[0][0].float().contiguous()

    layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    layer.self_attn.register_forward_hook(post, with_kwargs=True)
with torch.no_grad():
    # the attention input rounded to bf16 is what the kernels read: feed that to the reference too
    model(input_ids=ids, use_cache=False)
os.makedirs(out, exist_ok=True)
save_file({k: v.cpu() for k, v in dump.items()}, out + "/model.safetensors")
print({k: tuple(v.shape) for k, v in dump.items()})
