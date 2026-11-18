"""The tiny checkpoint's Engram inputs for hy4-forward-check: its layer's hash multipliers, head moduli and the
reference's hash ids of the reference prompt (the multipliers depend on the table size, so the tiny ones differ)."""

import os
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

tiny, out = sys.argv[1], sys.argv[2]
model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long()[None]
text = model.model.language_model if hasattr(model.model, "language_model") else model.model
dump = {}
for i, layer in enumerate(text.layers):
    e = layer.engram_layer
    if e is None:
        continue
    ngrams = text.build_ngram_tokens(ids, None)
    dump[f"engram.{i}.mult"] = e.hash_multipliers.clone()
    dump[f"engram.{i}.moduli"] = e.head_vocab_sizes.to(torch.int64).clone()
    dump[f"engram.{i}.hash"] = e.hash_ngrams(ngrams)[0].to(torch.int32).contiguous()
os.makedirs(out, exist_ok=True)
save_file(dump, out + "/model.safetensors")
print({k: (tuple(v.shape), v.flatten()[:3].tolist()) for k, v in dump.items()})
