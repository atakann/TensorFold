"""real_reference.py <real6 dir> <tokenizer dir> <out dir>: the first 6 layers of the NVFP4 checkpoint, dequantized to
bf16 (e2m1 code x e4m3 block scale x fp32 tensor scale, Engram rows x their scale), run by transformers in bf16 on a
chat prompt: the hidden streams after each layer, the logits and the Engram hash ids, for hy4-real-check.
The model is built on the GPU first and each parameter filled in place, so the weights are held once."""

import glob
import importlib
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

src, tok_dir, out = sys.argv[1:4]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).cuda()

where = {}
files = [safe_open(p, "pt", device="cuda") for p in sorted(glob.glob(f"{src}/model-*.safetensors"))]
for f in files:
    for k in f.keys():
        where[k] = f


def get(k):
    return where[k].get_tensor(k)


def weight(k):
    """A stored tensor as bf16: NVFP4 codes dequantized (low nibble first), FP8 Engram rows times their scale."""
    v = get(k)
    b = k[:-6]
    if v.dtype == torch.uint8 and b + "weight_scale" in where:
        q = torch.stack((E2M1[(v & 0xF).long()], E2M1[(v >> 4).long()]), dim=-1).flatten(-2)
        s = get(b + "weight_scale").float().repeat_interleave(16, dim=-1)
        return (q * s * get(b + "weight_scale_2").float()).bfloat16()
    if k + "_scale_inv" in where:
        return (v.float() * get(k + "_scale_inv").float()).bfloat16()
    return v


class Fp8Rows(torch.nn.Module):
    """The Engram table kept as stored (fp8 rows, a bf16 scale a row): only the rows a prompt reads become bf16."""

    def forward(self, idx):
        return (self.w8[idx].float() * self.sc[idx].float().reshape(*idx.shape, 1)).bfloat16()


def engram_init(self, head_vocab_sizes, embedding_dim):
    torch.nn.Module.__init__(self)
    self.head_vocab_sizes = head_vocab_sizes
    self.offsets = torch.nn.Buffer(self.head_offsets(), persistent=False)
    self.embedding = Fp8Rows()


hy_mod = importlib.import_module("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
hy_mod.HYV4FlashEngramEmbedding.__init__ = engram_init
# transformers' conversion_mapping for hy_v4_flash, from its module names back to the checkpoint's
RENAMES = [(".self_attn.sinks", ".self_attn.learnable_sink_param"), (".self_attn.gate_proj.weight", ".self_attn.linear_gate.weight")]
RENAMES += [(f".{m}_hc.{x}", f".hc_{l}_layer.hc_pre.hc_{x}") for m, l in (("attn", "attn"), ("ffn", "mlp")) for x in ("fn", "base", "scale")]
RENAMES += [(f".hc_head.hc_{x}", f".hc_head.hc_head_{x}") for x in ("fn", "base", "scale")]
cfg = AutoConfig.from_pretrained(src)
for c in (cfg, getattr(cfg, "text_config", None)):
    if c is not None and hasattr(c, "quantization_config"):
        delattr(c, "quantization_config")
with torch.device("cuda"):
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16).eval()
filled, left = 0, []
with torch.no_grad():
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        k = name.replace("model.language_model.", "model.")
        for model_name, stored in RENAMES:
            k = k.replace(model_name, stored)
        if k.endswith("experts.gate_up_proj") or k.endswith("experts.down_proj"):
            pre = k.rsplit(".", 1)[0] + "."
            for e in range(p.shape[0]):
                if k.endswith("gate_up_proj"):
                    p[e].copy_(torch.cat((weight(f"{pre}{e}.gate_proj.weight"), weight(f"{pre}{e}.up_proj.weight"))))
                else:
                    p[e].copy_(weight(f"{pre}{e}.down_proj.weight"))
            filled += 1
        elif k in where:
            p.copy_(weight(k).to(p.dtype))
            filled += 1
        elif k.startswith(("model.layers.", "lm_head", "model.embed", "model.norm", "model.hc_head")) and name in dict(model.named_parameters()):
            left.append(k)
    print("filled", filled, "text parameters not in the checkpoint:", left[:8])
    assert not left
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li, layer in enumerate(text.layers):
        e = layer.engram_layer
        if e is not None:
            k = f"model.layers.{li}.engram_layer.multi_head_embedding.embedding.weight"
            e.multi_head_embedding.embedding.w8 = get(k)
            e.multi_head_embedding.embedding.sc = get(k + "_scale_inv")  # the multipliers are drawn from the loaded map, as _init_weights does
            e.hash_multipliers.copy_(e.draw_hash_multipliers(int(e.lookup_table.max()) + 1))
            e.multi_head_embedding.offsets.copy_(e.multi_head_embedding.head_offsets())
tok = AutoTokenizer.from_pretrained(tok_dir)
msgs = [{"role": "user", "content": "Explain in three short paragraphs why a mixture-of-experts model can run faster than a "
         "dense model of the same size, and what limits that speed on a machine where the GPU and CPU share memory."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", reasoning_effort="no_think")
ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids).cuda()
dump = {"input_ids": ids[0].to(torch.int32).cpu().contiguous()}
for i, layer in enumerate(text.layers):
    if layer.engram_layer is not None:
        e = layer.engram_layer
        dump[f"engram.{i}.hash"] = e.hash_ngrams(text.build_ngram_tokens(ids, None))[0].to(torch.int32).cpu().contiguous()
        dump[f"engram.{i}.mult"] = e.hash_multipliers.cpu().clone()
layer1 = text.layers[1]
layer1.engram_layer.register_forward_hook(lambda m, a, o: dump.__setitem__("engram.1.out", o[0].float().cpu().contiguous()))
layer1.engram_layer.value_proj.register_forward_hook(lambda m, a, o: dump.update({"engram.1.e": a[0][0].float().cpu().contiguous(), "engram.1.value": o[0].float().cpu().contiguous()}))
layer1.mlp.register_forward_hook(lambda m, a, o: dump.update({"mlp.1.in": a[0][0].float().cpu().contiguous(), "mlp.1.out": o[0].float().cpu().contiguous()}))
with torch.no_grad():
    o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
dump["logits"] = o.logits[0].float().cpu().contiguous()
for i, h in enumerate(o.hidden_states):
    dump[f"hidden.{i}"] = h[0].float().cpu().contiguous()
os.makedirs(out, exist_ok=True)
save_file(dump, f"{out}/model.safetensors")
print({k: tuple(v.shape) for k, v in dump.items()}, "next tokens:", [tok.decode(t) for t in o.logits[0, -1].topk(5).indices])
