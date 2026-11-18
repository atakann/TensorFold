"""Dev-time check: the reference's Engram layer against engram_check.cu, with its bf16 table and with fp8 rows."""

import sys

import torch
from safetensors.torch import load_file
from torch.utils.cpp_extension import load

src, tiny = sys.argv[1], sys.argv[2]
ext = load("hy4_engram_check", [src + "/tools/hy4_flash/engram_check.cu"], extra_cuda_cflags=["-O3", "-arch=sm_121a"])
from transformers import AutoModelForImageTextToText  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).cuda().eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long().cuda()[None]
text = model.model.language_model if hasattr(model.model, "language_model") else model.model
seen = []
for layer in text.layers:
    if layer.engram_layer is not None:
        layer.engram_layer.register_forward_hook(lambda mod, args, out: seen.append((mod, args, out)))
with torch.no_grad():
    model(input_ids=ids, use_cache=False)


def fp8_rows(table):  # engram_fp8.py's export: e4m3 rows, a bf16 scale each (amax / 448)
    sc = (table.float().abs().amax(1) / 448).clamp(min=1e-12).bfloat16()
    return (table.float() / sc.float()[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), sc


for mod, (x, ngrams, mask), want in seen:
    with torch.no_grad():
        hashes = mod.hash_ngrams(ngrams)[0].int().contiguous()
        emb = mod.multi_head_embedding
        offsets = emb.offsets.long().contiguous()
        norm_w = torch.stack([n.weight for n in mod.lookup_per_head_norms]).float().contiguous()
        eps = torch.finfo(torch.float32).eps  # nn.RMSNorm's default at the reference's dtype
        for kind in ("bf16 table", "fp8 rows"):
            table = emb.embedding.weight.bfloat16().contiguous()
            scales = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
            if kind == "fp8 rows":
                table, scales = fp8_rows(table)
            e = ext.rows(hashes, offsets, table.contiguous(), scales.contiguous(), norm_w, eps).float()
            key = torch.stack([p(e) for p in mod.key_projs], 1).contiguous()
            value = mod.value_proj(e).contiguous()
            xb = x[0].bfloat16().contiguous()
            nk = torch.stack([n.weight for n in mod.norm_key]).float().contiguous()
            nq = torch.stack([n.weight for n in mod.norm_query]).float().contiguous()
            hc = torch.stack(list(mod.unified_per_hc_scales)).float().contiguous()
            got = ext.mix(key, value, xb, nk, nq, mod.unified_pre_gamma.float().contiguous(), hc, eps)
            ref = want[0]
            rel = ((got - ref).norm() / ref.norm()).item()
            print(f"Engram layer {mod.layer_id}, {kind}: residual rel err {rel:.2e}")

# where the difference comes from: the normalized rows against the reference's own, at each stage
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import _fp8_round_trip  # noqa: E402

for mod, (x, ngrams, mask), want in seen:
    with torch.no_grad():
        hashes = mod.hash_ngrams(ngrams)
        raw = mod.multi_head_embedding(hashes)
        rt = _fp8_round_trip(raw)
        e_ref = torch.stack([n(rt[:, :, h]) for h, n in enumerate(mod.lookup_per_head_norms)], 2).flatten(-2)[0]
        norm_w = torch.stack([n.weight for n in mod.lookup_per_head_norms]).float().contiguous()
        e = ext.rows(hashes[0].int().contiguous(), mod.multi_head_embedding.offsets.long().contiguous(),
                     mod.multi_head_embedding.embedding.weight.bfloat16().contiguous(),
                     torch.zeros(1, device="cuda", dtype=torch.bfloat16), norm_w, torch.finfo(torch.float32).eps).float()
        print(f"rows: rel err {((e - e_ref).norm() / e_ref.norm()).item():.2e}; "
              f"round trip vs raw {((rt - raw).norm() / raw.norm()).item():.2e}")
        key = torch.stack([p(e_ref) for p in mod.key_projs], 1).contiguous()
        value = mod.value_proj(e_ref).contiguous()
        got = ext.mix(key, value, x[0].bfloat16().contiguous(),
                      torch.stack([n.weight for n in mod.norm_key]).float().contiguous(),
                      torch.stack([n.weight for n in mod.norm_query]).float().contiguous(),
                      mod.unified_pre_gamma.float().contiguous(),
                      torch.stack(list(mod.unified_per_hc_scales)).float().contiguous(), torch.finfo(torch.float32).eps)
        print(f"mix from the reference's rows: rel err {((got - want[0]).norm() / want[0].norm()).item():.2e}")

# the shipped model runs in bf16: its round trip takes a bf16 scale, as the kernel does
for mod, (x, ngrams, mask), want in seen:
    with torch.no_grad():
        hashes = mod.hash_ngrams(ngrams)
        raw = mod.multi_head_embedding(hashes)
        rt16 = _fp8_round_trip(raw.bfloat16()).float()
        e16 = torch.stack([n(rt16[:, :, h]) for h, n in enumerate(mod.lookup_per_head_norms)], 2).flatten(-2)[0]
        e = ext.rows(hashes[0].int().contiguous(), mod.multi_head_embedding.offsets.long().contiguous(),
                     mod.multi_head_embedding.embedding.weight.bfloat16().contiguous(),
                     torch.zeros(1, device="cuda", dtype=torch.bfloat16),
                     torch.stack([n.weight for n in mod.lookup_per_head_norms]).float().contiguous(),
                     torch.finfo(torch.float32).eps).float()
        print(f"rows against the bf16 round trip: rel err {((e - e16).norm() / e16.norm()).item():.2e}")
