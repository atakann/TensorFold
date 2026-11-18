"""Dev-time check: the reference's layer attention against absorb, the chunk body with the sink merge and expand."""

import sys

import torch
from safetensors.torch import load_file
from torch.utils.cpp_extension import load

src, tiny = sys.argv[1], sys.argv[2]
ext = load("hy4_attn_check", [src + "/tools/hy4_flash/attn_check.cu"], extra_include_paths=[src + "/zig/kernels/cuda"],
           extra_cuda_cflags=["-O3", "-arch=sm_121a", "--use_fast_math"])
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import apply_rotary_pos_emb  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32).cuda().eval()
ids = load_file(tiny + "/reference.safetensors")["input_ids"].long().cuda()[None]
seen = {}


def grab(i):
    def pre(mod, args, kwargs):
        seen[i] = {"x": (args[0] if args else kwargs["hidden_states"]).detach(), "pe": kwargs["position_embeddings"]}

    def post(mod, args, kwargs, out):
        seen[i]["out"] = out[0].detach()

    return pre, post


layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
for i, layer in enumerate(layers):
    pre, post = grab(i)
    layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    layer.self_attn.register_forward_hook(post, with_kwargs=True)
with torch.no_grad():
    model(input_ids=ids, use_cache=False)


def rms(x, w, eps):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def ours(i, a, x, pe):
    s, H, eps = x.shape[1], a.num_heads, a.config.rms_norm_eps
    bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
    x = x[0]
    q = bf(a.q_b_proj(bf(rms(a.q_a_proj(x), a.q_a_layernorm.weight, eps)).float())).view(s, H, 256)
    kv = a.kv_a_proj_with_mqa(x)
    lat = bf(rms(kv[:, :512], a.kv_a_layernorm.weight, eps))
    cos, sin = pe
    q_rot, k_rot = apply_rotary_pos_emb(q[None, :, :, 192:].transpose(1, 2).float(), kv[None, None, :, 512:], cos, sin)
    qa = torch.zeros(s, H, 576, dtype=torch.bfloat16, device="cuda")
    wkv = bf(a.kv_b_proj.weight).view(H, 448, 512)
    ext.absorb(q.contiguous(), wkv[:, :192].contiguous(), qa)
    qa[:, :, 512:] = bf(q_rot[0].transpose(0, 1))
    lc = torch.cat([lat, bf(k_rot[0, 0])], -1).contiguous()
    sliding = a.attention_type == "sliding_attention"
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    if sliding:  # the last 512 keys as a token list a row, live entries first, as vLLM's window indexer gives them
        w = 512
        tok = torch.zeros(s, w, dtype=torch.int32, device="cuda")
        for rr in range(s):
            keys = torch.arange(max(0, rr - w + 1), rr + 1, device="cuda", dtype=torch.int32)
            tok[rr, : len(keys)] = keys
        cnt = torch.clamp(torch.arange(1, s + 1, device="cuda"), max=w).int()
        out = ext.attend(qa, lc, pos, tok.contiguous(), cnt.contiguous(), a.sinks.float().contiguous(), a.scaling, 1, True, True)
    else:
        nch = (s + 511) // 512
        dummy = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
        out = ext.attend(qa, lc, pos, dummy, dummy[0], a.sinks.float().contiguous(), a.scaling, nch, False, True)
    o = ext.expand(out.contiguous(), wkv[:, 192:].contiguous()).float()
    o = o * torch.sigmoid(a.gate_proj(x).view(s, H, 256))
    return a.post_proj_norm(a.o_proj(o.reshape(s, -1)))


for i, layer in enumerate(layers):
    a = layer.self_attn
    with torch.no_grad():
        got = ours(i, a, seen[i]["x"], seen[i]["pe"])
    want = seen[i]["out"][0]
    rel = ((got - want).norm() / want.norm()).item()
    print(f"layer {i} {a.attention_type:18s} heads {a.num_heads}: rel err {rel:.2e}, max abs {(got - want).abs().max().item():.2e}")
