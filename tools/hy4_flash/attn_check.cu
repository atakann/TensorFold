// Dev-time check: the shared MLA chunk body at Hy4-flash's shape (576-wide rows) and a chunk merge with a learnable sink.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include "common/mla.cuh"

__global__ void __launch_bounds__(256) hy4_chunks(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos,
                                                  const int* tok, const int* cnt, float* po, float* pm, float* pl,
                                                  int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn::chunks_wide<16, false, 64>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0);
}

// tf_glm_merge's loop, then the sink: out = o / (l + exp(sink - m)), the sink taking no value row.
__global__ void __launch_bounds__(128) hy4_merge_sink(const float* po, const float* pm, const float* pl,
                                                      const float* sink, __nv_bfloat16* out, int64_t rows,
                                                      int64_t heads, int nch) {
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  if (sink) l = __fadd_rn(l, m == -INFINITY ? 0.0f : glm_triton::exp_approx(sink[h] - m));
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}

torch::Tensor attend(torch::Tensor qa, torch::Tensor lc, torch::Tensor pos, torch::Tensor tok, torch::Tensor cnt,
                     torch::Tensor sink, double scale, int64_t nch, bool sparse, bool use_sink) {
  const int64_t rows = qa.size(0), heads = qa.size(1), w = sparse ? tok.size(1) : 0;
  auto f = torch::TensorOptions().dtype(torch::kFloat32).device(qa.device());
  auto po = torch::zeros({nch, rows, heads, 512}, f), pm = torch::zeros({nch, rows, heads}, f),
       pl = torch::zeros({nch, rows, heads}, f);
  auto out = torch::zeros({rows, heads, 512}, qa.options());
  constexpr int smem = glm_attn::wide_smem<16, 64>();
  cudaFuncSetAttribute(hy4_chunks, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  auto bf = [](torch::Tensor t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  hy4_chunks<<<dim3(rows, (heads + 15) / 16, nch), 256, smem>>>(
      bf(qa), bf(lc), pos.data_ptr<int>(), sparse ? tok.data_ptr<int>() : nullptr, sparse ? cnt.data_ptr<int>() : nullptr,
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), rows, heads, w, (float)scale, sparse);
  hy4_merge_sink<<<dim3(rows, heads), 128>>>(po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
                                             use_sink ? sink.data_ptr<float>() : nullptr, bf(out), rows, heads, nch);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attend", &attend); }
