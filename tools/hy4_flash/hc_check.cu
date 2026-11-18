// Dev-time check: Hy4-flash's identity hyper-connections (no Sinkhorn mix): the pre side, the post side and the head.
#include <torch/extension.h>
#include <cuda_bf16.h>

namespace hy4_hc {

constexpr int S = 4, NT = 256;

__device__ __forceinline__ float sigmoid(float v) { return 1.0f / (1.0f + __expf(-v)); }

// One row: M mixes (fn rows) over the S * D stream values, times their inverse RMS, in fp32.
template <int M>
__device__ void mixes(const __nv_bfloat16* x, const float* fn, int64_t D, float* mix) {
  __shared__ float red[M + 1][NT / 32];
  float acc[M + 1] = {};
  const int64_t K = S * D;
  for (int64_t k = threadIdx.x; k < K; k += NT) {
    const float v = __bfloat162float(x[k]);
    for (int m = 0; m < M; ++m) acc[m] = __fmaf_rn(fn[m * K + k], v, acc[m]);
    acc[M] = __fmaf_rn(v, v, acc[M]);
  }
  for (int m = 0; m <= M; ++m)
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
  if ((threadIdx.x & 31) == 0)
    for (int m = 0; m <= M; ++m) red[m][threadIdx.x >> 5] = acc[m];
  __syncthreads();
  if (threadIdx.x < M + 1) {
    float s = 0.0f;
    for (int w = 0; w < NT / 32; ++w) s += red[threadIdx.x][w];
    red[threadIdx.x][0] = s;
  }
  __syncthreads();
  for (int m = 0; m < M; ++m) mix[m] = red[m][0];
  mix[M] = red[M][0];
}

}  // namespace hy4_hc

// attn_hc / ffn_hc: out [rows][D] = sum_k pre_k x_k, post [rows][S]; pre = sigmoid(mix s0 + b) + eps, post = mag sigmoid(..) + eps.
__global__ void __launch_bounds__(256) hy4_hc_pre(const __nv_bfloat16* x, const float* fn, const float* base,
                                                  const float* scale, __nv_bfloat16* out, float* post, int64_t D,
                                                  float eps, float hc_eps, float magnitude) {
  using namespace hy4_hc;
  const int64_t r = blockIdx.x;
  const __nv_bfloat16* xr = x + r * S * D;
  float mix[2 * S + 1];
  mixes<2 * S>(xr, fn, D, mix);
  const float rinv = rsqrtf(mix[2 * S] / (float)(S * D) + eps);
  float pre[S];
  for (int k = 0; k < S; ++k) {
    pre[k] = sigmoid(mix[k] * rinv * scale[0] + base[k]) + hc_eps;
    if (threadIdx.x == 0) post[r * S + k] = magnitude * sigmoid(mix[S + k] * rinv * scale[1] + base[S + k]) + hc_eps;
  }
  for (int64_t d = threadIdx.x; d < D; d += NT) {
    float s = 0.0f;
    for (int k = 0; k < S; ++k) s = __fmaf_rn(pre[k], __bfloat162float(xr[k * D + d]), s);
    out[r * D + d] = __float2bfloat16_rn(s);
  }
}

// The layer's residual update: x_k = post_k f + x_k in fp32, rounded to the stream dtype.
__global__ void __launch_bounds__(256) hy4_hc_post(__nv_bfloat16* x, const __nv_bfloat16* f, const float* post,
                                                   int64_t D) {
  const int64_t r = blockIdx.x;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float fv = __bfloat162float(f[r * D + d]);
    for (int k = 0; k < hy4_hc::S; ++k) {
      __nv_bfloat16* p = x + (r * hy4_hc::S + k) * D + d;
      *p = __float2bfloat16_rn(__fmaf_rn(post[r * hy4_hc::S + k], fv, __bfloat162float(*p)));
    }
  }
}

// hc_head: out [rows][D] = sum_k (sigmoid(mix_k scale + base_k) + eps) x_k.
__global__ void __launch_bounds__(256) hy4_hc_head(const __nv_bfloat16* x, const float* fn, const float* base,
                                                   const float* scale, __nv_bfloat16* out, int64_t D, float eps,
                                                   float hc_eps) {
  using namespace hy4_hc;
  const int64_t r = blockIdx.x;
  const __nv_bfloat16* xr = x + r * S * D;
  float mix[S + 1];
  mixes<S>(xr, fn, D, mix);
  const float rinv = rsqrtf(mix[S] / (float)(S * D) + eps);
  float pre[S];
  for (int k = 0; k < S; ++k) pre[k] = sigmoid(mix[k] * rinv * scale[0] + base[k]) + hc_eps;
  for (int64_t d = threadIdx.x; d < D; d += NT) {
    float s = 0.0f;
    for (int k = 0; k < S; ++k) s = __fmaf_rn(pre[k], __bfloat162float(xr[k * D + d]), s);
    out[r * D + d] = __float2bfloat16_rn(s);
  }
}

static __nv_bfloat16* bfp(torch::Tensor t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

std::vector<torch::Tensor> pre(torch::Tensor x, torch::Tensor fn, torch::Tensor base, torch::Tensor scale, double eps,
                               double hc_eps, double magnitude) {
  const int64_t rows = x.size(0), D = x.size(2);
  auto out = torch::empty({rows, D}, x.options());
  auto post = torch::empty({rows, 4}, fn.options());
  hy4_hc_pre<<<rows, 256>>>(bfp(x), fn.data_ptr<float>(), base.data_ptr<float>(), scale.data_ptr<float>(), bfp(out),
                            post.data_ptr<float>(), D, eps, hc_eps, magnitude);
  return {out, post};
}

void post_update(torch::Tensor x, torch::Tensor f, torch::Tensor post) {
  hy4_hc_post<<<x.size(0), 256>>>(bfp(x), bfp(f), post.data_ptr<float>(), x.size(2));
}

torch::Tensor head(torch::Tensor x, torch::Tensor fn, torch::Tensor base, torch::Tensor scale, double eps, double hc_eps) {
  const int64_t rows = x.size(0), D = x.size(2);
  auto out = torch::empty({rows, D}, x.options());
  hy4_hc_head<<<rows, 256>>>(bfp(x), fn.data_ptr<float>(), base.data_ptr<float>(), scale.data_ptr<float>(), bfp(out), D,
                             eps, hc_eps);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pre", &pre);
  m.def("post_update", &post_update);
  m.def("head", &head);
}
