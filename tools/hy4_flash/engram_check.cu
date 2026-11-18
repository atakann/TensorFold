// Dev-time check: Hy4-flash's Engram on the GPU: table rows (fp8 with a row scale, or bf16 rounded to fp8 the same
// way), per-head RMS norms, then the per-stream gates and the residual, around projections done elsewhere.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr int DH = 96, HEADS = 16, S = 4;

__device__ __forceinline__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.0f;
  for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += red[w];
  __syncthreads();
  return s;
}

// Block (row, head) of 96 threads: the head's row, as fp8 times its scale, normed by the head's weight.
__global__ void hy4_engram_rows(const int* hash, const int64_t* offsets, const unsigned char* table8,
                                const __nv_bfloat16* scales, const __nv_bfloat16* table16, const float* norm_w,
                                float eps, __nv_bfloat16* e, int64_t rows) {
  __shared__ float red[4];
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int d = threadIdx.x;
  const int id = hash[r * HEADS + h];
  float v = 0.0f;
  if (id >= 0) {
    const int64_t row = offsets[h] + id;
    if (table8) {
      v = __half2float(__half(__nv_cvt_fp8_to_halfraw(table8[row * DH + d], __NV_E4M3))) * __bfloat162float(scales[row]);
    } else {  // the reference's round trip: e4m3 at the row's amax / 448, back to the row's dtype
      const float x = __bfloat162float(table16[row * DH + d]);
      __shared__ float amax[4];
      float a = fabsf(x);
      for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
      if ((d & 31) == 0) amax[d >> 5] = a;
      __syncthreads();
      const float scale = fmaxf(fmaxf(amax[0], amax[1]), fmaxf(amax[2], 1e-12f)) / 448.0f;
      const __nv_bfloat16 sb = __float2bfloat16_rn(scale);
      const float q = fminf(fmaxf(__bfloat162float(__float2bfloat16_rn(x / __bfloat162float(sb))), -448.0f), 448.0f);
      const __nv_fp8_e4m3 c(q);
      v = __bfloat162float(__float2bfloat16_rn(float(c))) * __bfloat162float(sb);
    }
  }
  const float ms = block_sum(v * v, red) / DH;
  e[(r * HEADS + h) * DH + d] = __float2bfloat16_rn(v * rsqrtf(ms + eps) * norm_w[h * DH + d]);
}

// Block (row) of 256: per stream k, gate_k = sigmoid(signed sqrt(norm_key_k(key_k) . norm_query_k(x_k) / sqrt(D)));
// main = pre_gamma rmsnorm(value); residual_k = gate_k main + scale_k main.
__global__ void __launch_bounds__(256) hy4_engram_mix(const float* key, const float* value, const __nv_bfloat16* x,
                                                      const float* nk, const float* nq, const float* gamma,
                                                      const float* hc_scale, float eps, float* out, int64_t D) {
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  const float* vr = value + r * D;
  float ss = 0.0f;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) ss += vr[d] * vr[d];
  const float vinv = rsqrtf(block_sum(ss, red) / D + eps);
  for (int k = 0; k < S; ++k) {
    const float* kr = key + (r * S + k) * D;
    const __nv_bfloat16* xr = x + (r * S + k) * D;
    float kk = 0.0f, qq = 0.0f;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      kk += kr[d] * kr[d];
      const float q = __bfloat162float(xr[d]);
      qq += q * q;
    }
    const float kinv = rsqrtf(block_sum(kk, red) / D + eps), qinv = rsqrtf(block_sum(qq, red) / D + eps);
    float dot = 0.0f;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x)
      dot += (kr[d] * kinv * nk[k * D + d]) * (__bfloat162float(xr[d]) * qinv * nq[k * D + d]);
    const float g0 = block_sum(dot, red) * rsqrtf((float)D);
    const float g = 1.0f / (1.0f + __expf(-copysignf(sqrtf(fmaxf(fabsf(g0), 1e-6f)), g0)));
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      const float main = gamma[d] * vr[d] * vinv;
      out[(r * S + k) * D + d] = g * main + hc_scale[k * D + d] * main;
    }
  }
}

torch::Tensor rows(torch::Tensor hash, torch::Tensor offsets, torch::Tensor table, torch::Tensor scales,
                   torch::Tensor norm_w, double eps) {
  const int64_t n = hash.size(0);
  auto e = torch::empty({n, HEADS * DH}, norm_w.options().dtype(torch::kBFloat16));
  const bool fp8 = table.scalar_type() == torch::kFloat8_e4m3fn;
  hy4_engram_rows<<<dim3(n, HEADS), DH>>>(
      hash.data_ptr<int>(), offsets.data_ptr<int64_t>(), fp8 ? (const unsigned char*)table.data_ptr() : nullptr,
      fp8 ? (const __nv_bfloat16*)scales.data_ptr() : nullptr, fp8 ? nullptr : (const __nv_bfloat16*)table.data_ptr(),
      norm_w.data_ptr<float>(), eps, (__nv_bfloat16*)e.data_ptr(), n);
  return e;
}

torch::Tensor mix(torch::Tensor key, torch::Tensor value, torch::Tensor x, torch::Tensor nk, torch::Tensor nq,
                  torch::Tensor gamma, torch::Tensor hc_scale, double eps) {
  const int64_t n = x.size(0), D = x.size(2);
  auto out = torch::empty({n, S, D}, key.options());
  hy4_engram_mix<<<n, 256>>>(key.data_ptr<float>(), value.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(),
                             nk.data_ptr<float>(), nq.data_ptr<float>(), gamma.data_ptr<float>(), hc_scale.data_ptr<float>(),
                             eps, out.data_ptr<float>(), D);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rows", &rows);
  m.def("mix", &mix);
}
