// Hy4-flash: absorb and expand widths, the sink merge, hyper-connections, Engram (chunks: tf_mla_chunks_r16).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "common/mla.cuh"

// tf_glm_merge's loop, then the sink: out = o / (l + exp(sink - m)), the sink taking no value row.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_merge_sink(const float* po, const float* pm, const float* pl,
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

extern "C" __global__ void __launch_bounds__(128) tf_hy4_absorb_q_192(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa,
                                                    int64_t rows, int64_t heads, int64_t rb, int64_t qs, int64_t qaw) {
  tf_mla::absorb_q<192>(q, wk, qa, rows, heads, rb, qs, qaw);
}

extern "C" __global__ void __launch_bounds__(128) tf_hy4_expand_v_256(const __nv_bfloat16* ol, const __nv_bfloat16* wv,
                                                    __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t rb) {
  tf_mla::expand_v<256>(ol, wv, out, rows, heads, rb);
}

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

// attn_hc / ffn_hc: out = sum_k pre_k x_k; pre = sigmoid(mix s0 + b) + eps, post = magnitude sigmoid(..) + eps.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_hc_pre(const __nv_bfloat16* x, const float* fn, const float* base,
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
extern "C" __global__ void __launch_bounds__(256) tf_hy4_hc_post(__nv_bfloat16* x, const __nv_bfloat16* f, const float* post,
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
extern "C" __global__ void __launch_bounds__(256) tf_hy4_hc_head(const __nv_bfloat16* x, const float* fn, const float* base,
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

namespace hy4_engram {
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
extern "C" __global__ void tf_hy4_engram_rows(const int* hash, const int64_t* offsets, const unsigned char* table8,
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

// A row a block: gate_k = sigmoid(signed sqrt(nk(key_k) . nq(x_k) / sqrt D)); out_k = (gate_k + scale_k) main.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_engram_mix(const float* key, const float* value, const __nv_bfloat16* x,
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

}  // namespace hy4_engram

// A row a block: out = bf16(x rsqrt(mean(x^2) + eps) w), x fp32 (a dense_rows output) xs apart, out os apart.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_rmsnorm(const float* x, int64_t xs, const __nv_bfloat16* w,
                                                                __nv_bfloat16* out, int64_t os, int64_t n, float eps) {
  __shared__ float red[8];
  const float* xr = x + blockIdx.x * xs;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
  const float inv = rsqrtf(hy4_engram::block_sum(ss, red) / n + eps);
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    out[blockIdx.x * os + i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(xr[i] * inv)) * __bfloat162float(w[i]));
}

// A row a block: out = bf16(x), n values xs and os apart a row.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_cast(const float* x, int64_t xs, __nv_bfloat16* out,
                                                             int64_t os, int64_t n) {
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) out[blockIdx.x * os + i] = __float2bfloat16_rn(x[blockIdx.x * xs + i]);
}

// The elementwise output gate: o *= sigmoid(g), g fp32 laid out as o.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_gate(__nv_bfloat16* o, const float* g, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16_rn(__bfloat162float(o[i]) / (1.0f + __expf(-g[i])));
}

// tf_hy4_rmsnorm on bf16 rows: out = w bf16(x rsqrt(mean(x^2) + eps)), the reference's RMSNorm on a bf16 input.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_rmsnorm16(const __nv_bfloat16* x, int64_t xs,
                                                                  const __nv_bfloat16* w, __nv_bfloat16* out,
                                                                  int64_t os, int64_t n, float eps) {
  __shared__ float red[8];
  const __nv_bfloat16* xr = x + blockIdx.x * xs;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) ss += __bfloat162float(xr[i]) * __bfloat162float(xr[i]);
  const float inv = rsqrtf(hy4_engram::block_sum(ss, red) / n + eps);
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(xr[i]) * inv));
    out[blockIdx.x * os + i] = __float2bfloat16_rn(v * __bfloat162float(w[i]));
  }
}

// A row a block: out[r][i] += bias[i] (bf16), n values os apart a row.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_bias(float* out, int64_t os, const __nv_bfloat16* bias,
                                                             int64_t n) {
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) out[blockIdx.x * os + i] += __bfloat162float(bias[i]);
}

// x = bf16(x + f) over n values, x bf16 and f fp32 (Engram's residual into the streams).
extern "C" __global__ void __launch_bounds__(256) tf_hy4_add16(__nv_bfloat16* x, const float* f, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + f[i]);
}

// a += b over n fp32 values.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_add32(float* a, const float* b, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

// A row a block: out = bf16(silu(g) u), gate g then up u in gu's 2I columns, clamped at limit (0: none).
extern "C" __global__ void __launch_bounds__(256) tf_hy4_swiglu(const float* gu, __nv_bfloat16* out, int64_t I,
                                                               float limit) {
  const float* r = gu + blockIdx.x * 2 * I;
  for (int64_t i = threadIdx.x; i < I; i += blockDim.x) {
    float g = __bfloat162float(__float2bfloat16_rn(r[i])), u = __bfloat162float(__float2bfloat16_rn(r[I + i]));
    if (limit > 0.0f) {
      g = fminf(g, limit);
      u = fminf(fmaxf(u, -limit), limit);
    }
    const float a = __bfloat162float(__float2bfloat16_rn(g / (1.0f + __expf(-g))));
    out[blockIdx.x * I + i] = __float2bfloat16_rn(a * u);
  }
}

// A thread a row: sigmoid scores, the top k of score + bias, weights = scores normalized times scale.
extern "C" __global__ void tf_hy4_route(const float* logits, const float* bias, int* ids, float* wts, int64_t rows,
                                        int E, int K, float scale) {
  const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* l = logits + r * E;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < E; ++e) {
      bool taken = false;
      for (int j = 0; j < k; ++j) taken |= ids[r * K + j] == e;
      const float v = 1.0f / (1.0f + __expf(-l[e])) + bias[e];
      if (!taken && v > bv) bv = v, best = e;
    }
    ids[r * K + k] = best;
    wts[r * K + k] = 1.0f / (1.0f + __expf(-l[best]));
    sum += wts[r * K + k];
  }
  for (int k = 0; k < K; ++k) wts[r * K + k] = wts[r * K + k] / (sum + 1e-20f) * scale;
}

// a += b over n values, a fp32 and b bf16 (the shared expert's bf16 sums into the routed fp32 output).
extern "C" __global__ void __launch_bounds__(256) tf_hy4_addb(float* a, const __nv_bfloat16* b, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += __bfloat162float(b[i]);
}

// Expert parallel picks: ids lo..lo+count-1 to this rank's 0..count-1, the others to `count` (skipped) at weight 0.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_own(int* picks, float* wts, int64_t n, int lo, int count) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int e = picks[i] - lo;
  if (e < 0 || e >= count) {
    picks[i] = count;
    wts[i] = 0.0f;
  } else {
    picks[i] = e;
  }
}

// A row a block: row r at pos[0] + r reads keys first..p (at most `win`), as a token list for the chunk tiles.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_window(const int* pos, int* tok, int* cnt, int64_t win) {
  const int64_t r = blockIdx.x, p = (int64_t)pos[0] + r;
  const int64_t first = p + 1 >= win ? p + 1 - win : 0, n = p + 1 - first;
  if (threadIdx.x == 0) cnt[r] = (int)n;
  for (int64_t j = threadIdx.x; j < win; j += blockDim.x) tok[r * win + j] = j < n ? (int)(first + j) : 0;
}

// Rows [rows][w] bf16 into the cache at rows pos[0].. (a graph replays this at any position).
extern "C" __global__ void __launch_bounds__(256) tf_hy4_put(__nv_bfloat16* cache, const __nv_bfloat16* rows,
                                                            const int* pos, int64_t w) {
  const int64_t r = blockIdx.x, p = (int64_t)pos[0] + r;
  for (int64_t j = threadIdx.x; j < w; j += blockDim.x) cache[p * w + j] = rows[r * w + j];
}

// DFly's fusion: x = bf16(x + gate fused) over n values (gate bf16, fused fp32).
extern "C" __global__ void __launch_bounds__(256) tf_hy4_fuse(__nv_bfloat16* x, const float* fused,
                                                             const __nv_bfloat16* gate, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(gate[i]) * fused[i]);
}

// A row a block of 1024: the row's first maximum of n logits as (value, index + offset) for a vocabulary slice.
extern "C" __global__ void __launch_bounds__(1024) tf_hy4_argmax_local(const float* logits, int64_t n, int64_t offset,
                                                                      float2* out) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const float* r = logits + blockIdx.x * n;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    if (r[i] > bv) bv = r[i], bi = (int)i;
  for (int o = 16; o > 0; o >>= 1) {
    const float v = __shfl_xor_sync(0xffffffffu, bv, o);
    const int j = __shfl_xor_sync(0xffffffffu, bi, o);
    if (v > bv || (v == bv && j < bi)) bv = v, bi = j;
  }
  if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
      if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) bv = sv[w], bi = si[w];
    out[blockIdx.x] = make_float2(bv, __int_as_float(bi + (int)offset));
  }
}

// Each row's pick over the ranks' (value, index) pairs [world][rows]: the larger value, the lower index on a tie.
extern "C" __global__ void tf_hy4_argmax_pick(const float2* all, int world, int64_t rows, int* ids) {
  const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float2 b = all[r];
  for (int w = 1; w < world; ++w) {
    const float2 c = all[w * rows + r];
    if (c.x > b.x || (c.x == b.x && __float_as_int(c.y) < __float_as_int(b.y))) b = c;
  }
  ids[r] = __float_as_int(b.y);
}

namespace hy4_idx {
constexpr int D = 128, RATIO = 4, HEADS = 16, ROPE = 64;
}  // namespace hy4_idx

// A row a block of 128: a group its row closes pooled (fp32 gated softmax over 4 slots), normed, roped, into gk.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_idx_pool(const __nv_bfloat16* tk, __nv_bfloat16* gk,
                                                                 const int* pos, const float* wv, const float* wg,
                                                                 const float* ape, const __nv_bfloat16* norm_w,
                                                                 const float* inv_freq, float eps) {
  using namespace hy4_idx;
  __shared__ float x[RATIO][D];
  __shared__ float red[4];
  __shared__ float pooled[D];
  const int64_t p = (int64_t)pos[0] + blockIdx.x;
  if ((p + 1) % RATIO != 0) return;
  const int64_t g = p / RATIO;
  const int d = threadIdx.x;
  for (int o = 0; o < RATIO; ++o) x[o][d] = __bfloat162float(tk[(g * RATIO + o) * D + d]);
  __syncthreads();
  float v[RATIO], gt[RATIO], mx = -INFINITY;
  for (int o = 0; o < RATIO; ++o) {
    float a = 0.0f, b = 0.0f;
    for (int j = 0; j < D; ++j) {
      a = __fmaf_rn(wv[d * D + j], x[o][j], a);
      b = __fmaf_rn(wg[d * D + j], x[o][j], b);
    }
    v[o] = a;
    gt[o] = b + ape[o * D + d];
    mx = fmaxf(mx, gt[o]);
  }
  float den = 0.0f, num = 0.0f;
  for (int o = 0; o < RATIO; ++o) {
    const float e = __expf(gt[o] - mx);
    den += e;
    num = __fmaf_rn(v[o], e, num);
  }
  const float y = num / den;
  float ss = y * y;
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / D + eps);
  pooled[d] = __bfloat162float(__float2bfloat16_rn(__bfloat162float(norm_w[d]) * (y * inv)));
  __syncthreads();
  float out = pooled[d];
  if (d >= D - ROPE) {
    const int i = (d - (D - ROPE)) >> 1;
    float sn, cs;
    sincosf(__fmul_rn(inv_freq[i], (float)(g * RATIO)), &sn, &cs);
    const float a = pooled[d & ~1], b = pooled[d | 1];
    out = (d & 1) ? __fmaf_rn(b, cs, a * sn) : __fmaf_rn(a, cs, -b * sn);
  }
  gk[g * D + d] = __float2bfloat16_rn(out);
}

// Block (row, 32 groups) of 256: sum_h w_h scale relu(q_h . k_g) before the row's group, +inf on it, -inf past it.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_idx_scores(const __nv_bfloat16* q, const float* w,
                                                                   const __nv_bfloat16* gk, const int* pos, float* out,
                                                                   int64_t np, float scale) {
  using namespace hy4_idx;
  __shared__ float qs[HEADS][D];
  const int64_t r = blockIdx.x, own = ((int64_t)pos[0] + r) / RATIO;
  for (int i = threadIdx.x; i < HEADS * D; i += blockDim.x) qs[i / D][i % D] = __bfloat162float(q[r * HEADS * D + i]);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int j = warp; j < 32; j += 8) {
    const int64_t g = (int64_t)blockIdx.y * 32 + j;
    if (g >= np) return;
    float sc;
    if (g == own) sc = INFINITY;
    else if (g > own) sc = -INFINITY;
    else {
      float k4[4];
      for (int c = 0; c < 4; ++c) k4[c] = __bfloat162float(gk[g * D + lane * 4 + c]);
      sc = 0.0f;
      for (int h = 0; h < HEADS; ++h) {
        float dot = 0.0f;
        for (int c = 0; c < 4; ++c) dot = __fmaf_rn(qs[h][lane * 4 + c], k4[c], dot);
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        sc = __fmaf_rn(w[r * HEADS + h] * scale, fmaxf(dot, 0.0f), sc);
      }
    }
    if (lane == 0) out[r * np + g] = sc;
  }
}

// A row a block of 512: the recalled groups' keys at or before the row's position as a token list, and its count.
extern "C" __global__ void __launch_bounds__(512) tf_hy4_idx_tokens(const int64_t* sel, const int* pos, int* tok,
                                                                   int* cnt) {
  using namespace hy4_idx;
  __shared__ int sums[16];
  const int64_t r = blockIdx.x, p = (int64_t)pos[0] + r;
  const int64_t g = sel[r * 512 + threadIdx.x];
  int mine = 0;
  for (int o = 0; o < RATIO; ++o) mine += g >= 0 && g * RATIO + o <= p;
  int incl = mine;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) sums[warp] = incl;
  __syncthreads();
  int before = 0;
  for (int i = 0; i < warp; ++i) before += sums[i];
  int at = before + incl - mine;
  for (int o = 0; o < RATIO; ++o)
    if (g >= 0 && g * RATIO + o <= p) tok[r * 2048 + at++] = (int)(g * RATIO + o);
  if (threadIdx.x == 511) cnt[r] = at;
}
