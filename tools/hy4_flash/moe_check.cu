// Dev-time check: GLM's sigmoid top-k router (tf_glm_topk) on Hy4-flash's router logits.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include "glm_glue_moe.cuh"

std::vector<torch::Tensor> topk(torch::Tensor logits, torch::Tensor bias, int64_t top_k, double scale) {
  const int64_t rows = logits.size(0), ne = logits.size(1);
  auto pick = torch::empty({rows, top_k + 1}, logits.options().dtype(torch::kInt32));
  auto wts = torch::empty({rows, top_k + 1}, logits.options());
  tf_glm_topk<<<rows, 32>>>(logits.data_ptr<float>(), bias.data_ptr<float>(), pick.data_ptr<int>(), wts.data_ptr<float>(),
                            (float)scale, ne, top_k, 1);
  return {pick, wts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("topk", &topk); }
