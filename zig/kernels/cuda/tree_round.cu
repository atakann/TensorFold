// Verified tree rounds for every CUDA family (host reference: core/tree_round.zig, the Metal kernels' contract).

// Deterministic BF16 picks, each tree's verified path and emitted tokens, and the commit steps that read the result.

#include <stdint.h>

#include "round_ids.cuh"
#include "round_tree.cuh"

using namespace tf_tree_round;

extern "C" __global__ void __launch_bounds__(256) tf_round_bf16_argmax(const uint16_t* __restrict__ logits,
                                                                       Pick* __restrict__ picks, uint32_t vocab,
                                                                       uint32_t stride) {
    argmax(logits, picks, vocab, stride, ALL);
}

extern "C" __global__ void tf_tree_round_match(const uint32_t* __restrict__ tokens, const int32_t* __restrict__ parents,
                                               const Pick* __restrict__ picks, const uint32_t* __restrict__ eos,
                                               const Tree* __restrict__ trees, Result* __restrict__ results,
                                               uint32_t vocab, uint32_t count) {
    match(tokens, parents, picks, eos, trees, results, vocab, count, ALL);
}

extern "C" __global__ void tf_round_keep_rows(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                              int32_t* __restrict__ keep, uint32_t* __restrict__ counts, uint32_t count) {
    keep_rows(results, trees, keep, counts, count, ALL);
}

// out[t, i, :] = rows[first_t + path_t[i], :] (width bf16 a row) for each tree's kept rows: taps, hidden rows.
extern "C" __global__ void tf_round_gather_rows(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                                const uint16_t* __restrict__ rows, uint16_t* __restrict__ out,
                                                uint32_t width, uint32_t count) {
    const uint32_t t = blockIdx.y, i = blockIdx.z, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count || c >= width) return;
    const Result& r = results[t];
    if (!valid(r, trees[t].rows) || i >= r.consumed_count) return;
    out[(static_cast<uint64_t>(t) * MAX_ROWS + i) * width + c] = rows[static_cast<uint64_t>(trees[t].first + r.path[i]) * width + c];
}

// head[t, :] = logits[first_t + the tree's last kept row, :]: the row the next round's first pick reads.
extern "C" __global__ void tf_round_keep_head(const Result* __restrict__ results, const Tree* __restrict__ trees,
                                              const uint16_t* __restrict__ logits, uint16_t* __restrict__ head,
                                              uint32_t width, uint32_t stride, uint32_t count) {
    const uint32_t t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count || c >= width) return;
    const Result& r = results[t];
    if (!valid(r, trees[t].rows)) return;
    head[static_cast<uint64_t>(t) * width + c] = logits[static_cast<uint64_t>(trees[t].first + r.path[r.consumed_count - 1]) * stride + c];
}

// picks [rows] as int32 token ids for the next forward (a draft chain's next step): no host round trip between steps.
extern "C" __global__ void __launch_bounds__(256) tf_round_pick_ids(const uint32_t* picks, int* ids, unsigned rows,
                                                                     unsigned vocab, int fallback) {
    tf_round_ids::pick_ids(picks, ids, rows, vocab, fallback, tf_round_ids::ALL);
}
