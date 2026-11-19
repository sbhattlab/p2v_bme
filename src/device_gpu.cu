/*
 * device_gpu.cu
 *
 * The GPU device: every CUDA kernel and CUDA call of the package. Alignment copies (and start-tree
 * subsamples), the window workspace, the window solve (condensed K x K distances, discrete gradient, batched
 * moves, the exact acceptance check), the nearest-sample search and the edge-length pass. Distances: the run's
 * model (set_distance; JC69 by default) from the 5-bitplane alignment, by the per-pair formulas of
 * substitution_models.hpp and eigen_dist.hpp. JC69 needs two counts per pair (shared valid sites, matches) and
 * has tiled kernels; the other models need the 16 pair counts and take the per-pair path of each kernel.
 */
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "eigen_dist.hpp"
#include "search.hpp"
#include "substitution_models.hpp"

// Every CUDA call is checked; a failure throws std::runtime_error (the library never exits).
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err_ = (call);                                                                                     \
        if (err_ != cudaSuccess)                                                                                       \
            cuda_fail(err_, #call, __LINE__);                                                                          \
    } while (0)

namespace p2vbme {

namespace {

// What to try when a GPU runs out of memory.
constexpr const char *OUT_OF_MEMORY_HINT = "use other or more GPUs (--devices) or the CPU (--accelerator cpu)";

std::string gpu_name(int dev) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return "unknown";
    return prop.name;
}

[[noreturn]] void cuda_fail(cudaError_t err, const char *call, int line) {
    cudaGetLastError(); // clear a non-sticky error, so it is not reported again by a later call
    int dev = -1;
    cudaGetDevice(&dev);
    throw std::runtime_error("CUDA error on GPU " + std::to_string(dev) + " (" + gpu_name(dev) + "): " +
                             cudaGetErrorString(err) + " [" + call + ", device_gpu.cu:" + std::to_string(line) + "]");
}

// cudaMalloc that reports running out of memory: the GPU, the size requested and the memory free.
void gpu_alloc_bytes(void **ptr, size_t bytes, int line) {
    const cudaError_t err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess)
        return;
    if (err != cudaErrorMemoryAllocation)
        cuda_fail(err, "cudaMalloc", line);
    cudaGetLastError(); // clear the (non-sticky) allocation error
    int dev = -1;
    size_t free_b = 0, total_b = 0;
    cudaGetDevice(&dev);
    cudaMemGetInfo(&free_b, &total_b);
    char msg[512];
    snprintf(msg, sizeof(msg), "out of GPU memory on GPU %d (%s): %.1f MiB requested, %.1f MiB free of %.1f MiB; %s",
             dev, gpu_name(dev).c_str(), bytes / 1048576.0, free_b / 1048576.0, total_b / 1048576.0,
             OUT_OF_MEMORY_HINT);
    throw std::runtime_error(msg);
}
#define gpu_alloc(ptr, bytes) gpu_alloc_bytes((void **)(ptr), (bytes), __LINE__)

// ---- 2. Pairwise distances from the bit-packed alignment ----

using DeviceDistance = DistanceParams<float>; // the run's model on the GPU (float)

__device__ __forceinline__ float dev_jc69_dist_popc(int32_t x, int32_t y, const uint64_t *__restrict__ d_packed_seqs,
                                                    int32_t W) {
    if (x == y)
        return 0.0f;
    const uint64_t *sx = &d_packed_seqs[(size_t)x * 5 * W];
    const uint64_t *sy = &d_packed_seqs[(size_t)y * 5 * W];
    int32_t n_valid = 0;
    int32_t n_match = 0;
#pragma unroll 5
    for (int32_t w = 0; w < W; ++w) {
        uint64_t vx = __ldg(&sx[4 * W + w]);
        uint64_t vy = __ldg(&sy[4 * W + w]);
        n_valid += __popcll(vx & vy);
        uint64_t m = (__ldg(&sx[0 * W + w]) & __ldg(&sy[0 * W + w])) | (__ldg(&sx[1 * W + w]) & __ldg(&sy[1 * W + w])) |
                     (__ldg(&sx[2 * W + w]) & __ldg(&sy[2 * W + w])) | (__ldg(&sx[3 * W + w]) & __ldg(&sy[3 * W + w]));
        n_match += __popcll(m);
    }
    return jc69_distance<float>(n_valid, n_match);
}

// The 16 pair counts of leaves x and y: c[4 i + j] = sites with base i in x and base j in y.
__device__ __forceinline__ void dev_pair_counts(int32_t x, int32_t y, const uint64_t *__restrict__ d_packed_seqs,
                                                int32_t W, uint32_t c[16]) {
    const uint64_t *sx = &d_packed_seqs[(size_t)x * 5 * W];
    const uint64_t *sy = &d_packed_seqs[(size_t)y * 5 * W];
#pragma unroll
    for (int k = 0; k < 16; ++k)
        c[k] = 0u;
#pragma unroll 5
    for (int32_t w = 0; w < W; ++w) {
        uint64_t a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = __ldg(&sx[i * W + w]);
            b[i] = __ldg(&sy[i * W + w]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                c[4 * i + j] += __popcll(a[i] & b[j]);
    }
}

// The model's distance between leaves x and y (any model; JC69 has the faster dev_jc69_dist_popc).
__device__ __forceinline__ float dev_model_dist(int32_t x, int32_t y, const uint64_t *__restrict__ d_packed_seqs,
                                                int32_t W, const DeviceDistance &dm) {
    if (x == y)
        return 0.0f;
    uint32_t c[16];
    dev_pair_counts(x, y, d_packed_seqs, W, c);
    return pair_distance<float>(dm, c);
}

// Build K x K condensed distance matrix on the fly from dyadic-weighted piece profiles:
//   D_cond[a, b] = sum_{p in prof(a)} sum_{q in prof(b)} w_p * w_q * D(leaf_p, leaf_q).
// Each piece's profile is stored in CSR format (d_prof_offsets, d_prof_leaves, d_prof_weights)
// with nonnegative weights w_p summing to 1 (exact dyadic weights w_p = 2^{-depth(leaf_p)} when
// leaf_cnt <= profile_cap or profile_cap <= 0; approximate frontier-subtree collapse above cap).
// JC69 (template JC) or the model dm.
template <bool JC>
__global__ void k_build_real_seq_condensed_D(int32_t K, int32_t W, const int32_t *__restrict__ d_prof_offsets,
                                             const int32_t *__restrict__ d_prof_leaves,
                                             const float *__restrict__ d_prof_weights,
                                             const uint64_t *__restrict__ d_packed_seqs, float *__restrict__ d_D_cond,
                                             int32_t heavy_min, DeviceDistance dm) {
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= K || b >= K)
        return;
    if (a == b) {
        d_D_cond[(size_t)a * K + b] = 0.0f;
        return;
    }
    if (a < b) {
        int32_t off_a = d_prof_offsets[a], end_a = d_prof_offsets[a + 1];
        int32_t off_b = d_prof_offsets[b], end_b = d_prof_offsets[b + 1];
        // Pairs where both profiles have >= heavy_min leaves are left to the tiled kernel.
        if (end_a - off_a >= heavy_min && end_b - off_b >= heavy_min)
            return;
        float d_val = 0.0f;
        for (int32_t p = off_a; p < end_a; ++p) {
            int32_t la = d_prof_leaves[p];
            float wa = d_prof_weights[p];
            float row_sum = 0.0f;
            for (int32_t q = off_b; q < end_b; ++q) {
                int32_t lb = d_prof_leaves[q];
                float wb = d_prof_weights[q];
                if constexpr (JC)
                    row_sum += wb * dev_jc69_dist_popc(la, lb, d_packed_seqs, W);
                else
                    row_sum += wb * dev_model_dist(la, lb, d_packed_seqs, W, dm);
            }
            d_val += wa * row_sum;
        }
        d_D_cond[(size_t)a * K + b] = d_val;
        d_D_cond[(size_t)b * K + a] = d_val;
    }
}

// Same float formula as dev_jc69_dist_popc, from precomputed popcounts.
__device__ __forceinline__ float dev_jc69_from_counts(int32_t n_valid, int32_t n_match) {
    return jc69_distance<float>(n_valid, n_match);
}

constexpr int32_t HB_T = 64;        // leaf tile edge
constexpr int32_t HB_CW = 5;        // words per staged chunk (25 words = 5 chunks)
constexpr int32_t HB_HEAVY_MIN = 8; // both profiles at least this large -> tiled kernel

// Tiled k_build_real_seq_condensed_D for heavy pairs (one block per pair a < b).
// Leaf sequences are staged in shared memory in word chunks and each thread keeps a 4x4 tile of exact popcounts.
// Distances use the same float formula, and the weighted sums run in the same order as k_build_real_seq_condensed_D
// (row sums over q, then over p), so D_cond is bit-identical for any W > 0.
__global__ __launch_bounds__(256) void k_build_condensed_D_heavy_tiled(
    int32_t K, int32_t W, int32_t n_pairs, const int32_t *__restrict__ d_pair_a, const int32_t *__restrict__ d_pair_b,
    const int32_t *__restrict__ d_prof_offsets, const int32_t *__restrict__ d_prof_leaves,
    const float *__restrict__ d_prof_weights, const uint64_t *__restrict__ d_packed_seqs,
    float *__restrict__ d_D_cond) {
    __shared__ uint64_t sA[5 * HB_CW * HB_T]; // [channel][word][row]
    __shared__ uint64_t sB[5 * HB_CW * HB_T];
    __shared__ float s_dist[HB_T][HB_T + 1];
    __shared__ float s_row_sum[HB_T];
    __shared__ int32_t s_leaf_a[HB_T], s_leaf_b[HB_T];
    __shared__ float s_wa[HB_T], s_wb[HB_T];

    const int32_t pid = blockIdx.x;
    if (pid >= n_pairs)
        return;
    const int32_t a = d_pair_a[pid], b = d_pair_b[pid];
    const int32_t off_a = d_prof_offsets[a], len_a = d_prof_offsets[a + 1] - off_a;
    const int32_t off_b = d_prof_offsets[b], len_b = d_prof_offsets[b + 1] - off_b;
    const int32_t tid = (int32_t)threadIdx.x, tx = tid & 15, ty = tid >> 4;
    float d_val = 0.0f;

    for (int32_t pa0 = 0; pa0 < len_a; pa0 += HB_T) {
        const int32_t na = min(HB_T, len_a - pa0);
        __syncthreads();
        if (tid < HB_T) {
            s_row_sum[tid] = 0.0f;
            if (tid < na) {
                s_leaf_a[tid] = d_prof_leaves[off_a + pa0 + tid];
                s_wa[tid] = d_prof_weights[off_a + pa0 + tid];
            }
        }
        for (int32_t qb0 = 0; qb0 < len_b; qb0 += HB_T) {
            const int32_t nb = min(HB_T, len_b - qb0);
            __syncthreads();
            if (tid < nb) {
                s_leaf_b[tid] = d_prof_leaves[off_b + qb0 + tid];
                s_wb[tid] = d_prof_weights[off_b + qb0 + tid];
            }
            __syncthreads();

            int32_t nv[4][4], nm[4][4];
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    nv[i][j] = 0;
                    nm[i][j] = 0;
                }

            for (int32_t w0 = 0; w0 < W; w0 += HB_CW) {
                // Stage words w0..w0+HB_CW-1 of all 5 channels; consecutive threads read consecutive words of one
                // sequence.
                for (int32_t e = tid; e < 5 * HB_CW * HB_T; e += 256) {
                    const int32_t cw = e % HB_CW, ch = (e / HB_CW) % 5, r = e / (5 * HB_CW);
                    const int32_t s_idx = (ch * HB_CW + cw) * HB_T + r;
                    const int32_t w = w0 + cw;
                    const size_t g_off = (size_t)ch * W + w;
                    sA[s_idx] = (r < na && w < W) ? d_packed_seqs[(size_t)s_leaf_a[r] * 5 * W + g_off] : 0ull;
                    sB[s_idx] = (r < nb && w < W) ? d_packed_seqs[(size_t)s_leaf_b[r] * 5 * W + g_off] : 0ull;
                }
                __syncthreads();
#pragma unroll
                for (int32_t cw = 0; cw < HB_CW; ++cw) {
                    uint64_t ra[4], rb[4], m[4][4];
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i)
                        ra[i] = sA[(4 * HB_CW + cw) * HB_T + ty + 16 * i];
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j)
                        rb[j] = sB[(4 * HB_CW + cw) * HB_T + tx + 16 * j];
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                            nv[i][j] += __popcll(ra[i] & rb[j]);
#pragma unroll
                    for (int32_t c = 0; c < 4; ++c) {
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
                            ra[i] = sA[(c * HB_CW + cw) * HB_T + ty + 16 * i];
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                            rb[j] = sB[(c * HB_CW + cw) * HB_T + tx + 16 * j];
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                            for (int32_t j = 0; j < 4; ++j)
                                m[i][j] = (c == 0) ? (ra[i] & rb[j]) : (m[i][j] | (ra[i] & rb[j]));
                    }
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                            nm[i][j] += __popcll(m[i][j]);
                }
                __syncthreads();
            }

#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    const int32_t p = ty + 16 * i, q = tx + 16 * j;
                    if (p < na && q < nb) {
                        s_dist[p][q] = (s_leaf_a[p] == s_leaf_b[q]) ? 0.0f : dev_jc69_from_counts(nv[i][j], nm[i][j]);
                    }
                }
            }
            __syncthreads();
            if (tid < na) {
                float row_sum = s_row_sum[tid];
                for (int32_t q = 0; q < nb; ++q)
                    row_sum += s_wb[q] * s_dist[tid][q];
                s_row_sum[tid] = row_sum;
            }
        }
        __syncthreads();
        if (tid == 0) {
            for (int32_t p = 0; p < na; ++p)
                d_val += s_wa[p] * s_row_sum[p];
        }
    }
    if (tid == 0) {
        d_D_cond[(size_t)a * K + b] = d_val;
        d_D_cond[(size_t)b * K + a] = d_val;
    }
}

// Block-parallel exact refinement of selected piece pairs (a, b) using uncapped dyadic profiles (d_exact_offsets,
// d_exact_leaves, d_exact_weights) and on-the-fly JC69 popcount distances.
// Uses shared-memory staged, k-sliced, 4x4 register-tiled evaluation with adaptive A-tile heights (16, 32, 48, 64) and
// multi-pair packing across consecutive pairs sharing the same right-hand profile b, updating d_D_cond[a * K + b],
// d_D_cond[b * K + a], and d_pair_exact_D[pid].
template <bool JC>
__global__ __launch_bounds__(256, 2) void k_refine_real_seq_condensed_D_pairs(
    int32_t K, int32_t W, int32_t n_pairs, const int32_t *__restrict__ d_pair_a, const int32_t *__restrict__ d_pair_b,
    const int32_t *__restrict__ d_exact_offsets, const int32_t *__restrict__ d_exact_leaves,
    const float *__restrict__ d_exact_weights, const uint64_t *__restrict__ d_packed_seqs, float *__restrict__ d_D_cond,
    double *__restrict__ d_pair_exact_D, DeviceDistance dm) {
    const int32_t tid = (int32_t)threadIdx.x;
    const int32_t pid = (int32_t)blockIdx.x;
    if (pid >= n_pairs)
        return;

    constexpr int32_t BM = 64, BN = 64, TY = 16, TX = 16, RJ = 4;
    constexpr int32_t SW = 25, PA = BM + 1, PB = BN + 1;
    constexpr int32_t MAX_CAP_A = 128;
    const int32_t nsl = W / 5;
    const int32_t stride_x = 5 * W;
    const int32_t c_step = W - 5;

    __shared__ uint64_t sA[SW * PA];
    __shared__ uint64_t sB[SW * PB];
    __shared__ int32_t s_win_a[64];
    __shared__ int32_t s_win_b[64];
    __shared__ int32_t s_win_oa[64];
    __shared__ int32_t s_win_la[64];
    __shared__ int32_t s_win_ob[64];
    __shared__ int32_t s_win_lb[64];
    __shared__ int32_t s_pack_leaf_a[MAX_CAP_A];
    __shared__ double s_pack_wa[MAX_CAP_A];
    __shared__ uint8_t s_pack_pidx[MAX_CAP_A];
    __shared__ double s_pair_sum[64];
    __shared__ int32_t s_leaf_a[BM];
    __shared__ double s_wa[BM];
    __shared__ int32_t s_leaf_b[BN];
    __shared__ double s_wb[BN];
    __shared__ double s_row_sum[BM];
    __shared__ int32_t s_meta[6];

    // Per-pair path: models other than JC69, and word counts the tiled path does not handle
    if (!JC || W <= 0 || (W % 5) != 0) {
        int32_t a = d_pair_a[pid];
        int32_t b = d_pair_b[pid];
        int32_t off_a = d_exact_offsets[a], end_a = d_exact_offsets[a + 1];
        int32_t off_b = d_exact_offsets[b], end_b = d_exact_offsets[b + 1];
        int32_t len_b = end_b - off_b;
        int64_t total_ij = (int64_t)(end_a - off_a) * (int64_t)len_b;
        double my_sum = 0.0;
        for (int64_t idx = tid; idx < total_ij; idx += 256) {
            int32_t p = off_a + (int32_t)(idx / len_b);
            int32_t q = off_b + (int32_t)(idx % len_b);
            float wp = d_exact_weights[p], wq = d_exact_weights[q];
            if (wp == 0.0f || wq == 0.0f)
                continue;
            int32_t lx = d_exact_leaves[p], ly = d_exact_leaves[q];
            float d_jc = 0.0f;
            if constexpr (!JC) {
                if (lx != ly)
                    d_jc = dev_model_dist(lx, ly, d_packed_seqs, W, dm);
            } else if (lx != ly) {
                const uint64_t *sx = d_packed_seqs + (size_t)lx * (5 * W);
                const uint64_t *sy = d_packed_seqs + (size_t)ly * (5 * W);
                int32_t n_valid = 0, n_match = 0;
#pragma unroll 1
                for (int32_t w = 0; w < W; ++w) {
                    n_valid += __popcll(sx[4 * W + w] & sy[4 * W + w]);
                    n_match += __popcll((sx[0 * W + w] & sy[0 * W + w]) | (sx[1 * W + w] & sy[1 * W + w]) |
                                        (sx[2 * W + w] & sy[2 * W + w]) | (sx[3 * W + w] & sy[3 * W + w]));
                }
                d_jc = jc69_distance<float>(n_valid, n_match);
            }
            my_sum += ((double)wp * (double)wq) * (double)d_jc;
        }
        double *s_red = reinterpret_cast<double *>(sA);
        s_red[tid] = my_sum;
        __syncthreads();
        for (int32_t s = 128; s > 0; s >>= 1) {
            if (tid < s)
                s_red[tid] += s_red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            if (d_pair_exact_D != NULL)
                d_pair_exact_D[pid] = s_red[0];
            if (d_D_cond != NULL && a < K && b < K) {
                float fv = (float)s_red[0];
                d_D_cond[(size_t)a * K + b] = fv;
                d_D_cond[(size_t)b * K + a] = fv;
            }
        }
        return;
    }

    const int32_t ws = pid & ~63;
    const int32_t we = (ws + 64 < n_pairs) ? (ws + 64) : n_pairs;
    const int32_t wlen = we - ws;
    if (tid < wlen) {
        int32_t ak = d_pair_a[ws + tid];
        int32_t bk = d_pair_b[ws + tid];
        int32_t oa = d_exact_offsets[ak];
        int32_t ea = d_exact_offsets[ak + 1];
        int32_t ob = d_exact_offsets[bk];
        int32_t eb = d_exact_offsets[bk + 1];
        s_win_a[tid] = ak;
        s_win_b[tid] = bk;
        s_win_oa[tid] = oa;
        s_win_la[tid] = ea - oa;
        s_win_ob[tid] = ob;
        s_win_lb[tid] = eb - ob;
    }
    __syncthreads();

    const int32_t my_idx = pid - ws;
    int32_t lead_idx = my_idx, pack_m = 1, total_a = 0;
    int32_t cur = 0;
    while (cur < wlen) {
        int32_t b_cur = s_win_b[cur];
        int32_t lb_cur = s_win_lb[cur];
        int32_t la_cur = s_win_la[cur];
        int32_t cap_a = (n_pairs >= 4096) ? 64 : ((n_pairs >= 2048) ? 32 : 16);
        if (lb_cur <= 512) {
            cap_a = (n_pairs >= 2048) ? MAX_CAP_A : 64;
        }
        int32_t m = 1;
        int32_t sum_a = la_cur;
        if (la_cur > 0 && la_cur <= lb_cur && la_cur < cap_a) {
            while (cur + m < wlen) {
                int32_t b_nxt = s_win_b[cur + m];
                int32_t la_nxt = s_win_la[cur + m];
                if (b_nxt != b_cur || la_nxt <= 0 || la_nxt > lb_cur || sum_a + la_nxt > cap_a)
                    break;
                sum_a += la_nxt;
                ++m;
            }
        }
        if (my_idx >= cur && my_idx < cur + m) {
            lead_idx = cur;
            pack_m = m;
            total_a = sum_a;
            break;
        }
        cur += m;
    }
    if (my_idx != lead_idx)
        return;

    int32_t off_a0 = s_win_oa[lead_idx];
    int32_t len_a0 = s_win_la[lead_idx];
    int32_t off_b0 = s_win_ob[lead_idx];
    int32_t len_b0 = s_win_lb[lead_idx];
    if (pack_m == 1) {
        if (len_a0 > len_b0) {
            int32_t t;
            t = off_a0;
            off_a0 = off_b0;
            off_b0 = t;
            t = len_a0;
            len_a0 = len_b0;
            len_b0 = t;
        }
        total_a = len_a0;
    }
    if (tid < pack_m) {
        s_pair_sum[tid] = 0.0;
    }
    if (total_a <= 0 || len_b0 <= 0) {
        if (tid < pack_m) {
            int32_t gpid = ws + lead_idx + tid;
            int32_t ak = s_win_a[lead_idx + tid];
            int32_t bk = s_win_b[lead_idx + tid];
            if (d_pair_exact_D != NULL)
                d_pair_exact_D[gpid] = 0.0;
            if (d_D_cond != NULL && ak < K && bk < K) {
                d_D_cond[(size_t)ak * K + bk] = 0.0f;
                d_D_cond[(size_t)bk * K + ak] = 0.0f;
            }
        }
        return;
    }

    if (tid == 0) {
        s_meta[0] = lead_idx;
        s_meta[1] = pack_m;
        s_meta[2] = total_a;
        s_meta[3] = off_a0;
        s_meta[4] = off_b0;
        s_meta[5] = len_b0;
    }
    if (pack_m > 1 && tid < total_a) {
        int32_t rem = tid;
        int32_t p = 0;
        while (p + 1 < pack_m) {
            int32_t la_p = s_win_la[lead_idx + p];
            if (rem < la_p)
                break;
            rem -= la_p;
            ++p;
        }
        int32_t src_idx = s_win_oa[lead_idx + p] + rem;
        float w = d_exact_weights[src_idx];
        s_pack_leaf_a[tid] = (w != 0.0f) ? d_exact_leaves[src_idx] : -1;
        s_pack_wa[tid] = (double)w;
        s_pack_pidx[tid] = (uint8_t)p;
    }
    __syncthreads();

    const int32_t tx = tid & 15;
    const int32_t ty = tid >> 4;

#define NK_WL_STEP(RI, ASTRIDE, ABASE, wl)                                                                             \
    {                                                                                                                  \
        uint64_t a_reg[RI], b_reg[RJ];                                                                                 \
        const int32_t kv = 20 + (wl);                                                                                  \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) a_reg[i] = sA[((ABASE) + kv) * (ASTRIDE) + ty + TY * i];  \
        _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) b_reg[j] = sB[kv * PB + tx + TX * j];                       \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j)        \
            nv[i][j] += __popcll(a_reg[i] & b_reg[j]);                                                                 \
        uint64_t m_reg[RI][RJ];                                                                                        \
        _Pragma("unroll") for (int32_t c = 0; c < 4; ++c) {                                                            \
            const int32_t kc = c * 5 + (wl);                                                                           \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) a_reg[i] =                                            \
                sA[((ABASE) + kc) * (ASTRIDE) + ty + TY * i];                                                          \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) b_reg[j] = sB[kc * PB + tx + TX * j];                   \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j)    \
                m_reg[i][j] = (c == 0) ? (a_reg[i] & b_reg[j]) : (m_reg[i][j] | (a_reg[i] & b_reg[j]));                \
        }                                                                                                              \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j)        \
            nm[i][j] += __popcll(m_reg[i][j]);                                                                         \
    }

#define NK_RUN_ATILE_16()                                                                                              \
    {                                                                                                                  \
        double acc0 = 0.0;                                                                                             \
        _Pragma("unroll 1") for (int32_t q0 = 0; q0 < s_meta[5]; q0 += BN) {                                           \
            const int32_t tile_b = (s_meta[5] - q0 < BN) ? (s_meta[5] - q0) : BN;                                      \
            __syncthreads();                                                                                           \
            if (tid < BN) {                                                                                            \
                if (tid < tile_b) {                                                                                    \
                    float w = d_exact_weights[s_meta[4] + q0 + tid];                                                   \
                    s_leaf_b[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[4] + q0 + tid] : -1;                           \
                    s_wb[tid] = (double)w;                                                                             \
                } else {                                                                                               \
                    s_leaf_b[tid] = -1;                                                                                \
                    s_wb[tid] = 0.0;                                                                                   \
                }                                                                                                      \
            }                                                                                                          \
            uint32_t nv[1][RJ], nm[1][RJ];                                                                             \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) {                                                       \
                nv[0][j] = 0u;                                                                                         \
                nm[0][j] = 0u;                                                                                         \
            }                                                                                                          \
            _Pragma("unroll 1") for (int32_t s = 0; s < nsl; ++s) {                                                    \
                __syncthreads();                                                                                       \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < 16 * SW; idx += 256) {                               \
                    int32_t r = idx / SW;                                                                              \
                    int32_t k = idx - r * SW;                                                                          \
                    int32_t c = k / 5;                                                                                 \
                    int32_t la = s_leaf_a[r];                                                                          \
                    sA[k * PA + r] =                                                                                   \
                        (la >= 0) ? __ldg(&d_packed_seqs[(size_t)la * stride_x + s * 5 + c * c_step + k]) : 0ULL;      \
                }                                                                                                      \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < BN * SW; idx += 256) {                               \
                    int32_t r = idx / SW;                                                                              \
                    int32_t k = idx - r * SW;                                                                          \
                    int32_t c = k / 5;                                                                                 \
                    int32_t lb = s_leaf_b[r];                                                                          \
                    sB[k * PB + r] =                                                                                   \
                        (lb >= 0) ? __ldg(&d_packed_seqs[(size_t)lb * stride_x + s * 5 + c * c_step + k]) : 0ULL;      \
                }                                                                                                      \
                __syncthreads();                                                                                       \
                NK_WL_STEP(1, PA, 0, 0)                                                                                \
                NK_WL_STEP(1, PA, 0, 1)                                                                                \
                NK_WL_STEP(1, PA, 0, 2)                                                                                \
                NK_WL_STEP(1, PA, 0, 3)                                                                                \
                NK_WL_STEP(1, PA, 0, 4)                                                                                \
            }                                                                                                          \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) {                                                       \
                const double wq = s_wb[tx + TX * j];                                                                   \
                const float d_jc = jc69_distance<float>((int32_t)nv[0][j], (int32_t)nm[0][j]);                         \
                acc0 += wq * (double)d_jc;                                                                             \
            }                                                                                                          \
        }                                                                                                              \
        _Pragma("unroll") for (int32_t offset = 8; offset > 0; offset >>= 1) {                                         \
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);                                                       \
        }                                                                                                              \
        if (tx == 0 && ty < tile_a) {                                                                                  \
            s_row_sum[ty] = s_wa[ty] * acc0;                                                                           \
        }                                                                                                              \
    }

#define NK_RUN_ATILE_64(RI)                                                                                            \
    {                                                                                                                  \
        double acc[RI];                                                                                                \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) acc[i] = 0.0;                                             \
        _Pragma("unroll 1") for (int32_t q0 = 0; q0 < s_meta[5]; q0 += BN) {                                           \
            const int32_t tile_b = (s_meta[5] - q0 < BN) ? (s_meta[5] - q0) : BN;                                      \
            __syncthreads();                                                                                           \
            if (tid < BN) {                                                                                            \
                if (tid < tile_b) {                                                                                    \
                    float w = d_exact_weights[s_meta[4] + q0 + tid];                                                   \
                    s_leaf_b[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[4] + q0 + tid] : -1;                           \
                    s_wb[tid] = (double)w;                                                                             \
                } else {                                                                                               \
                    s_leaf_b[tid] = -1;                                                                                \
                    s_wb[tid] = 0.0;                                                                                   \
                }                                                                                                      \
            }                                                                                                          \
            uint32_t nv[RI][RJ], nm[RI][RJ];                                                                           \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) {  \
                nv[i][j] = 0u;                                                                                         \
                nm[i][j] = 0u;                                                                                         \
            }                                                                                                          \
            _Pragma("unroll 1") for (int32_t s = 0; s < nsl; ++s) {                                                    \
                __syncthreads();                                                                                       \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < (RI) * 16 * SW; idx += 256) {                        \
                    int32_t r = idx / SW;                                                                              \
                    int32_t k = idx - r * SW;                                                                          \
                    int32_t c = k / 5;                                                                                 \
                    int32_t la = s_leaf_a[r];                                                                          \
                    sA[k * PA + r] =                                                                                   \
                        (la >= 0) ? __ldg(&d_packed_seqs[(size_t)la * stride_x + s * 5 + c * c_step + k]) : 0ULL;      \
                }                                                                                                      \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < BN * SW; idx += 256) {                               \
                    int32_t r = idx / SW;                                                                              \
                    int32_t k = idx - r * SW;                                                                          \
                    int32_t c = k / 5;                                                                                 \
                    int32_t lb = s_leaf_b[r];                                                                          \
                    sB[k * PB + r] =                                                                                   \
                        (lb >= 0) ? __ldg(&d_packed_seqs[(size_t)lb * stride_x + s * 5 + c * c_step + k]) : 0ULL;      \
                }                                                                                                      \
                __syncthreads();                                                                                       \
                NK_WL_STEP(RI, PA, 0, 0)                                                                               \
                NK_WL_STEP(RI, PA, 0, 1)                                                                               \
                NK_WL_STEP(RI, PA, 0, 2)                                                                               \
                NK_WL_STEP(RI, PA, 0, 3)                                                                               \
                NK_WL_STEP(RI, PA, 0, 4)                                                                               \
            }                                                                                                          \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) {                                                       \
                const double wq = s_wb[tx + TX * j];                                                                   \
                _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) {                                                 \
                    const float d_jc = jc69_distance<float>((int32_t)nv[i][j], (int32_t)nm[i][j]);                     \
                    acc[i] += wq * (double)d_jc;                                                                       \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) {                                                         \
            double v = acc[i];                                                                                         \
            _Pragma("unroll") for (int32_t offset = 8; offset > 0; offset >>= 1) {                                     \
                v += __shfl_down_sync(0xffffffffu, v, offset);                                                         \
            }                                                                                                          \
            int32_t r = ty + TY * i;                                                                                   \
            if (tx == 0 && r < tile_a) {                                                                               \
                s_row_sum[r] = s_wa[r] * v;                                                                            \
            }                                                                                                          \
        }                                                                                                              \
    }

#pragma unroll 1
    for (int32_t p0 = 0; p0 < s_meta[2]; p0 += BM) {
        const int32_t tile_a = (s_meta[2] - p0 < BM) ? (s_meta[2] - p0) : BM;
        if (tid < BM) {
            if (tid < tile_a) {
                if (s_meta[1] == 1) {
                    float w = d_exact_weights[s_meta[3] + p0 + tid];
                    s_leaf_a[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[3] + p0 + tid] : -1;
                    s_wa[tid] = (double)w;
                } else {
                    s_leaf_a[tid] = s_pack_leaf_a[p0 + tid];
                    s_wa[tid] = s_pack_wa[p0 + tid];
                }
            } else {
                s_leaf_a[tid] = -1;
                s_wa[tid] = 0.0;
            }
        }
        __syncthreads();

        if (tile_a <= 16) {
            NK_RUN_ATILE_16()
        } else if (tile_a <= 32) {
            NK_RUN_ATILE_64(2)
        } else if (tile_a <= 48) {
            NK_RUN_ATILE_64(3)
        } else {
            NK_RUN_ATILE_64(4)
        }
        __syncthreads();

        if (tid < s_meta[1]) {
            double s = s_pair_sum[tid];
            for (int32_t r = 0; r < tile_a; ++r) {
                if (s_meta[1] == 1 || s_pack_pidx[p0 + r] == (uint8_t)tid) {
                    s += s_row_sum[r];
                }
            }
            s_pair_sum[tid] = s;
        }
        __syncthreads();
    }

#undef NK_RUN_ATILE_64
#undef NK_RUN_ATILE_16
#undef NK_WL_STEP

    if (tid < s_meta[1]) {
        int32_t gpid = (pid & ~63) + s_meta[0] + tid;
        int32_t ak = s_win_a[s_meta[0] + tid];
        int32_t bk = s_win_b[s_meta[0] + tid];
        double exact_val = s_pair_sum[tid];
        if (d_pair_exact_D != NULL)
            d_pair_exact_D[gpid] = exact_val;
        if (d_D_cond != NULL && ak < K && bk < K) {
            float fv = (float)exact_val;
            d_D_cond[(size_t)ak * K + bk] = fv;
            d_D_cond[(size_t)bk * K + ak] = fv;
        }
    }
}

// ---- 3. Exact GPU Discrete Gradient Kernels ----

__global__ void k_copy_leaf_rows_perm(const float *__restrict__ D_orig, const int32_t *__restrict__ d_perm,
                                      float *__restrict__ subdist, int32_t N, int32_t M) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        int32_t r_orig = d_perm[row];
        int32_t c_orig = d_perm[col];
        subdist[(size_t)row * M + col] = D_orig[(size_t)r_orig * N + c_orig];
    }
}

__global__ void k_bottom_up_leaves(float *__restrict__ subdist, const int32_t *__restrict__ c1_r,
                                   const int32_t *__restrict__ c2_r, int32_t N, int32_t M, int32_t root_u) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N)
        return;
    for (int32_t u = N; u < root_u; ++u) {
        int32_t c1 = c1_r[u], c2 = c2_r[u];
        subdist[(size_t)u * M + v] = 0.5f * (subdist[(size_t)c1 * M + v] + subdist[(size_t)c2 * M + v]);
    }
}

__global__ void k_transpose_leaf_block(float *__restrict__ subdist, int32_t N, int32_t M, int32_t root_u) {
    __shared__ float tile[16][17];
    int32_t u = N + blockIdx.y * 16 + threadIdx.y;
    int32_t v = blockIdx.x * 16 + threadIdx.x;
    if (u < root_u && v < N)
        tile[threadIdx.y][threadIdx.x] = subdist[(size_t)u * M + v];
    __syncthreads();
    int32_t v_t = blockIdx.x * 16 + threadIdx.y;
    int32_t u_t = N + blockIdx.y * 16 + threadIdx.x;
    if (v_t < N && u_t < root_u)
        subdist[(size_t)v_t * M + u_t] = tile[threadIdx.x][threadIdx.y];
}

__global__ void k_bottom_up_internals(float *__restrict__ subdist, const int32_t *__restrict__ c1_r,
                                      const int32_t *__restrict__ c2_r, int32_t N, int32_t M, int32_t root_u) {
    int32_t v = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= root_u)
        return;
    for (int32_t u = N; u < root_u; ++u) {
        int32_t c1 = c1_r[u], c2 = c2_r[u];
        subdist[(size_t)u * M + v] = 0.5f * (subdist[(size_t)c1 * M + v] + subdist[(size_t)c2 * M + v]);
    }
}

__global__ void k_top_down_and_root(float *__restrict__ subdist, const int32_t *__restrict__ c1_r,
                                    const int32_t *__restrict__ c2_r, const int32_t *__restrict__ tin,
                                    const int32_t *__restrict__ sub_size, int32_t N, int32_t M, int32_t root_u,
                                    int32_t rc0, int32_t rc1, int32_t rc2) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= M)
        return;
    if (v == root_u) {
        subdist[(size_t)root_u * M + root_u] = 0.0f;
        return;
    }
    int32_t tv = tin[v];
    int32_t curr;
    float root_val;
    if (tv < tin[rc1]) {
        curr = rc0;
        root_val = 0.5f * (subdist[(size_t)rc1 * M + v] + subdist[(size_t)rc2 * M + v]);
    } else if (tv < tin[rc2]) {
        curr = rc1;
        root_val = 0.5f * (subdist[(size_t)rc0 * M + v] + subdist[(size_t)rc2 * M + v]);
    } else {
        curr = rc2;
        root_val = 0.5f * (subdist[(size_t)rc0 * M + v] + subdist[(size_t)rc1 * M + v]);
    }
    subdist[(size_t)root_u * M + v] = root_val;
    subdist[(size_t)v * M + root_u] = root_val;

    int32_t g = root_u;
    while (curr != v) {
        int32_t c1 = c1_r[curr], c2 = c2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib = (tv < tin[c2]) ? c2 : c1;
        float val = 0.5f * (subdist[(size_t)sib * M + v] + subdist[(size_t)g * M + v]);
        subdist[(size_t)curr * M + v] = val;
        subdist[(size_t)v * M + curr] = val;
        g = curr;
        curr = next_curr;
    }
    subdist[(size_t)v * M + v] = 0.0f;
}

__global__ void k_populate_nd_u(const float *__restrict__ subdist, const int32_t *__restrict__ adj_u,
                                float *__restrict__ nd_u, int32_t N, int32_t M, int32_t root_u) {
    int32_t u = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (u > root_u)
        return;
    int32_t n0 = adj_u[3 * u + 0], n1 = adj_u[3 * u + 1], n2 = adj_u[3 * u + 2];
    nd_u[3 * u + 0] = subdist[(size_t)n1 * M + n2];
    nd_u[3 * u + 1] = subdist[(size_t)n0 * M + n2];
    nd_u[3 * u + 2] = subdist[(size_t)n0 * M + n1];
}

__global__ void k_compute_bme_length(const float *__restrict__ subdist, const int32_t *__restrict__ parent_u, int32_t N,
                                     int32_t M, double *__restrict__ d_out_L) {
    __shared__ double s_sum[256];
    double acc = 0.0;
    for (int32_t x = threadIdx.x; x < N; x += blockDim.x) {
        acc += (double)subdist[(size_t)x * M + parent_u[x]];
    }
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *d_out_L = 0.5 * s_sum[0];
}

struct SprSharedQueue {
    int32_t u[256];
    int32_t c[256];
    float pow_val[256];
    float dkc[256];
    float dist[256];
    int32_t bounds[4];
};

__device__ __forceinline__ void
dev_expand_subtree_stack(int32_t u, int32_t C, float pow_neg_len, float d_kc, float curr_dist, int32_t n_leaves,
                         int32_t n_nodes_u, const int32_t *__restrict__ adj_u, const int32_t *__restrict__ parent_u,
                         const int32_t *__restrict__ sub_size, const float *__restrict__ nd_u,
                         const float *__restrict__ row_root, const float *__restrict__ row_orig,
                         float *__restrict__ edge_delta) {
    int32_t st_u[18], st_c[18];
    float st_pow[18], st_dkc[18], st_dist[18];
    int32_t top = 0;

    for (;;) {
        int32_t a0 = adj_u[3 * u + 0], a1 = adj_u[3 * u + 1];
        int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;

        int32_t b0 = adj_u[3 * u + idx0], b1 = adj_u[3 * u + idx1];
        float d_ab = nd_u[3 * u + ki];
        float r_b0 = row_root[b0], r_b1 = row_root[b1];

        float d_ac0 = nd_u[3 * u + idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
        edge_delta[(parent_u[u] == b0) ? u : b0] = ndist0;

        float d_ac1 = nd_u[3 * u + idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
        edge_delta[(parent_u[u] == b1) ? u : b1] = ndist1;

        float next_pow = pow_neg_len * 0.5f;
        if (b0 >= n_leaves && b1 >= n_leaves) {
            int32_t s0 = (parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u]);
            int32_t s1 = (parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u]);
            if (s0 >= s1) {
                st_u[top] = b0;
                st_c[top] = u;
                st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b1);
                st_dist[top] = ndist0;
                top++;
                C = u;
                u = b1;
                pow_neg_len = next_pow;
                d_kc = 0.5f * (d_kc + r_b0);
                curr_dist = ndist1;
            } else {
                st_u[top] = b1;
                st_c[top] = u;
                st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b0);
                st_dist[top] = ndist1;
                top++;
                C = u;
                u = b0;
                pow_neg_len = next_pow;
                d_kc = 0.5f * (d_kc + r_b1);
                curr_dist = ndist0;
            }
        } else if (b0 >= n_leaves) {
            C = u;
            u = b0;
            pow_neg_len = next_pow;
            d_kc = 0.5f * (d_kc + r_b1);
            curr_dist = ndist0;
        } else if (b1 >= n_leaves) {
            C = u;
            u = b1;
            pow_neg_len = next_pow;
            d_kc = 0.5f * (d_kc + r_b0);
            curr_dist = ndist1;
        } else {
            if (top == 0)
                break;
            --top;
            u = st_u[top];
            C = st_c[top];
            pow_neg_len = st_pow[top];
            d_kc = st_dkc[top];
            curr_dist = st_dist[top];
        }
    }
}

__device__ __forceinline__ void
dev_eval_cherry_or_enqueue(int32_t b, int32_t u, int32_t sb, float pow_neg_len, float d_kc, float curr_dist,
                           const int32_t *__restrict__ adj_u, const int32_t *__restrict__ parent_u,
                           const float *__restrict__ nd_u, const float *__restrict__ row_root,
                           const float *__restrict__ row_orig, float *__restrict__ edge_delta,
                           SprSharedQueue *__restrict__ sq, int32_t *__restrict__ q_tail_ptr) {
    if (sb == 3) {
        int32_t a0 = adj_u[3 * b + 0], a1 = adj_u[3 * b + 1];
        int32_t ki = (a0 == u) ? 0 : ((a1 == u) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;
        int32_t c0 = adj_u[3 * b + idx0], c1 = adj_u[3 * b + idx1];
        float d_ab = nd_u[3 * b + ki];
        float r_c0 = row_root[c0], r_c1 = row_root[c1];
        float d_ac0 = nd_u[3 * b + idx0] + (row_orig[c1] - r_c1) * pow_neg_len;
        edge_delta[(parent_u[b] == c0) ? b : c0] = curr_dist + 0.25f * (d_ac0 + r_c0 - d_ab - d_kc);
        float d_ac1 = nd_u[3 * b + idx1] + (row_orig[c0] - r_c0) * pow_neg_len;
        edge_delta[(parent_u[b] == c1) ? b : c1] = curr_dist + 0.25f * (d_ac1 + r_c1 - d_ab - d_kc);
    } else {
        int32_t qt = *q_tail_ptr;
        sq->u[qt] = b;
        sq->c[qt] = u;
        sq->pow_val[qt] = pow_neg_len;
        sq->dkc[qt] = d_kc;
        sq->dist[qt] = curr_dist;
        *q_tail_ptr = qt + 1;
    }
}

__device__ __forceinline__ void
dev_seed_half_tree_fifo(int32_t start_node, int32_t orig_node, int32_t parent_node, int32_t q_base, int32_t n_leaves,
                        int32_t n_nodes_u, const int32_t *__restrict__ adj_u, const int32_t *__restrict__ parent_u,
                        const int32_t *__restrict__ sub_size, const float *__restrict__ nd_u,
                        const float *__restrict__ row_root, const float *__restrict__ row_orig,
                        float *__restrict__ edge_delta, SprSharedQueue *__restrict__ sq, int32_t *__restrict__ out_head,
                        int32_t *__restrict__ out_tail) {
    int32_t q_tail = q_base;
    int32_t q_lim = q_base + 128;
    int32_t tau = (n_nodes_u > 512) ? (n_nodes_u / 64) : 16;

    if (start_node >= n_leaves) {
        int32_t n0 = adj_u[3 * start_node + 0], n1 = adj_u[3 * start_node + 1];
        int32_t k_in = (n0 == parent_node) ? 0 : ((n1 == parent_node) ? 1 : 2);
        int32_t k0 = (k_in == 0) ? 1 : 0;
        int32_t k1 = (k_in == 2) ? 1 : 2;

        int32_t B0 = adj_u[3 * start_node + k0], B1 = adj_u[3 * start_node + k1];
        float D_AB = nd_u[3 * start_node + k_in];
        float d_kc_init = row_root[orig_node];

        float new_dist0 = 0.25f * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
        edge_delta[(parent_u[start_node] == B0) ? start_node : B0] = new_dist0;

        float new_dist1 = 0.25f * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
        edge_delta[(parent_u[start_node] == B1) ? start_node : B1] = new_dist1;

        int32_t h_u[18], h_c[18];
        float h_pow[18], h_dkc[18], h_dist[18];
        int32_t h_top = 0;

        if (B0 >= n_leaves) {
            int32_t s0 = (parent_u[B0] == start_node) ? sub_size[B0] : (n_nodes_u - sub_size[start_node]);
            if (s0 > tau) {
                h_u[h_top] = B0;
                h_c[h_top] = start_node;
                h_pow[h_top] = 0.25f;
                h_dkc[h_top] = 0.5f * (d_kc_init + row_root[B1]);
                h_dist[h_top] = new_dist0;
                h_top++;
            } else {
                dev_eval_cherry_or_enqueue(B0, start_node, s0, 0.25f, 0.5f * (d_kc_init + row_root[B1]), new_dist0,
                                           adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
            }
        }
        if (B1 >= n_leaves) {
            int32_t s1 = (parent_u[B1] == start_node) ? sub_size[B1] : (n_nodes_u - sub_size[start_node]);
            if (s1 > tau) {
                h_u[h_top] = B1;
                h_c[h_top] = start_node;
                h_pow[h_top] = 0.25f;
                h_dkc[h_top] = 0.5f * (d_kc_init + row_root[B0]);
                h_dist[h_top] = new_dist1;
                h_top++;
            } else {
                dev_eval_cherry_or_enqueue(B1, start_node, s1, 0.25f, 0.5f * (d_kc_init + row_root[B0]), new_dist1,
                                           adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
            }
        }

        while (h_top > 0) {
            --h_top;
            int32_t u = h_u[h_top], C = h_c[h_top];
            float pow_neg_len = h_pow[h_top], d_kc = h_dkc[h_top], curr_dist = h_dist[h_top];

            for (;;) {
                if (q_tail + h_top + 2 > q_lim) {
                    sq->u[q_tail] = u;
                    sq->c[q_tail] = C;
                    sq->pow_val[q_tail] = pow_neg_len;
                    sq->dkc[q_tail] = d_kc;
                    sq->dist[q_tail] = curr_dist;
                    q_tail++;
                    while (h_top > 0) {
                        --h_top;
                        sq->u[q_tail] = h_u[h_top];
                        sq->c[q_tail] = h_c[h_top];
                        sq->pow_val[q_tail] = h_pow[h_top];
                        sq->dkc[q_tail] = h_dkc[h_top];
                        sq->dist[q_tail] = h_dist[h_top];
                        q_tail++;
                    }
                    break;
                }
                int32_t a0 = adj_u[3 * u + 0], a1 = adj_u[3 * u + 1];
                int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
                int32_t idx0 = (ki == 0) ? 1 : 0;
                int32_t idx1 = (ki == 2) ? 1 : 2;

                int32_t b0 = adj_u[3 * u + idx0], b1 = adj_u[3 * u + idx1];
                float d_ab = nd_u[3 * u + ki], r_b0 = row_root[b0], r_b1 = row_root[b1];

                float d_ac0 = nd_u[3 * u + idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
                float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
                edge_delta[(parent_u[u] == b0) ? u : b0] = ndist0;

                float d_ac1 = nd_u[3 * u + idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
                float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
                edge_delta[(parent_u[u] == b1) ? u : b1] = ndist1;

                float next_pow = pow_neg_len * 0.5f;
                int32_t s0 = (b0 >= n_leaves) ? ((parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u])) : 0;
                int32_t s1 = (b1 >= n_leaves) ? ((parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u])) : 0;

                if (s0 > tau && s1 > tau) {
                    if (s0 >= s1) {
                        h_u[h_top] = b0;
                        h_c[h_top] = u;
                        h_pow[h_top] = next_pow;
                        h_dkc[h_top] = 0.5f * (d_kc + r_b1);
                        h_dist[h_top] = ndist0;
                        h_top++;
                        C = u;
                        u = b1;
                        pow_neg_len = next_pow;
                        d_kc = 0.5f * (d_kc + r_b0);
                        curr_dist = ndist1;
                    } else {
                        h_u[h_top] = b1;
                        h_c[h_top] = u;
                        h_pow[h_top] = next_pow;
                        h_dkc[h_top] = 0.5f * (d_kc + r_b0);
                        h_dist[h_top] = ndist1;
                        h_top++;
                        C = u;
                        u = b0;
                        pow_neg_len = next_pow;
                        d_kc = 0.5f * (d_kc + r_b1);
                        curr_dist = ndist0;
                    }
                } else if (s0 > tau) {
                    if (b1 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b1, u, s1, next_pow, 0.5f * (d_kc + r_b0), ndist1, adj_u, parent_u,
                                                   nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    C = u;
                    u = b0;
                    pow_neg_len = next_pow;
                    d_kc = 0.5f * (d_kc + r_b1);
                    curr_dist = ndist0;
                } else if (s1 > tau) {
                    if (b0 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b0, u, s0, next_pow, 0.5f * (d_kc + r_b1), ndist0, adj_u, parent_u,
                                                   nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    C = u;
                    u = b1;
                    pow_neg_len = next_pow;
                    d_kc = 0.5f * (d_kc + r_b0);
                    curr_dist = ndist1;
                } else {
                    if (b0 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b0, u, s0, next_pow, 0.5f * (d_kc + r_b1), ndist0, adj_u, parent_u,
                                                   nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    if (b1 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b1, u, s1, next_pow, 0.5f * (d_kc + r_b0), ndist1, adj_u, parent_u,
                                                   nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    break;
                }
            }
        }
    }
    *out_head = q_base;
    *out_tail = q_tail;
}

__device__ __forceinline__ void dev_run_parallel_spr_dfs(
    int32_t tree_root, int32_t parent, int32_t other0, int32_t other1, int32_t n_leaves, int32_t n_nodes_u,
    const int32_t *__restrict__ adj_u, const int32_t *__restrict__ parent_u, const int32_t *__restrict__ sub_size,
    const float *__restrict__ nd_u, const float *__restrict__ row_root, const float *__restrict__ row_o0,
    const float *__restrict__ row_o1, float *__restrict__ edge_delta, SprSharedQueue *__restrict__ sq) {
    for (int32_t u = threadIdx.x; u < n_nodes_u; u += blockDim.x) {
        edge_delta[u] = 1e30f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 1e30f;
        edge_delta[(parent_u[other0] == parent) ? other0 : parent] = 1e30f;
        edge_delta[(parent_u[other1] == parent) ? other1 : parent] = 1e30f;
        dev_seed_half_tree_fifo(other0, other1, parent, 0, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                                row_root, row_o1, edge_delta, sq, &sq->bounds[0], &sq->bounds[1]);
    } else if (threadIdx.x == 32) {
        dev_seed_half_tree_fifo(other1, other0, parent, 128, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                                row_root, row_o0, edge_delta, sq, &sq->bounds[2], &sq->bounds[3]);
    }
    __syncthreads();

    int32_t h0 = sq->bounds[0], t0 = sq->bounds[1];
    int32_t h1 = sq->bounds[2], t1 = sq->bounds[3];
    int32_t n0 = t0 - h0, n1 = t1 - h1, total = n0 + n1;
    for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int32_t q_idx = (idx < n0) ? (h0 + idx) : (h1 + (idx - n0));
        const float *r_orig = (idx < n0) ? row_o1 : row_o0;
        dev_expand_subtree_stack(sq->u[q_idx], sq->c[q_idx], sq->pow_val[q_idx], sq->dkc[q_idx], sq->dist[q_idx],
                                 n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, row_root, r_orig, edge_delta);
    }
    __syncthreads();
}

__global__ void k_spr_rows_top1(int32_t r_start, int32_t n_leaves, int32_t n_nodes_u, int32_t other_root_child,
                                const int64_t *__restrict__ v_curr, const int32_t *__restrict__ v_map,
                                const int32_t *__restrict__ adj_u, const int32_t *__restrict__ parent_u,
                                const int32_t *__restrict__ sub_size, const int32_t *__restrict__ m_lmin,
                                const int32_t *__restrict__ next_k_same_lmin, const float *__restrict__ subdist,
                                const float *__restrict__ nd_u, float *__restrict__ row_tmp_pool,
                                float *__restrict__ best_delta_out, int32_t *__restrict__ best_j_out) {
    extern __shared__ float s_edge_delta_top1_buf[];
    __shared__ SprSharedQueue s_q;
    __shared__ float s_best_val[128];
    __shared__ int32_t s_best_j[128];

    int32_t v_index = r_start + blockIdx.x;
    int32_t i = v_index - 1;
    int32_t root_u = n_nodes_u - 1;
    float *edge_delta = s_edge_delta_top1_buf;
    (void)row_tmp_pool;
    (void)v_curr;
    (void)m_lmin;
    (void)next_k_same_lmin;

    int32_t tree_root = v_map[v_index];
    int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
    int32_t other0 = -1, other1 = -1;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb_node = adj_u[3 * parent + k];
        if (nb_node != -1 && nb_node != tree_root) {
            if (other0 == -1)
                other0 = nb_node;
            else
                other1 = nb_node;
        }
    }
    if (other0 == -1) {
        if (threadIdx.x == 0) {
            best_delta_out[i] = 1e30f;
            best_j_out[i] = -1;
        }
        return;
    }

    dev_run_parallel_spr_dfs(tree_root, parent, other0, other1, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                             &subdist[(size_t)tree_root * n_nodes_u], &subdist[(size_t)other0 * n_nodes_u],
                             &subdist[(size_t)other1 * n_nodes_u], edge_delta, &s_q);

    float b_val = 1e30f;
    int32_t b_j = -1;
    for (int32_t du = threadIdx.x; du < root_u; du += blockDim.x) {
        float val = edge_delta[du];
        if (val < b_val || (val == b_val && du < b_j)) {
            b_val = val;
            b_j = du;
        }
    }
    s_best_val[threadIdx.x] = b_val;
    s_best_j[threadIdx.x] = b_j;
    __syncthreads();

    for (int32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float v_other = s_best_val[threadIdx.x + stride];
            int32_t j_other = s_best_j[threadIdx.x + stride];
            float v_self = s_best_val[threadIdx.x];
            int32_t j_self = s_best_j[threadIdx.x];
            if (v_other < v_self || (v_other == v_self && j_other < j_self)) {
                s_best_val[threadIdx.x] = v_other;
                s_best_j[threadIdx.x] = j_other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        best_delta_out[i] = s_best_val[0];
        best_j_out[i] = s_best_j[0];
    }
}

// ---- Start tree: gathering a subsample's rows; the nearest sampled sequence ----

__global__ void k_gather_rows(const uint64_t *__restrict__ src, const int32_t *__restrict__ idx, int64_t n_rows,
                              int32_t row_words, uint64_t *__restrict__ dst) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = n_rows * row_words;
    for (; t < total; t += (int64_t)gridDim.x * blockDim.x) {
        int64_t r = t / row_words, off = t % row_words;
        dst[t] = src[(int64_t)idx[r] * row_words + off];
    }
}

// Tiled nearest-sample search.
// Block = 64 queries x (all samples, 64 at a time), 256 threads, each thread owns a 4x4 block of (query, sample) pairs.
// Sequences are streamed through shared memory 4 words at a time.
// Output: position (in samp[]) of the nearest sample per query.
#define NN_TQ 64
#define NN_TS 64
#define NN_WC 4
__global__ void __launch_bounds__(256)
    k_nearest_sample(const uint64_t *__restrict__ seqs, int32_t W, const int32_t *__restrict__ qry, int32_t n_q,
                     const int32_t *__restrict__ samp, int32_t n_s, int32_t *__restrict__ best_pos) {
    __shared__ uint64_t qs[5][NN_WC][NN_TQ];
    __shared__ uint64_t ss[5][NN_WC][NN_TS];
    __shared__ unsigned long long best_key[NN_TQ];
    const int t = threadIdx.x;
    const int tq = t / 16, ts = t % 16;
    const int32_t q0 = blockIdx.x * NN_TQ;
    const int64_t row_words = 5LL * W;
    if (t < NN_TQ)
        best_key[t] = ~0ULL;

    for (int32_t s0 = 0; s0 < n_s; s0 += NN_TS) {
        int32_t match[4][4], valid[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                match[a][b] = 0;
                valid[a][b] = 0;
            }

        for (int32_t w0 = 0; w0 < W; w0 += NN_WC) {
            __syncthreads();
            for (int e = t; e < 5 * NN_WC * NN_TQ; e += 256) {
                int row = e % NN_TQ, rest = e / NN_TQ;
                int wl = rest % NN_WC, p = rest / NN_WC;
                int32_t w = w0 + wl;
                int32_t qi = q0 + row, si = s0 + row;
                qs[p][wl][row] = (qi < n_q && w < W) ? seqs[(int64_t)qry[qi] * row_words + (int64_t)p * W + w] : 0ULL;
                ss[p][wl][row] = (si < n_s && w < W) ? seqs[(int64_t)samp[si] * row_words + (int64_t)p * W + w] : 0ULL;
            }
            __syncthreads();
#pragma unroll
            for (int wl = 0; wl < NN_WC; ++wl) {
                uint64_t qa[4], qc[4], qg[4], qt[4], qv[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    int r = tq * 4 + a;
                    qa[a] = qs[0][wl][r];
                    qc[a] = qs[1][wl][r];
                    qg[a] = qs[2][wl][r];
                    qt[a] = qs[3][wl][r];
                    qv[a] = qs[4][wl][r];
                }
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int r = ts * 4 + b;
                    uint64_t sa = ss[0][wl][r], sc = ss[1][wl][r], sg = ss[2][wl][r];
                    uint64_t st = ss[3][wl][r], sv = ss[4][wl][r];
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        valid[a][b] += __popcll(qv[a] & sv);
                        match[a][b] += __popcll((qa[a] & sa) | (qc[a] & sc) | (qg[a] & sg) | (qt[a] & st));
                    }
                }
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            unsigned long long kbest = ~0ULL;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                int32_t si = s0 + ts * 4 + b;
                if (si >= n_s)
                    continue;
                float p = (valid[a][b] > 0) ? (float)(valid[a][b] - match[a][b]) / (float)valid[a][b]
                                            : p2vbme::NN_P_NO_OVERLAP;
                unsigned long long key = ((unsigned long long)__float_as_uint(p) << 32) | (unsigned int)si;
                if (key < kbest)
                    kbest = key;
            }
            if (kbest != ~0ULL)
                atomicMin(&best_key[tq * 4 + a], kbest);
        }
    }
    __syncthreads();
    if (t < NN_TQ && q0 + t < n_q) {
        unsigned long long k = best_key[t];
        best_pos[q0 + t] = (k == ~0ULL) ? 0 : (int32_t)(k & 0xffffffffULL);
    }
}

// ---- Branch lengths: the edge-length pass ----

__device__ __forceinline__ float dev_jc69_from_nv_nm(int32_t n_valid, int32_t n_match) {
    return jc69_distance<float>(n_valid, n_match);
}

__device__ __forceinline__ float dev_jc69_bl_popc(const uint64_t *__restrict__ d_packed_seqs, int32_t W, int32_t leaf_u,
                                                  int32_t leaf_v) {
    if (leaf_u == leaf_v)
        return 0.0f;
    const uint64_t *su = d_packed_seqs + (size_t)leaf_u * 5 * W;
    const uint64_t *sv = d_packed_seqs + (size_t)leaf_v * 5 * W;
    int32_t n_valid = 0;
    int32_t n_match = 0;
    for (int32_t w = 0; w < W; ++w) {
        uint64_t vu = su[4 * W + w];
        uint64_t vv = sv[4 * W + w];
        n_valid += __popcll(vu & vv);
        uint64_t m = (su[0 * W + w] & sv[0 * W + w]) | (su[1 * W + w] & sv[1 * W + w]) |
                     (su[2 * W + w] & sv[2 * W + w]) | (su[3 * W + w] & sv[3 * W + w]);
        n_match += __popcll(m);
    }
    return dev_jc69_from_nv_nm(n_valid, n_match);
}

constexpr int32_t BL_TILE = 32;
constexpr int32_t BL_CW = 4;

// 64 threads per block (8x8 grid of 4x4 register tiles covering 32x32 tiles).
// One block evaluates one (local_i, kind) job in [0, 3 * n_int_chunk).
//   - Fast path when len1 == 1 or len2 == 1: 64 threads evaluate up to 64 leaf pairs in 1 step.
//   - Tiled shared-memory path when len1 > 1 and len2 > 1: stages sequences into sA/sB in
//     chunks of BL_CW = 4 words and accumulates 4x4 popcount tiles in registers.
template <bool JC>
__global__ __launch_bounds__(64) void k_eval_internal_node_triples_tiled(
    int32_t n_int_chunk, int32_t N, int32_t W, int32_t cap, const int32_t *__restrict__ d_c1,
    const int32_t *__restrict__ d_c2, const int32_t *__restrict__ d_in_len, const int32_t *__restrict__ d_in_leaves,
    const float *__restrict__ d_in_weights, const int32_t *__restrict__ d_out_len,
    const int32_t *__restrict__ d_out_leaves, const float *__restrict__ d_out_weights,
    const uint64_t *__restrict__ d_packed_seqs, float *__restrict__ d_dAB, float *__restrict__ d_dAO,
    float *__restrict__ d_dBO, DeviceDistance dm) {
    __shared__ uint64_t sA[5 * BL_CW * BL_TILE];
    __shared__ uint64_t sB[5 * BL_CW * BL_TILE];
    __shared__ int32_t s_la[BL_TILE], s_lb[BL_TILE];
    __shared__ float s_wa[BL_TILE], s_wb[BL_TILE];
    __shared__ float s_warp_sum[2];

    const int32_t job_id = (int32_t)blockIdx.x;
    if (job_id >= 3 * n_int_chunk)
        return;

    const int32_t local_i = job_id / 3;
    const int32_t kind = job_id - local_i * 3; // 0: AB, 1: AO, 2: BO
    const int32_t tid = (int32_t)threadIdx.x;
    const int32_t tx = tid & 7;
    const int32_t ty = tid >> 3;

    const int32_t u_left = d_c1[local_i];
    const int32_t u_right = d_c2[local_i];

    int32_t len1 = 0, len2 = 0;
    int32_t single_leaf1 = -1, single_leaf2 = -1;
    const int32_t *leaves1 = nullptr;
    const float *weights1 = nullptr;
    const int32_t *leaves2 = nullptr;
    const float *weights2 = nullptr;

    int32_t node1 = (kind == 2) ? u_right : u_left;
    if (node1 < N) {
        len1 = 1;
        single_leaf1 = node1;
    } else {
        int32_t idx1 = node1 - N;
        len1 = d_in_len[idx1];
        leaves1 = d_in_leaves + (size_t)idx1 * cap;
        weights1 = d_in_weights + (size_t)idx1 * cap;
    }

    if (kind == 0) {
        if (u_right < N) {
            len2 = 1;
            single_leaf2 = u_right;
        } else {
            int32_t idx2 = u_right - N;
            len2 = d_in_len[idx2];
            leaves2 = d_in_leaves + (size_t)idx2 * cap;
            weights2 = d_in_weights + (size_t)idx2 * cap;
        }
    } else {
        len2 = d_out_len[local_i];
        leaves2 = d_out_leaves + (size_t)local_i * cap;
        weights2 = d_out_weights + (size_t)local_i * cap;
    }

    float thread_sum = 0.0f;

    // Per-pair path: at least one side has length <= 2 (covers all N pendant leaf edges), or a model other than JC69
    // (the tiled path counts JC69's two numbers only), for which every pair goes this way.
    if (!JC || len1 <= 2 || len2 <= 2) {
        const int32_t total = len1 * len2;
        for (int32_t idx = tid; idx < total; idx += 64) {
            const int32_t i = idx / len2;
            const int32_t j = idx - i * len2;
            const int32_t lu = (single_leaf1 >= 0) ? single_leaf1 : leaves1[i];
            const float wu = (single_leaf1 >= 0) ? 1.0f : weights1[i];
            const int32_t lv = (single_leaf2 >= 0) ? single_leaf2 : leaves2[j];
            const float wv = (single_leaf2 >= 0) ? 1.0f : weights2[j];
            if constexpr (JC)
                thread_sum += wu * wv * dev_jc69_bl_popc(d_packed_seqs, W, lu, lv);
            else
                thread_sum += wu * wv * dev_model_dist(lu, lv, d_packed_seqs, W, dm);
        }
    } else {
        // Shared-memory staged 32x32 tiled path with 4x4 register tiles per thread
        for (int32_t pa0 = 0; pa0 < len1; pa0 += BL_TILE) {
            const int32_t na = min(BL_TILE, len1 - pa0);
            if (tid < na) {
                s_la[tid] = leaves1[pa0 + tid];
                s_wa[tid] = weights1[pa0 + tid];
            }
            for (int32_t qb0 = 0; qb0 < len2; qb0 += BL_TILE) {
                const int32_t nb = min(BL_TILE, len2 - qb0);
                if (tid < nb) {
                    s_lb[tid] = leaves2[qb0 + tid];
                    s_wb[tid] = weights2[qb0 + tid];
                }
                __syncthreads();

                int32_t nv[4][4], nm[4][4];
#pragma unroll
                for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j) {
                        nv[i][j] = 0;
                        nm[i][j] = 0;
                    }

                for (int32_t w0 = 0; w0 < W; w0 += BL_CW) {
                    for (int32_t e = tid; e < 5 * BL_CW * BL_TILE; e += 64) {
                        const int32_t cw = e % BL_CW;
                        const int32_t ch = (e / BL_CW) % 5;
                        const int32_t r = e / (5 * BL_CW);
                        const int32_t s_idx = (ch * BL_CW + cw) * BL_TILE + r;
                        const int32_t w = w0 + cw;
                        const size_t g_off = (size_t)ch * W + w;
                        sA[s_idx] = (r < na && w < W) ? d_packed_seqs[(size_t)s_la[r] * 5 * W + g_off] : 0ULL;
                        sB[s_idx] = (r < nb && w < W) ? d_packed_seqs[(size_t)s_lb[r] * 5 * W + g_off] : 0ULL;
                    }
                    __syncthreads();

#pragma unroll
                    for (int32_t cw = 0; cw < BL_CW; ++cw) {
                        uint64_t ra[4], rb[4], m[4][4];
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
                            ra[i] = sA[(4 * BL_CW + cw) * BL_TILE + ty + 8 * i];
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                            rb[j] = sB[(4 * BL_CW + cw) * BL_TILE + tx + 8 * j];
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                            for (int32_t j = 0; j < 4; ++j)
                                nv[i][j] += __popcll(ra[i] & rb[j]);

#pragma unroll
                        for (int32_t c = 0; c < 4; ++c) {
#pragma unroll
                            for (int32_t i = 0; i < 4; ++i)
                                ra[i] = sA[(c * BL_CW + cw) * BL_TILE + ty + 8 * i];
#pragma unroll
                            for (int32_t j = 0; j < 4; ++j)
                                rb[j] = sB[(c * BL_CW + cw) * BL_TILE + tx + 8 * j];
#pragma unroll
                            for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                                for (int32_t j = 0; j < 4; ++j)
                                    m[i][j] = (c == 0) ? (ra[i] & rb[j]) : (m[i][j] | (ra[i] & rb[j]));
                        }
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
#pragma unroll
                            for (int32_t j = 0; j < 4; ++j)
                                nm[i][j] += __popcll(m[i][j]);
                    }
                    __syncthreads();
                }

#pragma unroll
                for (int32_t i = 0; i < 4; ++i) {
                    const int32_t p = ty + 8 * i;
                    if (p < na) {
                        const int32_t lp = s_la[p];
                        const float wp = s_wa[p];
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j) {
                            const int32_t q = tx + 8 * j;
                            if (q < nb) {
                                const int32_t lq = s_lb[q];
                                const float wq = s_wb[q];
                                const float d = (lp == lq) ? 0.0f : dev_jc69_from_nv_nm(nv[i][j], nm[i][j]);
                                thread_sum += wp * wq * d;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

// Reduce across the 64 threads (2 warps)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, offset);
    }
    if ((tid & 31) == 0) {
        s_warp_sum[tid >> 5] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total_sum = s_warp_sum[0] + s_warp_sum[1];
        if (kind == 0)
            d_dAB[local_i] = total_sum;
        else if (kind == 1)
            d_dAO[local_i] = total_sum;
        else
            d_dBO[local_i] = total_sum;
    }
}

// ---- 5. GPU Workspace for Full-Tree and Multi-GPU Minibatch Optimization ----

struct RnaSimGpuWorkspace {
    int32_t dev_id;
    int32_t max_K;
    size_t exact_prof_cap;
    size_t pair_cap;
    cudaStream_t stream = NULL;
    int32_t *d_prof_offsets = NULL;
    int32_t *d_prof_leaves = NULL;
    float *d_prof_weights = NULL;
    int32_t *d_exact_offsets = NULL;
    int32_t *d_exact_leaves = NULL;
    float *d_exact_weights = NULL;
    int32_t *d_pair_a = NULL;
    int32_t *d_pair_b = NULL;
    double *d_pair_exact_D = NULL;
    int32_t *d_perm = NULL;
    float *d_D = NULL;
    float *d_subdist = NULL;
    float *d_nd_u = NULL;
    float *d_row_scratch = NULL;
    int64_t *d_v_curr = NULL;
    int32_t *d_c1_r = NULL;
    int32_t *d_c2_r = NULL;
    int32_t *d_v_map = NULL;
    int32_t *d_m_lmin = NULL;
    int32_t *d_next_k = NULL;
    int32_t *d_parent_u = NULL;
    int32_t *d_sub_size = NULL;
    int32_t *d_tin = NULL;
    int32_t *d_adj_u = NULL;
    float *d_best_delta = NULL;
    int32_t *d_best_j = NULL;
    double *d_curr_L = NULL;
    float *d_refine_D_scratch = NULL;
    bool refine_split_ready;
    // Heavy-pair list for k_build_condensed_D_heavy_tiled (grown on demand).
    int32_t *d_heavy_a = NULL;
    int32_t *d_heavy_b = NULL;
    size_t heavy_cap;
    std::vector<int32_t> h_heavy_a, h_heavy_b;

    void init(int32_t gpu_id, int32_t K_cap) {
        dev_id = gpu_id;
        max_K = K_cap;
        d_refine_D_scratch = NULL;
        refine_split_ready = false;
        d_heavy_a = NULL;
        d_heavy_b = NULL;
        heavy_cap = 0;
        int32_t R = 2 * K_cap - 1;
        int32_t M = 2 * K_cap - 2;
        size_t prof_cap = std::max((size_t)256 * (size_t)K_cap, (size_t)4096);
        exact_prof_cap = std::max(prof_cap, (size_t)16384);
        pair_cap = (size_t)K_cap * (size_t)K_cap / 2 + 16;
        CUDA_CHECK(cudaSetDevice(dev_id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        gpu_alloc(&d_prof_offsets, (size_t)(K_cap + 1) * sizeof(int32_t));
        gpu_alloc(&d_prof_leaves, prof_cap * sizeof(int32_t));
        gpu_alloc(&d_prof_weights, prof_cap * sizeof(float));
        gpu_alloc(&d_exact_offsets, (size_t)(K_cap + 1) * sizeof(int32_t));
        gpu_alloc(&d_exact_leaves, exact_prof_cap * sizeof(int32_t));
        gpu_alloc(&d_exact_weights, exact_prof_cap * sizeof(float));
        gpu_alloc(&d_pair_a, pair_cap * sizeof(int32_t));
        gpu_alloc(&d_pair_b, pair_cap * sizeof(int32_t));
        gpu_alloc(&d_pair_exact_D, pair_cap * sizeof(double));
        gpu_alloc(&d_perm, (size_t)K_cap * sizeof(int32_t));
        gpu_alloc(&d_D, (size_t)K_cap * K_cap * sizeof(float));
        gpu_alloc(&d_subdist, (size_t)M * M * sizeof(float));
        gpu_alloc(&d_nd_u, (size_t)3 * M * sizeof(float));
        gpu_alloc(&d_row_scratch, (size_t)K_cap * (2 * K_cap) * sizeof(float));
        gpu_alloc(&d_v_curr, (size_t)K_cap * sizeof(int64_t));
        gpu_alloc(&d_c1_r, (size_t)R * sizeof(int32_t));
        gpu_alloc(&d_c2_r, (size_t)R * sizeof(int32_t));
        gpu_alloc(&d_v_map, (size_t)K_cap * sizeof(int32_t));
        gpu_alloc(&d_m_lmin, (size_t)K_cap * sizeof(int32_t));
        gpu_alloc(&d_next_k, (size_t)K_cap * sizeof(int32_t));
        gpu_alloc(&d_parent_u, (size_t)M * sizeof(int32_t));
        gpu_alloc(&d_sub_size, (size_t)M * sizeof(int32_t));
        gpu_alloc(&d_tin, (size_t)M * sizeof(int32_t));
        gpu_alloc(&d_adj_u, (size_t)3 * M * sizeof(int32_t));
        gpu_alloc(&d_best_delta, (size_t)K_cap * sizeof(float));
        gpu_alloc(&d_best_j, (size_t)K_cap * sizeof(int32_t));
        gpu_alloc(&d_curr_L, sizeof(double));
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, 90112));
    }

    void ensure_exact_prof_cap(size_t needed) {
        if (needed > exact_prof_cap) {
            CUDA_CHECK(cudaSetDevice(dev_id));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // freed pointers are cleared and the capacity set only once allocated, so a failed allocation leaves
            // nothing for destroy() to free twice
            cudaFree(d_exact_leaves);
            cudaFree(d_exact_weights);
            d_exact_leaves = NULL;
            d_exact_weights = NULL;
            exact_prof_cap = 0;
            const size_t new_cap = needed + 4096;
            gpu_alloc(&d_exact_leaves, new_cap * sizeof(int32_t));
            gpu_alloc(&d_exact_weights, new_cap * sizeof(float));
            exact_prof_cap = new_cap;
        }
        if (!refine_split_ready) {
            // Room for the load-balanced exact refinement in optimize_skeleton_on_gpu: up to 4 * max_K virtual CSR
            // slots and appended launch pairs, plus a D scratch matrix the refinement kernel writes at indices
            // a * K + b and b * K + a with a, b < 5 * K.
            // Allocated on first use, so workspaces that never refine exactly do not pay for it.
            CUDA_CHECK(cudaSetDevice(dev_id));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            size_t extra = (size_t)4 * (size_t)max_K;
            cudaFree(d_exact_offsets);
            cudaFree(d_pair_a);
            cudaFree(d_pair_b);
            cudaFree(d_pair_exact_D);
            d_exact_offsets = d_pair_a = d_pair_b = NULL;
            d_pair_exact_D = NULL;
            const size_t new_pair_cap = pair_cap + extra;
            gpu_alloc(&d_exact_offsets, ((size_t)max_K + 1 + extra) * sizeof(int32_t));
            gpu_alloc(&d_pair_a, new_pair_cap * sizeof(int32_t));
            gpu_alloc(&d_pair_b, new_pair_cap * sizeof(int32_t));
            gpu_alloc(&d_pair_exact_D, new_pair_cap * sizeof(double));
            pair_cap = new_pair_cap;
            gpu_alloc(&d_refine_D_scratch, (size_t)5 * (size_t)max_K * ((size_t)max_K + 1) * sizeof(float));
            refine_split_ready = true;
        }
    }

    // Frees everything allocated so far (also after a failed init); never throws.
    void destroy() noexcept {
        cudaSetDevice(dev_id);
        if (stream)
            cudaStreamDestroy(stream);
        stream = NULL;
        cudaFree(d_prof_offsets);
        cudaFree(d_prof_leaves);
        cudaFree(d_prof_weights);
        cudaFree(d_exact_offsets);
        cudaFree(d_exact_leaves);
        cudaFree(d_exact_weights);
        cudaFree(d_pair_a);
        cudaFree(d_pair_b);
        cudaFree(d_pair_exact_D);
        cudaFree(d_perm);
        cudaFree(d_D);
        cudaFree(d_subdist);
        cudaFree(d_nd_u);
        cudaFree(d_row_scratch);
        cudaFree(d_v_curr);
        cudaFree(d_c1_r);
        cudaFree(d_c2_r);
        cudaFree(d_v_map);
        cudaFree(d_m_lmin);
        cudaFree(d_next_k);
        cudaFree(d_parent_u);
        cudaFree(d_sub_size);
        cudaFree(d_tin);
        cudaFree(d_adj_u);
        cudaFree(d_best_delta);
        cudaFree(d_best_j);
        cudaFree(d_curr_L);
        cudaFree(d_refine_D_scratch);
        d_refine_D_scratch = NULL;
        refine_split_ready = false;
        cudaFree(d_heavy_a);
        cudaFree(d_heavy_b);
        d_heavy_a = d_heavy_b = NULL;
        heavy_cap = 0;
    }
};

// The GPU parts of a window solve; the sweeps of batched moves and the exact check's bookkeeping are
// solve_window_host's (search.cpp), shared with the CPU device.
class GpuWindowKernels final : public WindowKernels {
  public:
    GpuWindowKernels(RnaSimGpuWorkspace &ws, const uint64_t *seqs, int32_t n_words, const DeviceDistance &model)
        : gpu_ws(ws), d_packed_seqs(seqs), W(n_words), dm(model), jc69(model.model == Model::JC69) {}

    void distances(int32_t K_window, const int32_t *profile_offsets, const int32_t *profile_leaves,
                   const float *profile_weights) override {
        K = K_window;
        M_sk = 2 * K - 2;
        root_u_sk = M_sk - 1;
        CUDA_CHECK(cudaSetDevice(gpu_ws.dev_id));
        int32_t total_prof = profile_offsets[K];
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_offsets, profile_offsets, (size_t)(K + 1) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_leaves, profile_leaves, (size_t)total_prof * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_weights, profile_weights, (size_t)total_prof * sizeof(float),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        dim3 blk2d(16, 16);
        dim3 grd2d((K + 15) / 16, (K + 15) / 16);
        // JC69: pairs whose profiles both have >= HB_HEAVY_MIN leaves go to the tiled kernel; the rest (singletons,
        // singleton x outgroup) stay on the per-thread kernel.
        // Other models: all pairs there.
        const int32_t heavy_min = (W > 0 && jc69) ? HB_HEAVY_MIN : INT32_MAX;
        gpu_ws.h_heavy_a.clear();
        gpu_ws.h_heavy_b.clear();
        if (heavy_min != INT32_MAX) {
            std::vector<int32_t> heavy_nodes;
            for (int32_t a = 0; a < K; ++a) {
                if (profile_offsets[a + 1] - profile_offsets[a] >= heavy_min)
                    heavy_nodes.push_back(a);
            }
            for (size_t ia = 0; ia < heavy_nodes.size(); ++ia) {
                for (size_t ib = ia + 1; ib < heavy_nodes.size(); ++ib) {
                    gpu_ws.h_heavy_a.push_back(heavy_nodes[ia]);
                    gpu_ws.h_heavy_b.push_back(heavy_nodes[ib]);
                }
            }
        }
        const int32_t n_heavy = (int32_t)gpu_ws.h_heavy_a.size();
        if ((size_t)n_heavy > gpu_ws.heavy_cap) {
            cudaFree(gpu_ws.d_heavy_a);
            cudaFree(gpu_ws.d_heavy_b);
            gpu_ws.d_heavy_a = gpu_ws.d_heavy_b = NULL;
            gpu_ws.heavy_cap = 0;
            gpu_alloc(&gpu_ws.d_heavy_a, (size_t)n_heavy * sizeof(int32_t));
            gpu_alloc(&gpu_ws.d_heavy_b, (size_t)n_heavy * sizeof(int32_t));
            gpu_ws.heavy_cap = (size_t)n_heavy;
        }
        if (jc69)
            k_build_real_seq_condensed_D<true>
                <<<grd2d, blk2d, 0, gpu_ws.stream>>>(K, W, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves,
                                                     gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D, heavy_min, dm);
        else
            k_build_real_seq_condensed_D<false>
                <<<grd2d, blk2d, 0, gpu_ws.stream>>>(K, W, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves,
                                                     gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D, heavy_min, dm);
        CUDA_CHECK(cudaGetLastError());
        if (n_heavy > 0) {
            CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_heavy_a, gpu_ws.h_heavy_a.data(), (size_t)n_heavy * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, gpu_ws.stream));
            CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_heavy_b, gpu_ws.h_heavy_b.data(), (size_t)n_heavy * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, gpu_ws.stream));
            k_build_condensed_D_heavy_tiled<<<n_heavy, 256, 0, gpu_ws.stream>>>(
                K, W, n_heavy, gpu_ws.d_heavy_a, gpu_ws.d_heavy_b, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves,
                gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    double score(const SkeletonMeta &s, const std::vector<int32_t> &perm) override {
        const int32_t R_sk = 2 * K - 1;
        dim3 blk2d(16, 16);
        dim3 grd2d((K + 15) / 16, (K + 15) / 16);
        const std::vector<int32_t> &c1_cf = s.c1, &c2_cf = s.c2, &v_map_cf = s.v_map, &parent_u_cf = s.parent_u;
        const std::vector<int32_t> &sub_size_cf = s.sub_size, &tin_cf = s.tin, &adj_u_cf = s.adj_u;
        const int32_t rc0 = s.rc0, rc1 = s.rc1, rc2 = s.rc2;
        double curr_L = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_perm, perm.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_c1_r, c1_cf.data(), (size_t)R_sk * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_c2_r, c2_cf.data(), (size_t)R_sk * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_v_map, v_map_cf.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_parent_u, parent_u_cf.data(), (size_t)M_sk * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_sub_size, sub_size_cf.data(), (size_t)M_sk * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_tin, tin_cf.data(), (size_t)M_sk * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_adj_u, adj_u_cf.data(), (size_t)3 * M_sk * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));

        k_copy_leaf_rows_perm<<<grd2d, blk2d, 0, gpu_ws.stream>>>(gpu_ws.d_D, gpu_ws.d_perm, gpu_ws.d_subdist, K, M_sk);
        k_bottom_up_leaves<<<(K + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r,
                                                                       K, M_sk, root_u_sk);
        int32_t n_int = root_u_sk - K;
        if (n_int > 0) {
            dim3 grd_tr((K + 15) / 16, (n_int + 15) / 16);
            k_transpose_leaf_block<<<grd_tr, blk2d, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, K, M_sk, root_u_sk);
            k_bottom_up_internals<<<(n_int + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r,
                                                                                  gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        }
        k_top_down_and_root<<<(M_sk + 255) / 256, 256, 0, gpu_ws.stream>>>(
            gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, gpu_ws.d_tin, gpu_ws.d_sub_size, K, M_sk, root_u_sk, rc0,
            rc1, rc2);
        k_compute_bme_length<<<1, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_parent_u, K, M_sk,
                                                           gpu_ws.d_curr_L);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&curr_L, gpu_ws.d_curr_L, sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
        return curr_L;
    }

    void best_moves(const SkeletonMeta &s, std::vector<double> &best_delta, std::vector<int32_t> &h_best_j) override {
        const int32_t other_rc = s.other_rc;
        std::vector<float> h_best_delta(K);
        h_best_j.resize(K);
        k_populate_nd_u<<<(K - 1 + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_adj_u, gpu_ws.d_nd_u,
                                                                        K, M_sk, root_u_sk);
        k_spr_rows_top1<<<K - 2, 128, (size_t)M_sk * sizeof(float), gpu_ws.stream>>>(
            2, K, M_sk, other_rc, gpu_ws.d_v_curr, gpu_ws.d_v_map, gpu_ws.d_adj_u, gpu_ws.d_parent_u, gpu_ws.d_sub_size,
            gpu_ws.d_m_lmin, gpu_ws.d_next_k, gpu_ws.d_subdist, gpu_ws.d_nd_u, gpu_ws.d_row_scratch,
            gpu_ws.d_best_delta, gpu_ws.d_best_j);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_best_delta.data(), gpu_ws.d_best_delta, (size_t)K * sizeof(float),
                                   cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_best_j.data(), gpu_ws.d_best_j, (size_t)K * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
        best_delta.assign(h_best_delta.begin(), h_best_delta.end()); // exact: float to double
        // rows 0 and K - 1 have no move (the kernel leaves them unwritten)
        best_delta[0] = best_delta[K - 1] = 1e30;
        h_best_j[0] = h_best_j[K - 1] = -1;
    }

    void exact_distances(const std::vector<int32_t> &pair_a, const std::vector<int32_t> &pair_b,
                         const int32_t *exact_profile_offsets, const int32_t *exact_profile_leaves,
                         const float *exact_profile_weights, std::vector<double> &h_pair_exact_D) override {
        h_pair_exact_D.assign(pair_a.size(), 0.0);
        if (pair_a.empty())
            return;
        int32_t n_pairs = (int32_t)pair_a.size();
        int32_t total_exact_prof = exact_profile_offsets[K];
        gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);

        // Load balancing.
        // The refinement kernel runs one 256-thread block per launch pair, so a capped pair with ~1e9 leaf pairs would
        // occupy one SM while the rest of the GPU idles.
        // Such pairs are split into row chunks of their smaller exact profile.
        // Each chunk is a virtual piece id >= K + 1 whose CSR range in refine_offsets points into the same exact
        // leaf and weight arrays.
        // Chunk 0 takes the pair's own launch slot, chunks 1..c-1 are appended after n_pairs, and the chunk sums are
        // added back per pair in a fixed order, so results are deterministic.
        // With a split, d_exact_offsets receives K + 1 + slots entries, the launch has n_launch > n_pairs slots, and D
        // is written to the scratch matrix because virtual ids fall outside the K x K layout of d_D.
        // Without a heavy pair the launch equals the unsplit one.
        std::vector<int32_t> launch_a(pair_a), launch_b(pair_b), chunk_owner;
        std::vector<int32_t> refine_offsets;
        {
            const double chunk_leaf_pairs = 4194304.0;  // target leaf pairs per block
            const int64_t b_chunk_target = 6144;        // ~6-8k leaf B-chunks sized for L2 cache
            const int64_t max_virtual = (int64_t)4 * K; // slots reserved by ensure_exact_prof_cap
            std::vector<int32_t> heavy_pid, heavy_small, heavy_large;
            std::vector<int64_t> heavy_rows;
            std::vector<std::vector<int32_t>> b_split_pids((size_t)K);
            std::vector<int32_t> b_split_profs;
            for (int32_t pid = 0; pid < n_pairs; ++pid) {
                int32_t u = pair_a[pid], v = pair_b[pid];
                int64_t lu = exact_profile_offsets[u + 1] - exact_profile_offsets[u];
                int64_t lv = exact_profile_offsets[v + 1] - exact_profile_offsets[v];
                int32_t s = (lu <= lv) ? u : v;
                int32_t l = (lu <= lv) ? v : u;
                int64_t ls = std::min(lu, lv), ll = std::max(lu, lv);
                int64_t b_c =
                    (ll > b_chunk_target) ? std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target) : 1;
                int64_t b_rows = (b_c >= 2) ? ((((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63) : ll;
                if (b_c >= 2 && (double)ls * (double)b_rows <= 2.0 * chunk_leaf_pairs) {
                    if (b_split_pids[l].empty())
                        b_split_profs.push_back(l);
                    b_split_pids[l].push_back(pid);
                } else if (ls >= 32 && (double)ls * (double)ll > 2.0 * chunk_leaf_pairs) {
                    int64_t rows = (int64_t)std::ceil(chunk_leaf_pairs / (double)ll);
                    rows = std::max((int64_t)16, ((rows + 15) / 16) * 16);
                    heavy_pid.push_back(pid);
                    heavy_small.push_back(s);
                    heavy_large.push_back(l);
                    heavy_rows.push_back(rows);
                }
            }
            int64_t b_slots = 0, b_extra_pairs = 0;
            for (int32_t l : b_split_profs) {
                int64_t ll = exact_profile_offsets[l + 1] - exact_profile_offsets[l];
                int64_t b_c = std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target);
                int64_t b_rows = (((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63;
                int64_t c = (ll + b_rows - 1) / b_rows;
                if (c >= 2) {
                    b_slots += c + 1;
                    b_extra_pairs += (c - 1) * (int64_t)b_split_pids[l].size();
                }
            }
            int64_t slots = 0, extra_pairs = 0;
            for (int32_t pass = 0; pass < 64; ++pass) {
                slots = b_slots;
                extra_pairs = b_extra_pairs;
                for (size_t h = 0; h < heavy_pid.size(); ++h) {
                    int32_t s = heavy_small[h];
                    int64_t ls = exact_profile_offsets[s + 1] - exact_profile_offsets[s];
                    int64_t c = (ls + heavy_rows[h] - 1) / heavy_rows[h];
                    if (c >= 2) {
                        slots += c + 1;
                        extra_pairs += c - 1;
                    }
                }
                if (slots <= max_virtual)
                    break;
                for (size_t h = 0; h < heavy_rows.size(); ++h)
                    heavy_rows[h] *= 2;
            }
            if (slots > 0 && slots <= max_virtual &&
                (size_t)(n_pairs + extra_pairs) <= (size_t)K * (size_t)K / 2 + 16 + (size_t)4 * K) {
                refine_offsets.assign(exact_profile_offsets, exact_profile_offsets + K + 1);
                for (size_t h = 0; h < heavy_pid.size(); ++h) {
                    int32_t pid = heavy_pid[h], s = heavy_small[h], l = heavy_large[h];
                    int64_t o0 = exact_profile_offsets[s], o1 = exact_profile_offsets[s + 1];
                    int64_t rows = heavy_rows[h];
                    int64_t c = (o1 - o0 + rows - 1) / rows;
                    if (c < 2)
                        continue;
                    int32_t base = (int32_t)refine_offsets.size();
                    for (int64_t j = 0; j < c; ++j)
                        refine_offsets.push_back((int32_t)(o0 + j * rows));
                    refine_offsets.push_back((int32_t)o1);
                    if (launch_a[pid] == s)
                        launch_a[pid] = base;
                    else
                        launch_b[pid] = base;
                    for (int64_t j = 1; j < c; ++j) {
                        launch_a.push_back(base + (int32_t)j);
                        launch_b.push_back(l);
                        chunk_owner.push_back(pid);
                    }
                }
                for (int32_t l : b_split_profs) {
                    int64_t o0 = exact_profile_offsets[l], o1 = exact_profile_offsets[l + 1];
                    int64_t ll = o1 - o0;
                    int64_t b_c = std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target);
                    int64_t b_rows = (((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63;
                    int64_t c = (ll + b_rows - 1) / b_rows;
                    if (c < 2)
                        continue;
                    int32_t base = (int32_t)refine_offsets.size();
                    for (int64_t j = 0; j < c; ++j)
                        refine_offsets.push_back((int32_t)(o0 + j * b_rows));
                    refine_offsets.push_back((int32_t)o1);
                    for (int32_t pid : b_split_pids[l]) {
                        int32_t s_pid = (launch_a[pid] == l) ? launch_b[pid] : launch_a[pid];
                        launch_a[pid] = s_pid;
                        launch_b[pid] = base;
                    }
                    for (int64_t j = 1; j < c; ++j) {
                        for (int32_t pid : b_split_pids[l]) {
                            launch_a.push_back(launch_a[pid]);
                            launch_b.push_back(base + (int32_t)j);
                            chunk_owner.push_back(pid);
                        }
                    }
                }
            }
        }
        const int32_t n_launch = (int32_t)launch_a.size();
        const bool split_pairs = !refine_offsets.empty();

        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_offsets, split_pairs ? refine_offsets.data() : exact_profile_offsets,
                                   (split_pairs ? refine_offsets.size() : (size_t)(K + 1)) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_leaves, exact_profile_leaves,
                                   (size_t)total_exact_prof * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_weights, exact_profile_weights,
                                   (size_t)total_exact_prof * sizeof(float), cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_a, launch_a.data(), (size_t)n_launch * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_b, launch_b.data(), (size_t)n_launch * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, gpu_ws.stream));

        auto refine = jc69 ? k_refine_real_seq_condensed_D_pairs<true> : k_refine_real_seq_condensed_D_pairs<false>;
        refine<<<n_launch, 256, 0, gpu_ws.stream>>>(
            K, W, n_launch, gpu_ws.d_pair_a, gpu_ws.d_pair_b, gpu_ws.d_exact_offsets, gpu_ws.d_exact_leaves,
            gpu_ws.d_exact_weights, d_packed_seqs, split_pairs ? gpu_ws.d_refine_D_scratch : gpu_ws.d_D,
            gpu_ws.d_pair_exact_D, dm);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> h_launch_exact_D((size_t)n_launch, 0.0);
        CUDA_CHECK(cudaMemcpyAsync(h_launch_exact_D.data(), gpu_ws.d_pair_exact_D, (size_t)n_launch * sizeof(double),
                                   cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
        for (int32_t pid = 0; pid < n_pairs; ++pid)
            h_pair_exact_D[pid] = h_launch_exact_D[pid];
        for (int32_t t = n_pairs; t < n_launch; ++t)
            h_pair_exact_D[chunk_owner[t - n_pairs]] += h_launch_exact_D[t];
    }

    void read_distances(std::vector<double> &D) override {
        std::vector<float> h_D_cond((size_t)K * K);
        CUDA_CHECK(cudaMemcpyAsync(h_D_cond.data(), gpu_ws.d_D, (size_t)K * K * sizeof(float), cudaMemcpyDeviceToHost,
                                   gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
        D.assign(h_D_cond.begin(), h_D_cond.end()); // exact: float to double
    }

  private:
    RnaSimGpuWorkspace &gpu_ws;
    const uint64_t *d_packed_seqs;
    int32_t W;
    DeviceDistance dm;
    bool jc69;
    int32_t K = 0, M_sk = 0, root_u_sk = 0;
};

// ---- The GPU device ----

class GpuDevice final : public Device {
  public:
    GpuDevice(int gpu, int32_t max_k) : id_(gpu) {
        int n_visible = 0;
        if (cudaGetDeviceCount(&n_visible) != cudaSuccess)
            n_visible = 0;
        if (gpu < 0 || gpu >= n_visible)
            throw std::runtime_error("GPU " + std::to_string(gpu) + " does not exist (" + std::to_string(n_visible) +
                                     " visible)");
        try {
            ws_.init(gpu, max_k);
        } catch (...) {
            ws_.destroy();
            throw;
        }
    }

    ~GpuDevice() override {
        cudaSetDevice(id_);
        for (Rows &r : rows_)
            cudaFree(r.d);
        ws_.destroy();
    }

    void load(const uint64_t *packed, int32_t n_taxa, int32_t n_words) override {
        CUDA_CHECK(cudaSetDevice(id_));
        for (Rows &r : rows_)
            cudaFree(r.d);
        rows_.clear(); // if the copy below fails, the device holds no alignment (packed() throws)
        n_words_ = n_words;
        const size_t total_u64 = (size_t)n_taxa * 5 * (size_t)n_words;
        Rows r;
        r.n = n_taxa;
        gpu_alloc(&r.d, total_u64 * sizeof(uint64_t));
        CUDA_CHECK(cudaMemcpy(r.d, packed, total_u64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        rows_.push_back(r);
    }

    void set_distance(const SubstitutionModel &model, const EigenSettings &eigen) override {
        dm_ = distance_params<float>(model, eigen);
    }

    void push_subset(const std::vector<int32_t> &rows) override {
        const int32_t n0 = (int32_t)rows.size();
        const int32_t row_words = 5 * n_words_;
        Rows sub;
        sub.n = n0;
        CUDA_CHECK(cudaSetDevice(id_));
        int32_t *d_idx = NULL;
        gpu_alloc(&d_idx, (size_t)n0 * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpy(d_idx, rows.data(), (size_t)n0 * sizeof(int32_t), cudaMemcpyHostToDevice));
        gpu_alloc(&sub.d, (size_t)n0 * row_words * sizeof(uint64_t));
        k_gather_rows<<<4096, 256>>>(packed(), d_idx, n0, row_words, sub.d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_idx);
        rows_.push_back(sub);
    }

    void pop_subset() override {
        CUDA_CHECK(cudaSetDevice(id_));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(rows_.back().d);
        rows_.pop_back();
    }

    int32_t max_window() const override { return ws_.max_K; }

    void synchronize() override {
        CUDA_CHECK(cudaSetDevice(id_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::string name() const override { return "GPU " + std::to_string(id_) + ": " + gpu_name(id_); }

    double memory_in_use_mb() const override {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaSetDevice(id_));
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        return (double)(total_b - free_b) / 1048576.0;
    }

    void nearest_sampled(const int32_t *queries, int32_t n_queries, const int32_t *samples, int32_t n_samples,
                         int32_t *best_pos) override {
        const int32_t W = n_words_, nq_g = n_queries, n0 = n_samples;
        CUDA_CHECK(cudaSetDevice(id_));
        int32_t *d_q = NULL, *d_s = NULL, *d_best = NULL;
        gpu_alloc(&d_q, (size_t)nq_g * sizeof(int32_t));
        gpu_alloc(&d_s, (size_t)n0 * sizeof(int32_t));
        gpu_alloc(&d_best, (size_t)nq_g * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpy(d_q, queries, (size_t)nq_g * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, samples, (size_t)n0 * sizeof(int32_t), cudaMemcpyHostToDevice));
        int32_t blocks = (nq_g + NN_TQ - 1) / NN_TQ;
        k_nearest_sample<<<blocks, 256>>>(packed(), W, d_q, nq_g, d_s, n0, d_best);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(best_pos, d_best, (size_t)nq_g * sizeof(int32_t), cudaMemcpyDeviceToHost));
        cudaFree(d_q);
        cudaFree(d_s);
        cudaFree(d_best);
    }

    double solve_window(int32_t K, int32_t max_sweeps, int32_t max_batch_cap, double exact_check_eta,
                        const int32_t *profile_offsets, const int32_t *profile_leaves, const float *profile_weights,
                        std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                        const ExactProfileBuilder *exact_builder) override {
        GpuWindowKernels kernels(ws_, packed(), n_words_, dm_);
        return solve_window_host(K, max_sweeps, max_batch_cap, exact_check_eta, profile_offsets, profile_leaves,
                                 profile_weights, v_cf, perm, exact_builder, kernels);
    }

    void edge_lengths(int32_t n_leaves, int32_t cap, int32_t count, const int32_t *c1, const int32_t *c2, int32_t n_int,
                      const int32_t *in_len, const int32_t *in_leaves, const float *in_weights, const int32_t *out_len,
                      const int32_t *out_leaves, const float *out_weights, double *dAB, double *dAO,
                      double *dBO) override {
        const int32_t N = n_leaves, W = n_words_;
        CUDA_CHECK(cudaSetDevice(id_));
        int32_t *d_c1_g = nullptr;
        int32_t *d_c2_g = nullptr;
        int32_t *d_in_len_g = nullptr;
        int32_t *d_in_leaves_g = nullptr;
        float *d_in_weights_g = nullptr;
        int32_t *d_out_len_g = nullptr;
        int32_t *d_out_leaves_g = nullptr;
        float *d_out_weights_g = nullptr;
        float *d_dAB_g = nullptr;
        float *d_dAO_g = nullptr;
        float *d_dBO_g = nullptr;

        gpu_alloc(&d_c1_g, (size_t)count * sizeof(int32_t));
        gpu_alloc(&d_c2_g, (size_t)count * sizeof(int32_t));
        gpu_alloc(&d_in_len_g, (size_t)n_int * sizeof(int32_t));
        gpu_alloc(&d_in_leaves_g, (size_t)n_int * cap * sizeof(int32_t));
        gpu_alloc(&d_in_weights_g, (size_t)n_int * cap * sizeof(float));
        gpu_alloc(&d_out_len_g, (size_t)count * sizeof(int32_t));
        gpu_alloc(&d_out_leaves_g, (size_t)count * cap * sizeof(int32_t));
        gpu_alloc(&d_out_weights_g, (size_t)count * cap * sizeof(float));
        gpu_alloc(&d_dAB_g, (size_t)count * sizeof(float));
        gpu_alloc(&d_dAO_g, (size_t)count * sizeof(float));
        gpu_alloc(&d_dBO_g, (size_t)count * sizeof(float));

        CUDA_CHECK(cudaMemcpy(d_c1_g, c1, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c2_g, c2, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_len_g, in_len, (size_t)n_int * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_leaves_g, in_leaves, (size_t)n_int * cap * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_weights_g, in_weights, (size_t)n_int * cap * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_len_g, out_len, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_out_leaves_g, out_leaves, (size_t)count * cap * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_out_weights_g, out_weights, (size_t)count * cap * sizeof(float), cudaMemcpyHostToDevice));

        int32_t blocks = 3 * count;
        auto kernel = dm_.model == Model::JC69 ? k_eval_internal_node_triples_tiled<true>
                                               : k_eval_internal_node_triples_tiled<false>;
        kernel<<<blocks, 64>>>(count, N, W, cap, d_c1_g, d_c2_g, d_in_len_g, d_in_leaves_g, d_in_weights_g, d_out_len_g,
                               d_out_leaves_g, d_out_weights_g, packed(), d_dAB_g, d_dAO_g, d_dBO_g, dm_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        std::vector<float> hAB(count), hAO(count), hBO(count);
        CUDA_CHECK(cudaMemcpy(hAB.data(), d_dAB_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hAO.data(), d_dAO_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hBO.data(), d_dBO_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(hAB.begin(), hAB.end(), dAB); // float to double, exact
        std::copy(hAO.begin(), hAO.end(), dAO);
        std::copy(hBO.begin(), hBO.end(), dBO);

        cudaFree(d_c1_g);
        cudaFree(d_c2_g);
        cudaFree(d_in_len_g);
        cudaFree(d_in_leaves_g);
        cudaFree(d_in_weights_g);
        cudaFree(d_out_len_g);
        cudaFree(d_out_leaves_g);
        cudaFree(d_out_weights_g);
        cudaFree(d_dAB_g);
        cudaFree(d_dAO_g);
        cudaFree(d_dBO_g);
    }

  private:
    struct Rows {
        uint64_t *d = NULL; // n x 5 x n_words_ bitplanes on the device
        int32_t n = 0;
    };
    const uint64_t *packed() const {
        if (rows_.empty())
            throw std::logic_error("GPU " + std::to_string(id_) + ": no alignment loaded");
        return rows_.back().d;
    }

    int id_;
    DeviceDistance dm_ = distance_params<float>(SubstitutionModel(), EigenSettings()); // JC69 until set_distance
    int32_t n_words_ = 0;
    std::vector<Rows> rows_; // the alignment, then one entry per pushed subsample
    RnaSimGpuWorkspace ws_;
};

} // namespace

std::unique_ptr<Device> make_gpu_device(int gpu, int32_t max_k) {
    return std::unique_ptr<Device>(new GpuDevice(gpu, max_k));
}

int visible_gpus() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

Devices make_devices(const std::vector<int> &gpu_ids) {
    std::vector<int> ids = gpu_ids;
    if (ids.empty()) {
        const int n = visible_gpus();
        if (n < 1)
            throw std::runtime_error("no CUDA device found");
        for (int g = 0; g < n; ++g)
            ids.push_back(g);
    }
    Devices devices;
    for (int g : ids)
        devices.push_back(make_gpu_device(g, WORKSPACE_MAX_K));
    return devices;
}

std::string gpu_build_info() {
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    std::string archs;
#ifdef __CUDA_ARCH_LIST__
#define P2VBME_STR2(x) #x
#define P2VBME_STR(x) P2VBME_STR2(x)
    archs = P2VBME_STR(__CUDA_ARCH_LIST__);
#endif
    char buf[160];
    snprintf(buf, sizeof(buf), "CUDA runtime %d.%d, driver %d.%d, compiled for compute capabilities %s", runtime / 1000,
             (runtime % 1000) / 10, driver / 1000, (driver % 1000) / 10, archs.empty() ? "unknown" : archs.c_str());
    return buf;
}

} // namespace p2vbme
