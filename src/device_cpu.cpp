/*
 * device_cpu.cpp
 *
 * The CPU device: the device steps of device_gpu.cu on the host, so that
 * --accelerator cpu runs the same algorithm as the GPU. Single precision by default (the GPU's), double with
 * --precision double. One CpuDevice per worker (--devices N: N workers); a step uses the run's threads when it
 * runs alone and one thread when several devices work in parallel (inside the search's parallel region).
 *
 * The window solve's sweeps are search.cpp's solve_window; here are its kernels, ported
 * from the GPU kernels value for value (up to floating-point contraction and summation order): the window
 * distances (k_build_real_seq_condensed_D's summation order), the subtree averages and BME length of a
 * skeleton (k_copy_leaf_rows_perm ... k_compute_bme_length), each row's best move (k_spr_rows_top1: the seeds
 * of dev_seed_half_tree_fifo, then dev_expand_subtree_stack's step on each internal node, depth-first here).
 * Without CUDA this file also provides the GPU entry points of device.hpp, which then report that there is no GPU.
 */
#include <omp.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "eigen_dist.hpp"
#include "search.hpp"
#include "substitution_models.hpp"

namespace p2vbme {

namespace {

// The model's distance between two rows of the 5-bitplane layout (JC69 from its two counts, as the GPU).
template <class T> T row_distance(const uint64_t *a, const uint64_t *b, int32_t W, const DistanceParams<T> &dm) {
    if (dm.model == Model::JC69) {
        int32_t n_valid = 0, n_match = 0;
        for (int32_t w = 0; w < W; ++w) {
            n_valid += __builtin_popcountll(a[4 * W + w] & b[4 * W + w]);
            n_match += __builtin_popcountll((a[w] & b[w]) | (a[W + w] & b[W + w]) | (a[2 * W + w] & b[2 * W + w]) |
                                            (a[3 * W + w] & b[3 * W + w]));
        }
        return jc69_distance<T>(n_valid, n_match);
    }
    const PairCounts c = count_pair(a, b, W);
    return pair_distance<T>(dm, c.n);
}

// Threads for a step of `items` independent items: at most `threads`, and at least `per_thread` items each (a small
// window does not pay for a large team).
inline int threads_for(int threads, int64_t items, int64_t per_thread) {
    return (int)std::max<int64_t>(1, std::min<int64_t>(threads, items / per_thread));
}

// The start tree's subsamples, shared by the CPU devices of one run: each device pushes the same rows, and the first
// one gathers them (a GPU copies them to each GPU; the CPU workers share one host copy).
struct SubsetStore {
    std::mutex mutex;
    std::vector<std::pair<std::vector<int32_t>, std::weak_ptr<const std::vector<uint64_t>>>> levels;
};

// What one CPU device keeps between windows: the window's distances and the skeleton's subtree averages.
template <class T> struct CpuWorkspace {
    std::vector<T> D, subdist, nd_u;
};

template <class T> class CpuWindowKernels final : public WindowKernels {
  public:
    CpuWindowKernels(const uint64_t *rows, int32_t W, const DistanceParams<T> &dm, int threads, CpuWorkspace<T> &ws)
        : rows_(rows), W_(W), dm_(dm), threads_(threads), ws_(ws) {}
    // with a workspace of its own (make_cpu_window_kernels)
    CpuWindowKernels(const uint64_t *rows, int32_t W, const DistanceParams<T> &dm, int threads)
        : rows_(rows), W_(W), dm_(dm), threads_(threads), owned_(new CpuWorkspace<T>()), ws_(*owned_) {}

    // K x K profile-weighted distances, summed as on the GPU: row sums over q, then over p.
    void distances(int32_t K, const int32_t *off, const int32_t *leaves, const float *weights) override {
        K_ = K;
        M_ = 2 * K - 2;
        std::vector<T> &D = ws_.D;
        D.assign((size_t)K * K, T(0));
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads_for(threads_, K, 16))
        for (int32_t a = 0; a < K; ++a)
            for (int32_t b = a + 1; b < K; ++b) {
                T d_val = T(0);
                for (int32_t p = off[a]; p < off[a + 1]; ++p) {
                    T row_sum = T(0);
                    for (int32_t q = off[b]; q < off[b + 1]; ++q)
                        row_sum += (T)weights[q] * leaf_distance(leaves[p], leaves[q]);
                    d_val += (T)weights[p] * row_sum;
                }
                D[(size_t)a * K + b] = d_val;
                D[(size_t)b * K + a] = d_val;
            }
    }

    double score(const SkeletonMeta &s, const std::vector<int32_t> &perm) override {
        const int32_t K = K_, M = M_, root_u = M - 1;
        ws_.subdist.resize((size_t)M * M);
        T *sd = ws_.subdist.data();
        const T *D = ws_.D.data();
        // bottom up: internal nodes to leaves, then (after the transpose) to internal nodes; then top down, where node
        // v reads and writes entries no other node touches (as one GPU thread per v)
#pragma omp parallel num_threads(threads_for(threads_, M, 64))
        {
#pragma omp for
            for (int32_t row = 0; row < K; ++row)
                for (int32_t col = 0; col < K; ++col)
                    sd[(size_t)row * M + col] = D[(size_t)perm[row] * K + perm[col]];
#pragma omp for
            for (int32_t v = 0; v < K; ++v)
                for (int32_t u = K; u < root_u; ++u)
                    sd[(size_t)u * M + v] = T(0.5) * (sd[(size_t)s.c1[u] * M + v] + sd[(size_t)s.c2[u] * M + v]);
#pragma omp for
            for (int32_t u = K; u < root_u; ++u)
                for (int32_t v = 0; v < K; ++v)
                    sd[(size_t)v * M + u] = sd[(size_t)u * M + v];
#pragma omp for
            for (int32_t v = K; v < root_u; ++v)
                for (int32_t u = K; u < root_u; ++u)
                    sd[(size_t)u * M + v] = T(0.5) * (sd[(size_t)s.c1[u] * M + v] + sd[(size_t)s.c2[u] * M + v]);
#pragma omp for
            for (int32_t v = 0; v < M; ++v)
                top_down(s, v);
        }
        double acc = 0.0;
        for (int32_t x = 0; x < K; ++x)
            acc += (double)sd[(size_t)x * M + s.parent_u[x]];
        // for best_moves (k_populate_nd_u): per internal node, the average distance between its other neighbours
        ws_.nd_u.resize((size_t)3 * M);
        T *nd = ws_.nd_u.data();
        for (int32_t u = K; u <= root_u; ++u) {
            const int32_t n0 = s.adj_u[3 * u], n1 = s.adj_u[3 * u + 1], n2 = s.adj_u[3 * u + 2];
            nd[3 * u + 0] = sd[(size_t)n1 * M + n2];
            nd[3 * u + 1] = sd[(size_t)n0 * M + n2];
            nd[3 * u + 2] = sd[(size_t)n0 * M + n1];
        }
        return 0.5 * acc;
    }

    void best_moves(const SkeletonMeta &s, std::vector<double> &best_delta, std::vector<int32_t> &best_j) override {
        const int32_t K = K_, M = M_, root_u = M - 1;
        best_delta.assign(K, 1e30);
        best_j.assign(K, -1);
        const T *sd = ws_.subdist.data();
        // per-thread buffers, allocated before the parallel region (an exception must not leave it); the stack never
        // holds more than the M nodes
        const int nt = threads_for(threads_, K, 8);
        std::vector<std::vector<T>> edges(nt, std::vector<T>((size_t)M));
        std::vector<std::vector<Frame>> stacks(nt);
        for (auto &st : stacks)
            st.reserve((size_t)M);
#pragma omp parallel num_threads(nt)
        {
            std::vector<T> &edge = edges[omp_get_thread_num()];
            std::vector<Frame> &stack = stacks[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 8)
            for (int32_t i = 1; i < K - 1; ++i) {
                const int32_t tree_root = s.v_map[i + 1];
                const int32_t parent = (tree_root == root_u) ? s.other_rc : s.parent_u[tree_root];
                int32_t other0 = -1, other1 = -1;
                for (int k = 0; k < 3; ++k) {
                    const int32_t nb = s.adj_u[3 * parent + k];
                    if (nb != -1 && nb != tree_root) {
                        if (other0 == -1)
                            other0 = nb;
                        else
                            other1 = nb;
                    }
                }
                if (other0 == -1)
                    continue;
                // every edge starts at "no move"; the edges at the pruning point stay so
                std::fill(edge.begin(), edge.end(), T(1e30));
                const T *row_root = sd + (size_t)tree_root * M;
                seed(s, other0, other1, parent, row_root, sd + (size_t)other1 * M, edge.data(), stack);
                seed(s, other1, other0, parent, row_root, sd + (size_t)other0 * M, edge.data(), stack);
                // the smallest change; ties to the smallest edge (the GPU reduction's rule)
                T b_val = T(1e30);
                int32_t b_j = -1;
                for (int32_t du = 0; du < root_u; ++du)
                    if (edge[du] < b_val) {
                        b_val = edge[du];
                        b_j = du;
                    }
                best_delta[i] = (double)b_val;
                best_j[i] = b_j;
            }
        }
    }

  private:
    struct Frame {
        int32_t u, c;
        T pow, dkc, dist;
    };

    T leaf_distance(int32_t x, int32_t y) const {
        if (x == y)
            return T(0);
        return row_distance<T>(rows_ + (size_t)x * 5 * W_, rows_ + (size_t)y * 5 * W_, W_, dm_);
    }

    void top_down(const SkeletonMeta &s, int32_t v) {
        const int32_t M = M_, root_u = M - 1;
        T *sd = ws_.subdist.data();
        if (v == root_u) {
            sd[(size_t)root_u * M + root_u] = T(0);
            return;
        }
        const int32_t tv = s.tin[v];
        int32_t curr;
        T root_val;
        if (tv < s.tin[s.rc1]) {
            curr = s.rc0;
            root_val = T(0.5) * (sd[(size_t)s.rc1 * M + v] + sd[(size_t)s.rc2 * M + v]);
        } else if (tv < s.tin[s.rc2]) {
            curr = s.rc1;
            root_val = T(0.5) * (sd[(size_t)s.rc0 * M + v] + sd[(size_t)s.rc2 * M + v]);
        } else {
            curr = s.rc2;
            root_val = T(0.5) * (sd[(size_t)s.rc0 * M + v] + sd[(size_t)s.rc1 * M + v]);
        }
        sd[(size_t)root_u * M + v] = root_val;
        sd[(size_t)v * M + root_u] = root_val;
        int32_t g = root_u;
        while (curr != v) {
            const int32_t c1 = s.c1[curr], c2 = s.c2[curr];
            const int32_t next_curr = (tv < s.tin[c2]) ? c1 : c2;
            const int32_t sib = (tv < s.tin[c2]) ? c2 : c1;
            const T val = T(0.5) * (sd[(size_t)sib * M + v] + sd[(size_t)g * M + v]);
            sd[(size_t)curr * M + v] = val;
            sd[(size_t)v * M + curr] = val;
            g = curr;
            curr = next_curr;
        }
        sd[(size_t)v * M + v] = T(0);
    }

    // The score changes of the regraft edges in the half tree beyond `start` (a neighbour of the pruning point
    // `parent`; `orig` is its other neighbour), for the subtree whose row of averages is row_root.
    void seed(const SkeletonMeta &s, int32_t start, int32_t orig, int32_t parent, const T *row_root, const T *row_orig,
              T *edge, std::vector<Frame> &stack) const {
        const int32_t K = K_;
        if (start < K)
            return;
        const T *nd = ws_.nd_u.data();
        const int32_t n0 = s.adj_u[3 * start + 0], n1 = s.adj_u[3 * start + 1];
        const int32_t k_in = (n0 == parent) ? 0 : ((n1 == parent) ? 1 : 2);
        const int32_t k0 = (k_in == 0) ? 1 : 0, k1 = (k_in == 2) ? 1 : 2;
        const int32_t B0 = s.adj_u[3 * start + k0], B1 = s.adj_u[3 * start + k1];
        const T D_AB = nd[3 * start + k_in];
        const T d_kc_init = row_root[orig];
        const T nd0 = T(0.25) * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
        edge[(s.parent_u[start] == B0) ? start : B0] = nd0;
        const T nd1 = T(0.25) * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
        edge[(s.parent_u[start] == B1) ? start : B1] = nd1;
        stack.clear();
        if (B0 >= K)
            stack.push_back({B0, start, T(0.25), T(0.5) * (d_kc_init + row_root[B1]), nd0});
        if (B1 >= K)
            stack.push_back({B1, start, T(0.25), T(0.5) * (d_kc_init + row_root[B0]), nd1});
        while (!stack.empty()) {
            const Frame f = stack.back();
            stack.pop_back();
            const int32_t u = f.u;
            const int32_t a0 = s.adj_u[3 * u + 0], a1 = s.adj_u[3 * u + 1];
            const int32_t ki = (a0 == f.c) ? 0 : ((a1 == f.c) ? 1 : 2);
            const int32_t idx0 = (ki == 0) ? 1 : 0, idx1 = (ki == 2) ? 1 : 2;
            const int32_t b0 = s.adj_u[3 * u + idx0], b1 = s.adj_u[3 * u + idx1];
            const T d_ab = nd[3 * u + ki], r_b0 = row_root[b0], r_b1 = row_root[b1];
            const T d_ac0 = nd[3 * u + idx0] + (row_orig[b1] - r_b1) * f.pow;
            const T ndist0 = f.dist + T(0.25) * (d_ac0 + r_b0 - d_ab - f.dkc);
            edge[(s.parent_u[u] == b0) ? u : b0] = ndist0;
            const T d_ac1 = nd[3 * u + idx1] + (row_orig[b0] - r_b0) * f.pow;
            const T ndist1 = f.dist + T(0.25) * (d_ac1 + r_b1 - d_ab - f.dkc);
            edge[(s.parent_u[u] == b1) ? u : b1] = ndist1;
            const T next_pow = f.pow * T(0.5);
            if (b0 >= K)
                stack.push_back({b0, u, next_pow, T(0.5) * (f.dkc + r_b1), ndist0});
            if (b1 >= K)
                stack.push_back({b1, u, next_pow, T(0.5) * (f.dkc + r_b0), ndist1});
        }
    }

    const uint64_t *rows_;
    int32_t W_;
    DistanceParams<T> dm_;
    int threads_;
    std::unique_ptr<CpuWorkspace<T>> owned_;
    CpuWorkspace<T> &ws_;
    int32_t K_ = 0, M_ = 0;
};

template <class T> class CpuDevice final : public Device {
  public:
    CpuDevice(int32_t worker, int32_t threads, std::shared_ptr<SubsetStore> store)
        : worker_(worker), threads_(std::max(1, threads)), store_(std::move(store)) {}

    // The CPU device reads the caller's alignment in place (no copy): it must stay alive while the device uses it.
    void load(const uint64_t *packed, int32_t, int32_t n_words) override {
        subsets_.clear();
        base_ = packed;
        n_words_ = n_words;
    }

    void set_distance(const SubstitutionModel &model, const EigenSettings &eigen) override {
        dm_ = distance_params<T>(model, eigen);
    }

    // The rows of the current alignment (k_gather_rows), shared with the run's other CPU devices at this depth.
    void push_subset(const std::vector<int32_t> &rows) override {
        const size_t depth = subsets_.size();
        std::shared_ptr<const std::vector<uint64_t>> sub;
        {
            std::lock_guard<std::mutex> lock(store_->mutex);
            if (store_->levels.size() > depth && store_->levels[depth].first == rows)
                sub = store_->levels[depth].second.lock();
        }
        if (!sub) {
            const size_t row_words = (size_t)5 * n_words_;
            const uint64_t *src = packed();
            auto gathered = std::make_shared<std::vector<uint64_t>>(rows.size() * row_words);
            for (size_t r = 0; r < rows.size(); ++r)
                memcpy(&(*gathered)[r * row_words], src + (size_t)rows[r] * row_words, row_words * sizeof(uint64_t));
            sub = gathered;
            std::lock_guard<std::mutex> lock(store_->mutex);
            store_->levels.resize(depth + 1);
            store_->levels[depth] = std::make_pair(rows, std::weak_ptr<const std::vector<uint64_t>>(sub));
        }
        subsets_.push_back(std::move(sub));
    }

    void pop_subset() override { subsets_.pop_back(); }

    int32_t max_window() const override { return WORKSPACE_MAX_K; }

    void synchronize() override {}

    std::string name() const override {
        return "CPU worker " + std::to_string(worker_) + " (" + (sizeof(T) == 4 ? "single" : "double") + " precision)";
    }

    double memory_in_use_mb() const override { return 0.0; }

    // The nearest sample of each query by p-distance (as k_nearest_sample: the same float p, ties to the lower
    // position).
    void nearest_sampled(const int32_t *queries, int32_t n_queries, const int32_t *samples, int32_t n_samples,
                         int32_t *best_pos) override {
        const uint64_t *seqs = packed();
        const int32_t W = n_words_;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads_for(step_threads(), n_queries, 256))
        for (int32_t qi = 0; qi < n_queries; ++qi) {
            const uint64_t *q = seqs + (size_t)queries[qi] * 5 * W;
            uint64_t best_key = ~0ULL;
            for (int32_t si = 0; si < n_samples; ++si) {
                const uint64_t *sp = seqs + (size_t)samples[si] * 5 * W;
                int32_t valid = 0, match = 0;
                for (int32_t w = 0; w < W; ++w) {
                    valid += __builtin_popcountll(q[4 * W + w] & sp[4 * W + w]);
                    match += __builtin_popcountll((q[w] & sp[w]) | (q[W + w] & sp[W + w]) |
                                                  (q[2 * W + w] & sp[2 * W + w]) | (q[3 * W + w] & sp[3 * W + w]));
                }
                const float p = valid > 0 ? (float)(valid - match) / (float)valid : NN_P_NO_OVERLAP;
                uint32_t bits;
                memcpy(&bits, &p, sizeof(bits));
                const uint64_t key = ((uint64_t)bits << 32) | (uint32_t)si;
                if (key < best_key)
                    best_key = key;
            }
            best_pos[qi] = best_key == ~0ULL ? 0 : (int32_t)(best_key & 0xffffffffULL);
        }
    }

    std::unique_ptr<WindowKernels> window_kernels() override {
        return std::make_unique<CpuWindowKernels<T>>(packed(), n_words_, dm_, step_threads(), ws_);
    }

    // Per internal node and kind (0: the two child subtrees, 1 and 2: each child and the rest of the tree), the
    // profile-weighted distance, as k_eval_internal_node_triples_tiled.
    void edge_lengths(int32_t n_leaves, int32_t cap, int32_t count, const int32_t *c1, const int32_t *c2, int32_t,
                      const int32_t *in_len, const int32_t *in_leaves, const float *in_weights, const int32_t *out_len,
                      const int32_t *out_leaves, const float *out_weights, double *dAB, double *dAO,
                      double *dBO) override {
        const int32_t N = n_leaves, W = n_words_;
        const uint64_t *rows = packed();
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads_for(step_threads(), 3 * (int64_t)count, 64))
        for (int32_t job = 0; job < 3 * count; ++job) {
            const int32_t i = job / 3, kind = job % 3;
            // one side: a single leaf or a node's inside profile; the other: the sibling's or the outside profile
            const int32_t node1 = (kind == 2) ? c2[i] : c1[i];
            int32_t len1, len2, leaf1 = -1, leaf2 = -1;
            const int32_t *l1 = nullptr, *l2 = nullptr;
            const float *w1 = nullptr, *w2 = nullptr;
            if (node1 < N) {
                len1 = 1;
                leaf1 = node1;
            } else {
                len1 = in_len[node1 - N];
                l1 = in_leaves + (size_t)(node1 - N) * cap;
                w1 = in_weights + (size_t)(node1 - N) * cap;
            }
            if (kind == 0) {
                if (c2[i] < N) {
                    len2 = 1;
                    leaf2 = c2[i];
                } else {
                    len2 = in_len[c2[i] - N];
                    l2 = in_leaves + (size_t)(c2[i] - N) * cap;
                    w2 = in_weights + (size_t)(c2[i] - N) * cap;
                }
            } else {
                len2 = out_len[i];
                l2 = out_leaves + (size_t)i * cap;
                w2 = out_weights + (size_t)i * cap;
            }
            T sum = T(0);
            for (int32_t p = 0; p < len1; ++p) {
                const int32_t lu = leaf1 >= 0 ? leaf1 : l1[p];
                const T wu = leaf1 >= 0 ? T(1) : (T)w1[p];
                for (int32_t q = 0; q < len2; ++q) {
                    const int32_t lv = leaf2 >= 0 ? leaf2 : l2[q];
                    const T wv = leaf2 >= 0 ? T(1) : (T)w2[q];
                    const T d =
                        lu == lv ? T(0) : row_distance<T>(rows + (size_t)lu * 5 * W, rows + (size_t)lv * 5 * W, W, dm_);
                    sum += wu * wv * d;
                }
            }
            (kind == 0 ? dAB : kind == 1 ? dAO : dBO)[i] = (double)sum;
        }
    }

  private:
    // The run's threads when this step runs alone, one when several devices work in parallel.
    int step_threads() const { return omp_in_parallel() ? 1 : threads_; }

    const uint64_t *packed() const {
        if (!base_)
            throw std::logic_error("CPU worker " + std::to_string(worker_) + ": no alignment loaded");
        return subsets_.empty() ? base_ : subsets_.back()->data();
    }

    int32_t worker_;
    int threads_;
    DistanceParams<T> dm_ = distance_params<T>(SubstitutionModel(), EigenSettings()); // JC69 until set_distance
    const uint64_t *base_ = nullptr;
    int32_t n_words_ = 0;
    std::vector<std::shared_ptr<const std::vector<uint64_t>>> subsets_; // start-tree subsamples, innermost last
    std::shared_ptr<SubsetStore> store_;
    CpuWorkspace<T> ws_;
};

} // namespace

namespace {
std::unique_ptr<Device> cpu_device(int32_t worker, int32_t threads, bool double_precision,
                                   std::shared_ptr<SubsetStore> store) {
    if (double_precision)
        return std::unique_ptr<Device>(new CpuDevice<double>(worker, threads, std::move(store)));
    return std::unique_ptr<Device>(new CpuDevice<float>(worker, threads, std::move(store)));
}
} // namespace

std::unique_ptr<Device> make_cpu_device(int32_t worker, int32_t threads, bool double_precision) {
    return cpu_device(worker, threads, double_precision, std::make_shared<SubsetStore>());
}

std::unique_ptr<WindowKernels> make_cpu_window_kernels(const uint64_t *rows, int32_t n_words,
                                                       const SubstitutionModel &model, const EigenSettings &eigen,
                                                       bool double_precision) {
    if (double_precision)
        return std::unique_ptr<WindowKernels>(
            new CpuWindowKernels<double>(rows, n_words, distance_params<double>(model, eigen), 1));
    return std::unique_ptr<WindowKernels>(
        new CpuWindowKernels<float>(rows, n_words, distance_params<float>(model, eigen), 1));
}

Devices make_cpu_devices(int32_t n_workers, bool double_precision) {
    if (n_workers < 1)
        throw std::runtime_error("the CPU needs at least one thread");
    Devices devices;
    auto store = std::make_shared<SubsetStore>(); // one copy of each subsample for all workers
    for (int32_t w = 0; w < n_workers; ++w)
        devices.push_back(cpu_device(w, n_workers, double_precision, store));
    return devices;
}

#ifndef P2VBME_WITH_CUDA
// A build without CUDA has no GPU.
int visible_gpus() { return 0; }

Devices make_devices(const std::vector<int> &) { throw std::runtime_error("this p2v-bme was built without CUDA"); }

std::unique_ptr<Device> make_gpu_device(int, int32_t) {
    throw std::runtime_error("this p2v-bme was built without CUDA");
}

std::string gpu_build_info() { return "built without CUDA"; }
#endif

} // namespace p2vbme
