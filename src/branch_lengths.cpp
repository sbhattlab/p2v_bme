/*
 * branch_lengths.cpp
 *
 * BME branch lengths in O(N): bottom-up inside and top-down outside profiles of every internal node, capped at
 * `cap` leaves with dyadic weights, then the three subtree-pair distances per node on the devices.
 * Total profile-pair evaluations for the whole tree: 3 (N - 2).
 */
#include "branch_lengths.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace p2vbme {

namespace {

struct LeafWeight {
    int32_t leaf;
    float w;
};

inline int32_t merge_profiles_capped(const LeafWeight *__restrict__ p1, int32_t n1, const LeafWeight *__restrict__ p2,
                                     int32_t n2, LeafWeight *__restrict__ out, int32_t cap) {
    int32_t i = 0, j = 0, k = 0;
    double wsum = 0.0;
    while (k < cap && (i < n1 || j < n2)) {
        bool take1 = (i < n1 && j < n2) ? (p1[i].w >= p2[j].w) : (i < n1);
        if (take1) {
            float nw = 0.5f * p1[i].w;
            out[k++] = {p1[i].leaf, nw};
            wsum += nw;
            ++i;
        } else {
            float nw = 0.5f * p2[j].w;
            out[k++] = {p2[j].leaf, nw};
            wsum += nw;
            ++j;
        }
    }
    if (k > 0 && wsum > 0.0) {
        float inv = (float)(1.0 / wsum);
        for (int32_t m = 0; m < k; ++m)
            out[m].w *= inv;
    }
    return k;
}

// Step 4 of compute_bme_branch_lengths in precision T (float: the GPU's single precision).
template <class T>
double assemble_lengths(int32_t N, const int32_t *h_c1, const int32_t *h_c2, const std::vector<int32_t> &parent,
                        int32_t r_left, int32_t r_right, const std::vector<double> &h_dAB,
                        const std::vector<double> &h_dAO, const std::vector<double> &h_dBO,
                        std::vector<double> &h_branch_lengths) {
    double total_bme_len = 0.0;
    for (int32_t u = 0; u < 2 * N - 2; ++u) {
        if (u == r_right) {
            h_branch_lengths[u] = 0.0;
            continue;
        }
        if (u == r_left) {
            T cross_l = (r_left >= N) ? T(0.5) * (T(h_dAO[r_left - N]) + T(h_dBO[r_left - N]))
                                      : T(0.5) * (T(h_dAO[r_right - N]) + T(h_dBO[r_right - N]));
            T cross_r = (r_right >= N) ? T(0.5) * (T(h_dAO[r_right - N]) + T(h_dBO[r_right - N])) : cross_l;
            T div_l = (r_left >= N) ? T(h_dAB[r_left - N]) : T(0);
            T div_r = (r_right >= N) ? T(h_dAB[r_right - N]) : T(0);
            T bl = T(0.5) * (cross_l + cross_r) - T(0.5) * (div_l + div_r);
            h_branch_lengths[u] = bl;
            total_bme_len += bl;
            continue;
        }

        int32_t p = parent[u];
        int32_t p_idx = p - N;
        bool is_left = (h_c1[p] == u);
        T d_ab_p = T(h_dAB[p_idx]);
        T d_child_out_p = T(is_left ? h_dAO[p_idx] : h_dBO[p_idx]);
        T d_sib_out_p = T(is_left ? h_dBO[p_idx] : h_dAO[p_idx]);

        T bl;
        if (u < N) {
            bl = T(0.5) * (d_ab_p + d_child_out_p - d_sib_out_p);
        } else {
            int32_t u_idx = u - N;
            T cross_from_p = T(0.5) * (d_ab_p + d_child_out_p);
            T cross_from_u = T(0.5) * (T(h_dAO[u_idx]) + T(h_dBO[u_idx]));
            bl = T(0.5) * (cross_from_p + cross_from_u) - T(0.5) * (d_sib_out_p + T(h_dAB[u_idx]));
        }
        h_branch_lengths[u] = bl;
        total_bme_len += bl;
    }

    return total_bme_len;
}

} // namespace

double compute_bme_branch_lengths(const Tree &tree, Devices &devices, std::vector<double> &h_branch_lengths,
                                  int32_t cap, double *out_profile_sec, double *out_kernel_sec, bool double_precision) {
    const int32_t N = tree.n_leaves();
    const int32_t *h_c1 = tree.c1.data(), *h_c2 = tree.c2.data();
    const int32_t n_gpus = (int32_t)devices.size();
    const double t0 = omp_get_wtime();
    const int32_t root = 2 * N - 2;
    const int32_t n_int = N - 2; // internal nodes N .. 2N-3
    h_branch_lengths.assign(2 * N - 2, 0.0);
    if (N <= 2)
        return 0.0;

    // 1. Build post-order traversal of internal nodes N .. 2N-2 and parent pointers
    std::vector<int32_t> parent(2 * N - 1, -1);
    std::vector<int32_t> postorder;
    postorder.reserve(N - 1);
    {
        std::vector<int32_t> stack;
        stack.reserve(2 * N - 1);
        stack.push_back(root);
        while (!stack.empty()) {
            int32_t u = stack.back();
            stack.pop_back();
            if (u >= N) {
                postorder.push_back(u);
                int32_t l = h_c1[u], r = h_c2[u];
                parent[l] = u;
                parent[r] = u;
                stack.push_back(l);
                stack.push_back(r);
            }
        }
        std::reverse(postorder.begin(), postorder.end());
    }

    // 2. Allocate flat inside and outside profiles for the N - 2 internal nodes N .. 2N-3
    std::vector<int32_t> in_len(n_int, 0), out_len(n_int, 0);
    std::vector<int32_t> in_leaves((size_t)n_int * cap, 0), out_leaves((size_t)n_int * cap, 0);
    std::vector<float> in_weights((size_t)n_int * cap, 0.0f), out_weights((size_t)n_int * cap, 0.0f);

    std::vector<LeafWeight> tmp_a(cap), tmp_b(cap), tmp_out(cap);

    auto get_in_profile = [&](int32_t u, LeafWeight *buf) -> int32_t {
        if (u < N) {
            buf[0] = {u, 1.0f};
            return 1;
        }
        int32_t idx = u - N;
        int32_t l = in_len[idx];
        const int32_t *lv = &in_leaves[(size_t)idx * cap];
        const float *wv = &in_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < l; ++k)
            buf[k] = {lv[k], wv[k]};
        return l;
    };

    for (int32_t u : postorder) {
        if (u == root)
            continue;
        int32_t idx = u - N;
        int32_t na = get_in_profile(h_c1[u], tmp_a.data());
        int32_t nb = get_in_profile(h_c2[u], tmp_b.data());
        int32_t nm = merge_profiles_capped(tmp_a.data(), na, tmp_b.data(), nb, tmp_out.data(), cap);
        in_len[idx] = nm;
        int32_t *lv = &in_leaves[(size_t)idx * cap];
        float *wv = &in_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < nm; ++k) {
            lv[k] = tmp_out[k].leaf;
            wv[k] = tmp_out[k].w;
        }
    }

    const int32_t r_left = h_c1[root];
    const int32_t r_right = h_c2[root];
    auto set_out_profile = [&](int32_t u, const LeafWeight *buf, int32_t l) {
        if (u < N)
            return;
        int32_t idx = u - N;
        out_len[idx] = l;
        int32_t *lv = &out_leaves[(size_t)idx * cap];
        float *wv = &out_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < l; ++k) {
            lv[k] = buf[k].leaf;
            wv[k] = buf[k].w;
        }
    };

    {
        int32_t nr = get_in_profile(r_right, tmp_a.data());
        set_out_profile(r_left, tmp_a.data(), nr);
        int32_t nl = get_in_profile(r_left, tmp_b.data());
        set_out_profile(r_right, tmp_b.data(), nl);
    }

    for (int32_t pi = (int32_t)postorder.size() - 1; pi >= 0; --pi) {
        int32_t u = postorder[pi];
        if (u == root)
            continue;
        int32_t idx = u - N;
        int32_t no = out_len[idx];
        for (int32_t k = 0; k < no; ++k) {
            tmp_a[k] = {out_leaves[(size_t)idx * cap + k], out_weights[(size_t)idx * cap + k]};
        }
        int32_t c_left = h_c1[u];
        int32_t c_right = h_c2[u];
        if (c_left >= N) {
            int32_t nb = get_in_profile(c_right, tmp_b.data());
            int32_t nm = merge_profiles_capped(tmp_a.data(), no, tmp_b.data(), nb, tmp_out.data(), cap);
            set_out_profile(c_left, tmp_out.data(), nm);
        }
        if (c_right >= N) {
            int32_t na = get_in_profile(c_left, tmp_b.data());
            int32_t nm = merge_profiles_capped(tmp_a.data(), no, tmp_b.data(), na, tmp_out.data(), cap);
            set_out_profile(c_right, tmp_out.data(), nm);
        }
    }

    const double t1 = omp_get_wtime();
    if (out_profile_sec)
        *out_profile_sec = t1 - t0;

    // 3. Edge-length pass on the devices (tiled evaluation of d_AB(p), d_AO(p), d_BO(p) for all p in 0..n_int-1
    std::vector<double> h_dAB(n_int, 0.0), h_dAO(n_int, 0.0), h_dBO(n_int, 0.0);
    int32_t active_gpus = std::max(1, std::min(n_gpus, n_int));

    ParallelErrors errors;
    // one work item per device, handed out by iteration (not by thread number, so every chunk is computed even if
    // OpenMP gives fewer threads, e.g. when called from inside a parallel region)
#pragma omp parallel for num_threads(active_gpus) schedule(static, 1)
    for (int g = 0; g < active_gpus; ++g) {

        int32_t chunk = (n_int + active_gpus - 1) / active_gpus;
        int32_t start = g * chunk;
        int32_t end = std::min(n_int, start + chunk);
        int32_t count = std::max(0, end - start);

        if (count > 0) {
            try {
                devices[g]->edge_lengths(N, cap, count, h_c1 + N + start, h_c2 + N + start, n_int, in_len.data(),
                                         in_leaves.data(), in_weights.data(), out_len.data() + start,
                                         out_leaves.data() + (size_t)start * cap,
                                         out_weights.data() + (size_t)start * cap, h_dAB.data() + start,
                                         h_dAO.data() + start, h_dBO.data() + start);
            } catch (...) {
                errors.capture();
            }
        }
    }
    errors.rethrow();

    // 4. Assemble all 2N - 3 unrooted FastME/BME branch lengths in O(N), in the run's precision
    const double total_bme_len =
        double_precision
            ? assemble_lengths<double>(N, h_c1, h_c2, parent, r_left, r_right, h_dAB, h_dAO, h_dBO, h_branch_lengths)
            : assemble_lengths<float>(N, h_c1, h_c2, parent, r_left, r_right, h_dAB, h_dAO, h_dBO, h_branch_lengths);

    const double t2 = omp_get_wtime();
    if (out_kernel_sec)
        *out_kernel_sec = t2 - t1;
    return total_bme_len;
}

} // namespace p2vbme
