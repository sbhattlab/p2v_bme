/*
 * branch_lengths.cpp
 *
 * BME branch lengths in O(N): bottom-up inside and top-down outside profiles of every internal node, capped at
 * `cap` leaves with dyadic weights, then the three subtree-pair distances per node on the devices.
 * Total profile-pair evaluations for the whole tree: 3 (N - 2).
 *
 * A profile stands for a subtree in distance sums: some of its leaves, with weights that sum to 1.
 * The distance between two profiles is d(P, Q) = sum over leaves x of P and y of Q of P(x) Q(y) D(x, y).
 * In the unrooted tree (the root's two children joined by one edge), each node u but the root has two:
 *   - in(u), inside: the leaves below u, leaf x weighted 2^-(edges from u to x); a leaf's is itself, with weight 1.
 *   - out(u), outside: all other leaves, leaf x weighted 2^-(edges from v to x), with v the neighbour of u towards the
 *     root (its parent, or its sibling when the parent is the root).
 * They are search.cpp's clade and outgroup profiles of u, capped differently (merge_profiles_capped).
 * Each comes from two profiles one edge further away: in(u) from its children's, out(u) from its parent's.
 * The length of an edge combines the distances between the four subtrees around it (assemble_lengths).
 */
#include "branch_lengths.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "parallel_errors.hpp"

namespace p2vbme {

namespace {

// One entry of a profile.
struct LeafWeight {
    int32_t leaf;
    float w;
};

// Merges two subtrees' profiles into the profile of both, seen from the node they hang from:
// (p1, p2) -> out = (p1 + p2) / 2, at most cap leaves.
// Every weight is halved: each subtree is one edge further away.
// For in(u) = (in(c1) + in(c2)) / 2 and out(c1) = (out(u) + in(c2)) / 2: p1 and p2 never share a leaf.
// Profiles are sorted by decreasing weight: from two sorted inputs, the merge keeps the cap heaviest leaves (p1's on
// ties), sorted.
// If it drops leaves, the kept weights are rescaled to sum to 1: each kept leaf gets a share of the dropped weight in
// proportion to its own.
// Returns the number of leaves in out.
inline int32_t merge_profiles_capped(const LeafWeight *__restrict__ p1, int32_t n1, const LeafWeight *__restrict__ p2,
                                     int32_t n2, LeafWeight *__restrict__ out, int32_t cap) {
    int32_t i = 0, j = 0, k = 0; // next entry of p1, p2 and out
    double wsum = 0.0;           // weight kept in out
    while (k < cap && (i < n1 || j < n2)) {
        // the heavier of the next entries of p1 and p2, or the one left when the other input is used up
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
    // Rescale to sum 1 (no change, up to rounding, unless leaves were dropped).
    if (k > 0 && wsum > 0.0) {
        float inv = (float)(1.0 / wsum);
        for (int32_t m = 0; m < k; ++m)
            out[m].w *= inv;
    }
    return k;
}

// Combines the internal nodes' distances into the lengths of all edges: (dAB, dAO, dBO) -> h_branch_lengths.
// Returns their sum.
// Step 4 of compute_bme_branch_lengths in precision T (float: the GPU's single precision).
// Row v - N of h_dAB, h_dAO, h_dBO holds, for internal node v with children A = c1, B = c2 and the rest O, the
// distances d(A, B), d(A, O) and d(B, O) (two subtrees' distance is their profiles' distance; O's profile is out(v)).
// The edge above u has u's children A and B below it and, at its upper end p, u's sibling S and the rest O above:
//
//          O               in(u)  = (in(A) + in(B)) / 2
//          |               out(u) = (in(S) + out(p)) / 2
//          p
//         / \              length = d(in(u), out(u)) - d(A, B) / 2 - d(S, O) / 2
//        u   S                    = cross            - (div below + div above) / 2
//       / \
//      A   B
//
// cross is the distance across the edge; div, the distance between the two subtrees at one end (0 at a leaf).
// d is linear in each profile, so each internal end writes cross with its own distances:
// from p, (d(u, S) + d(u, O)) / 2; from u, (d(A, out(u)) + d(B, out(u))) / 2, the mean of u's dAO and dBO.
// With complete profiles the two are equal; with capped ones they differ, and the length takes their mean.
// The root edge joins r_left and r_right (out(r_left) = in(r_right)): its length goes above r_left; r_right gets 0.
template <class T>
double assemble_lengths(int32_t N, const int32_t *h_c1, const int32_t *h_c2, const std::vector<int32_t> &parent,
                        int32_t r_left, int32_t r_right, const std::vector<double> &h_dAB,
                        const std::vector<double> &h_dAO, const std::vector<double> &h_dBO,
                        std::vector<double> &h_branch_lengths) {
    double total_bme_len = 0.0;
    for (int32_t u = 0; u < 2 * N - 2; ++u) { // the edge above u
        if (u == r_right) {
            h_branch_lengths[u] = 0.0;
            continue;
        }
        if (u == r_left) {
            // The root edge, between r_left and r_right: cross from each end that is internal (a leaf end takes the
            // other end's), div from each end.
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

        // Any other edge: u below, its parent p above (p is not the root, so it has a row).
        int32_t p = parent[u];
        int32_t p_idx = p - N;
        bool is_left = (h_c1[p] == u);                              // u is p's A (c1), else its B (c2)
        T d_ab_p = T(h_dAB[p_idx]);                                 // d(u, S)
        T d_child_out_p = T(is_left ? h_dAO[p_idx] : h_dBO[p_idx]); // d(u, O)
        T d_sib_out_p = T(is_left ? h_dBO[p_idx] : h_dAO[p_idx]);   // d(S, O): div above

        T bl;
        if (u < N) {
            // A leaf: cross from p only, and no div below.
            bl = T(0.5) * (d_ab_p + d_child_out_p - d_sib_out_p);
        } else {
            int32_t u_idx = u - N;
            T cross_from_p = T(0.5) * (d_ab_p + d_child_out_p);
            T cross_from_u = T(0.5) * (T(h_dAO[u_idx]) + T(h_dBO[u_idx]));
            // The mean of the two crosses, minus half of div above (d(S, O)) and div below (d(A, B)).
            bl = T(0.5) * (cross_from_p + cross_from_u) - T(0.5) * (d_sib_out_p + T(h_dAB[u_idx]));
        }
        h_branch_lengths[u] = bl;
        total_bme_len += bl;
    }

    return total_bme_len;
}

} // namespace

// Computes the lengths in four steps:
// 1. orders the internal nodes, children before parents, and records each node's parent;
// 2. builds the inside profiles bottom-up, then the outside profiles top-down (host);
// 3. computes the three distances of each internal node but the root (devices);
// 4. combines them into the lengths (host, assemble_lengths).
double compute_bme_branch_lengths(const Tree &tree, Devices &devices, std::vector<double> &h_branch_lengths,
                                  int32_t cap, double *out_profile_sec, double *out_kernel_sec, bool double_precision) {
    const int32_t N = tree.n_leaves(); // leaves 0..N-1, internal nodes N..2N-2
    const int32_t *h_c1 = tree.c1.data(), *h_c2 = tree.c2.data();
    const int32_t n_gpus = (int32_t)devices.size();
    const double t0 = omp_get_wtime();
    const int32_t root = 2 * N - 2;
    // The internal nodes but the root, N .. 2N-3, have profiles and distances: node N + i in row i of each table.
    const int32_t n_int = N - 2;
    h_branch_lengths.assign(2 * N - 2, 0.0);
    if (N <= 2) // lengths left at 0 (infer() needs N >= 4)
        return 0.0;

    // 1. Order the internal nodes N .. 2N-2 children before parents (postorder), and record each node's parent.
    // A depth-first walk from the root visits parents before children; reversed, it is a postorder.
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

    // 2. Inside and outside profiles of the internal nodes N .. 2N-3, in flat tables (the layout the devices read).
    // Node N + i's profile is in row i: its in_len[i] entries in in_leaves and in_weights from i * cap on, sorted by
    // decreasing weight (out_* alike).
    std::vector<int32_t> in_len(n_int, 0), out_len(n_int, 0);
    std::vector<int32_t> in_leaves((size_t)n_int * cap, 0), out_leaves((size_t)n_int * cap, 0);
    std::vector<float> in_weights((size_t)n_int * cap, 0.0f), out_weights((size_t)n_int * cap, 0.0f);

    std::vector<LeafWeight> tmp_a(cap), tmp_b(cap), tmp_out(cap); // one merge's two inputs and its result

    // Copies in(u) into buf; returns its length.
    // A leaf has no row: its profile is itself, with weight 1.
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

    // 2a. Inside profiles, children before parents: in(u) = (in(c1) + in(c2)) / 2.
    // The root has no row: it is the lower end of no edge.
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
    // Stores buf (l entries) as out(u).
    // A leaf has no row: the edge above it uses its parent's distances, not its own outside profile.
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

    // 2b. Outside profiles, parents before children (postorder read backwards).
    // The root's children are joined by one edge, so each sees the other's subtree: out(r_left) = in(r_right) and
    // out(r_right) = in(r_left).
    {
        int32_t nr = get_in_profile(r_right, tmp_a.data());
        set_out_profile(r_left, tmp_a.data(), nr);
        int32_t nl = get_in_profile(r_left, tmp_b.data());
        set_out_profile(r_right, tmp_b.data(), nl);
    }

    // Below them, for each internal child: out(c1) = (out(u) + in(c2)) / 2 and out(c2) = (out(u) + in(c1)) / 2.
    for (int32_t pi = (int32_t)postorder.size() - 1; pi >= 0; --pi) {
        int32_t u = postorder[pi];
        if (u == root)
            continue;
        int32_t idx = u - N;
        int32_t no = out_len[idx];
        // out(u) into tmp_a, the input both children share
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

    // 3. Edge-length pass on the devices: the three distances of each internal node but the root.
    // Node N + i, with children A = c1 and B = c2 and the rest O: h_dAB[i] = d(in(A), in(B)),
    // h_dAO[i] = d(in(A), out(N + i)) and h_dBO[i] = d(in(B), out(N + i)).
    // Each device takes one block of consecutive rows.
    std::vector<double> h_dAB(n_int, 0.0), h_dAO(n_int, 0.0), h_dBO(n_int, 0.0);
    int32_t active_gpus = std::max(1, std::min(n_gpus, n_int)); // no more devices than rows

    ParallelErrors errors;
    // one work item per device, handed out by iteration (not by thread number, so every chunk is computed even if
    // OpenMP gives fewer threads, e.g. when called from inside a parallel region)
#pragma omp parallel for num_threads(active_gpus) schedule(static, 1)
    for (int g = 0; g < active_gpus; ++g) {

        // device g's rows: start .. end - 1
        int32_t chunk = (n_int + active_gpus - 1) / active_gpus;
        int32_t start = g * chunk;
        int32_t end = std::min(n_int, start + chunk);
        int32_t count = std::max(0, end - start);

        if (count > 0) {
            try {
                // c1, c2, the outside profiles and the distances from row start on; the inside profiles in full, since
                // the children of the block's nodes can be in any row.
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
    if (out_kernel_sec) // steps 3 and 4
        *out_kernel_sec = t2 - t1;
    return total_bme_len;
}

} // namespace p2vbme
