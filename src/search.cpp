/*
 * search.cpp
 *
 * The minibatch BME search (host side): Phylo2Vec encoding of skeletons, tree summaries, clade and outside
 * profiles, window extraction, the leaf-covering waves and the rounds. Window solves run on the devices.
 */
#include "search.hpp"

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parallel_errors.hpp"
#include "phylo2vec.hpp"
#include "timing.hpp"

namespace p2vbme {

void build_skeleton_metadata(const int64_t *v_cf, int32_t K, std::vector<int32_t> &c1_cf, std::vector<int32_t> &c2_cf,
                             std::vector<int32_t> &v_map_cf, std::vector<int32_t> &m_lmin_cf,
                             std::vector<int32_t> &next_k_cf, std::vector<int32_t> &parent_u_cf,
                             std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
                             std::vector<int32_t> &adj_u_cf, int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1,
                             int32_t *out_rc2) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;

    c1_cf.assign(R_sk, 0);
    c2_cf.assign(R_sk, 0);
    v_map_cf.assign(K, 0);
    m_lmin_cf.assign(K, 0);
    next_k_cf.assign(K, 0);
    parent_u_cf.assign(M_sk, -1);
    sub_sz_cf.assign(M_sk, 1);
    tin_cf.assign(M_sk, 0);
    adj_u_cf.assign(3 * M_sk, -1);

    std::vector<int32_t> scratch_uf(R_sk), last_seen(K, K);
    for (int32_t x = 0; x < K; ++x)
        scratch_uf[x] = x;

    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        int32_t b0 = scratch_uf[y], b1 = scratch_uf[x];
        c1_cf[m] = b0;
        c2_cf[m] = b1;
        m_lmin_cf[k] = x;
        v_map_cf[y] = b0;
        scratch_uf[x] = m;
    }
    for (int32_t k = K - 3; k >= 0; --k) {
        int32_t lmin = m_lmin_cf[k];
        next_k_cf[k] = last_seen[lmin];
        last_seen[lmin] = k;
    }
    int32_t rc1 = c1_cf[R_sk - 1], rc2 = c2_cf[R_sk - 1];
    int32_t other_rc = (rc1 < rc2) ? rc1 : rc2;
    int32_t r_c0 = c1_cf[root_u_sk], r_c1 = c2_cf[root_u_sk], r_c2 = other_rc;
    *out_other_rc = other_rc;
    *out_rc0 = r_c0;
    *out_rc1 = r_c1;
    *out_rc2 = r_c2;

    for (int32_t u = 0; u < K; ++u)
        sub_sz_cf[u] = 1;
    for (int32_t u = K; u < root_u_sk; ++u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        parent_u_cf[b0] = u;
        parent_u_cf[b1] = u;
        sub_sz_cf[u] = 1 + sub_sz_cf[b0] + sub_sz_cf[b1];
        adj_u_cf[3 * u + 0] = b0;
        adj_u_cf[3 * u + 1] = b1;
    }
    sub_sz_cf[root_u_sk] = M_sk; // the root's Euler interval spans every node: it is an ancestor of all
    parent_u_cf[r_c0] = root_u_sk;
    parent_u_cf[r_c1] = root_u_sk;
    parent_u_cf[r_c2] = root_u_sk;
    parent_u_cf[root_u_sk] = -1;
    for (int32_t u = 0; u < K; ++u) {
        adj_u_cf[3 * u + 0] = parent_u_cf[u];
        adj_u_cf[3 * u + 1] = -1;
        adj_u_cf[3 * u + 2] = -1;
    }
    for (int32_t u = K; u < root_u_sk; ++u)
        adj_u_cf[3 * u + 2] = parent_u_cf[u];
    adj_u_cf[3 * root_u_sk + 0] = r_c0;
    adj_u_cf[3 * root_u_sk + 1] = r_c1;
    adj_u_cf[3 * root_u_sk + 2] = r_c2;

    tin_cf[root_u_sk] = 0;
    tin_cf[r_c0] = 1;
    tin_cf[r_c1] = 1 + sub_sz_cf[r_c0];
    tin_cf[r_c2] = 1 + sub_sz_cf[r_c0] + sub_sz_cf[r_c1];
    for (int32_t u = root_u_sk - 1; u >= K; --u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        tin_cf[b0] = tin_cf[u] + 1;
        tin_cf[b1] = tin_cf[u] + 1 + sub_sz_cf[b0];
    }
}

void compute_tree_summary(const Tree &tree, TreeSummary &out) {
    const int32_t N = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    out.order.resize(N - 1);
    out.parent.assign(n_nodes_r, -1);
    int32_t ord_len = 0;
    out.order[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = out.order[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        out.parent[b0] = u;
        out.parent[b1] = u;
        if (b0 >= N)
            out.order[ord_len++] = b0;
        if (b1 >= N)
            out.order[ord_len++] = b1;
    }

    out.leaf_count.assign(n_nodes_r, 1);
    out.stand_in.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x) {
        out.stand_in[x] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = out.order[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        out.leaf_count[u] = out.leaf_count[b0] + out.leaf_count[b1];
        if (min_depth[b0] <= min_depth[b1]) {
            min_depth[u] = 1 + min_depth[b0];
            out.stand_in[u] = out.stand_in[b0];
        } else {
            min_depth[u] = 1 + min_depth[b1];
            out.stand_in[u] = out.stand_in[b1];
        }
    }
}

// Compute the dyadic leaf-weight profile w(x) = 2^{-depth_u(x)} for a rooted clade u.
// Expands internal nodes in best-first order (highest weight w = 2^{-d} first).
// When max_prof_leaves <= 0 or leaf_count[u] <= max_prof_leaves (for balanced, unbalanced, or caterpillar clades
// alike), all leaves in u are emitted with their exact dyadic weights.
// When max_prof_leaves > 0 and leaf_count[u] > max_prof_leaves, expansion stops at max_prof_leaves entries and each
// remaining frontier subtree in the priority queue collapses its mass onto a single representative leaf
// stand_in[top.u] (an approximation that can retain substantial collapsed mass on balanced clades of
// size > max_prof_leaves).
static void compute_clade_dyadic_profile(int32_t u, const Tree &tree, const TreeSummary &summary,
                                         int32_t max_prof_leaves, std::vector<int32_t> &prof_leaves_out,
                                         std::vector<float> &prof_weights_out) {
    const int32_t N = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    if (u < N) {
        prof_leaves_out.push_back(u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct StackEntry {
            int32_t u;
            double weight;
        };
        std::vector<StackEntry> st;
        st.reserve(64);
        st.push_back({u, 1.0});
        while (!st.empty()) {
            StackEntry top = st.back();
            st.pop_back();
            int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
            double hw = 0.5 * top.weight;
            if (b1 < N) {
                prof_leaves_out.push_back(b1);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b1, hw});
            }
            if (b0 < N) {
                prof_leaves_out.push_back(b0);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b0, hw});
            }
        }
        return;
    }
    struct WeightedNode {
        double weight;
        int32_t u;
        bool operator<(const WeightedNode &o) const {
            if (weight != o.weight)
                return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<WeightedNode> pq;
    pq.push({1.0, u});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves)
            break;
        WeightedNode top = pq.top();
        pq.pop();
        int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
        double hw = 0.5 * top.weight;
        if (b0 < N) {
            prof_leaves_out.push_back(b0);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b0});
        }
        if (b1 < N) {
            prof_leaves_out.push_back(b1);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b1});
        }
    }
    while (!pq.empty()) {
        WeightedNode top = pq.top();
        pq.pop();
        prof_leaves_out.push_back(summary.stand_in[top.u]);
        prof_weights_out.push_back((float)top.weight);
    }
}

// Compute the unrooted dyadic leaf-weight profile w_out(x) = 2^{-d_unrooted(sub_root, x)} for the entire outside
// component V \ L(sub_root) across the full ancestor path to root_r (suppressing degree-2 root_r).
// Expands directed subtrees in best-first weight order.
// Exact when max_prof_leaves <= 0 or N - leaf_count[sub_root] <= max_prof_leaves; above a positive cap
// max_prof_leaves > 0, unresolved directed frontier subtrees collapse onto their stand_in leaves.
static void compute_outgroup_dyadic_profile(int32_t sub_root, const Tree &tree, const TreeSummary &summary,
                                            int32_t max_prof_leaves, std::vector<int32_t> &prof_leaves_out,
                                            std::vector<float> &prof_weights_out) {
    const int32_t N = tree.n_leaves();
    const int32_t root_r = 2 * N - 2;
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    if (sub_root == root_r || summary.parent[sub_root] == -1)
        return;
    int32_t p0 = summary.parent[sub_root];
    int32_t start_u = (p0 == root_r) ? ((c1_r[root_r] == sub_root) ? c2_r[root_r] : c1_r[root_r]) : p0;
    if (start_u < N) {
        prof_leaves_out.push_back(start_u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct DirectedStackEntry {
            int32_t u;
            int32_t from_u;
            double weight;
        };
        std::vector<DirectedStackEntry> st;
        st.reserve(64);
        st.push_back({start_u, sub_root, 1.0});
        while (!st.empty()) {
            DirectedStackEntry top = st.back();
            st.pop_back();
            int32_t v = top.u;
            int32_t n0 = c1_r[v];
            int32_t n1 = c2_r[v];
            int32_t pv = summary.parent[v];
            int32_t n2 = (pv == root_r) ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r]) : pv;
            int32_t nbs[3] = {n0, n1, n2};
            double hw = 0.5 * top.weight;
            for (int k = 2; k >= 0; --k) {
                int32_t nb = nbs[k];
                if (nb == top.from_u || nb < 0)
                    continue;
                if (nb < N) {
                    prof_leaves_out.push_back(nb);
                    prof_weights_out.push_back((float)hw);
                } else {
                    st.push_back({nb, v, hw});
                }
            }
        }
        return;
    }
    struct DirectedWeightedNode {
        double weight;
        int32_t u;
        int32_t from_u;
        bool operator<(const DirectedWeightedNode &o) const {
            if (weight != o.weight)
                return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<DirectedWeightedNode> pq;
    pq.push({1.0, start_u, sub_root});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves)
            break;
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t n0 = c1_r[v];
        int32_t n1 = c2_r[v];
        int32_t pv = summary.parent[v];
        int32_t n2 = (pv == root_r) ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r]) : pv;
        int32_t nbs[3] = {n0, n1, n2};
        double hw = 0.5 * top.weight;
        for (int k = 0; k < 3; ++k) {
            int32_t nb = nbs[k];
            if (nb == top.from_u || nb < 0)
                continue;
            if (nb < N) {
                prof_leaves_out.push_back(nb);
                prof_weights_out.push_back((float)hw);
            } else {
                pq.push({hw, nb, v});
            }
        }
    }
    while (!pq.empty()) {
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t rep_leaf;
        if (top.from_u != c1_r[v] && top.from_u != c2_r[v]) {
            rep_leaf = summary.stand_in[v];
        } else {
            int32_t sib = (top.from_u == c1_r[v]) ? c2_r[v] : c1_r[v];
            rep_leaf = summary.stand_in[sib];
        }
        prof_leaves_out.push_back(rep_leaf);
        prof_weights_out.push_back((float)top.weight);
    }
}

void extract_active_skeleton(int32_t root_r, const Tree &tree, const TreeSummary &summary,
                             std::vector<int32_t> &super_roots, std::vector<int32_t> &split_internals,
                             std::vector<int32_t> &skel_left, std::vector<int32_t> &skel_right,
                             int32_t *out_root_skel_id, const uint8_t *stop_at_comp_root) {
    const int32_t N = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    if (root_r < N) {
        super_roots.push_back(root_r);
        *out_root_skel_id = -1;
        return;
    }

    struct ExpNode {
        int32_t u;
        int32_t left_exp;
        int32_t right_exp;
    };
    struct PqCand {
        int32_t prio;
        int32_t u;
        int32_t parent_exp;
        int32_t side;
        bool operator<(const PqCand &o) const {
            if (prio != o.prio)
                return prio < o.prio;
            return u < o.u;
        }
    };

    std::vector<ExpNode> exp_nodes;
    exp_nodes.reserve(std::min(p2vbme::WINDOW_MAX_INTERNALS, std::max(16, summary.leaf_count[root_r])));
    exp_nodes.push_back({root_r, -1, -1});

    auto is_eligible = [&](int32_t w) -> bool {
        if (w < N)
            return false;
        if (stop_at_comp_root && stop_at_comp_root[w])
            return false;
        return true;
    };

    std::priority_queue<PqCand> pq;
    int32_t b0_root = c1_r[root_r], b1_root = c2_r[root_r];
    if (is_eligible(b0_root))
        pq.push({summary.leaf_count[b0_root], b0_root, 0, 0});
    if (is_eligible(b1_root))
        pq.push({summary.leaf_count[b1_root], b1_root, 0, 1});

    while (!pq.empty() && (int32_t)exp_nodes.size() < p2vbme::WINDOW_MAX_INTERNALS) {
        PqCand cand = pq.top();
        pq.pop();
        int32_t my_exp = (int32_t)exp_nodes.size();
        exp_nodes.push_back({cand.u, -1, -1});
        if (cand.side == 0)
            exp_nodes[cand.parent_exp].left_exp = my_exp;
        else
            exp_nodes[cand.parent_exp].right_exp = my_exp;

        int32_t b0 = c1_r[cand.u], b1 = c2_r[cand.u];
        if (is_eligible(b0))
            pq.push({summary.leaf_count[b0], b0, my_exp, 0});
        if (is_eligible(b1))
            pq.push({summary.leaf_count[b1], b1, my_exp, 1});
    }

    struct Frame {
        int32_t exp_idx;
        int32_t state;
        int32_t left_id;
    };
    std::vector<Frame> st;
    st.reserve(256);
    st.push_back({0, 0, -1});
    int32_t last_ret = -1;

    while (!st.empty()) {
        Frame &f = st.back();
        const ExpNode &en = exp_nodes[f.exp_idx];
        if (f.state == 0) {
            f.state = 1;
            if (en.left_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c1_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.left_exp, 0, -1});
            }
        } else if (f.state == 1) {
            f.left_id = last_ret;
            f.state = 2;
            if (en.right_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c2_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.right_exp, 0, -1});
            }
        } else {
            int32_t right_id = last_ret;
            int32_t my_idx = (int32_t)split_internals.size();
            split_internals.push_back(en.u);
            skel_left.push_back(f.left_id);
            skel_right.push_back(right_id);
            last_ret = -1 - my_idx;
            st.pop_back();
        }
    }
    *out_root_skel_id = -1 - last_ret;
}

// Cuts the tree into components and schedules them as waves of windows.
// A component is a connected set of internal nodes: its root and the nodes below it, down to the roots of the
// components below.
// For example (sizes not to scale):
//
//              r                A = {r, x, y}: the root's component, level 1
//            /   \
//           x     y
//          / \   / \
//         b  l1 l2  c           b, c: the roots of components B and C, level 0
//        / \       / \
//      ...  ...  ...  ...       B, C: the nodes below b and c, down to the next component roots
//
// B and C are holes in A: A's window sees each as one piece (its pieces: B, l1, l2, C).
// l1 and l2 are A's own leaves, so A gets a window.
// B's window sees B's nodes, with the rest of the tree as one more piece (the outgroup).
// Components below the root have COMPONENT_MIN_NODES to max_cap internal nodes; the root's has up to max_cap + 4.
// is_comp_root marks the component roots.
// waves[k] lists the roots of the components at level k: 0 = no component below, k = one above the highest
// component below; components of one level are never nested, so their windows can run in parallel.
// Only components with a leaf of their own get a window; the others (connectors between components) are left to the
// backbone and V-cycle windows.
void build_component_waves(const Tree &tree, const TreeSummary &summary, int32_t max_cap,
                           std::vector<uint8_t> &is_comp_root, std::vector<std::vector<int32_t>> &waves) {
    // The smallest component below the root, and the smallest root component (heuristics).
    constexpr int32_t COMPONENT_MIN_NODES = 15;
    constexpr int32_t ROOT_COMPONENT_MIN_NODES = 5;
    // With a smaller cap, step 1 could close a component under COMPONENT_MIN_NODES.
    if (max_cap < 2 * COMPONENT_MIN_NODES)
        throw std::invalid_argument("build_component_waves: max_cap must be at least " +
                                    std::to_string(2 * COMPONENT_MIN_NODES));
    const int32_t N = tree.n_leaves();
    const std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    is_comp_root.assign(n_nodes_r, 0);
    waves.clear();
    if (N < p2vbme::WINDOW_MIN_K)
        return;

    // Step 1, bottom up: comp_size[u] is the number of internal nodes in the component u tops so far (0 for a leaf).
    // A node joins its children's components: comp_size[u] = 1 + their sizes.
    // If that would exceed max_cap, it first closes the components below it: both children's if each has at least
    // COMPONENT_MIN_NODES nodes, otherwise only the larger one, and the smaller joins u's component.
    // A closed component's size stays in comp_size at its root (step 2 reads it).
    std::vector<int32_t> comp_size(n_nodes_r, 0);
    for (int32_t idx = (int32_t)summary.order.size() - 1; idx >= 0; --idx) {
        int32_t u = summary.order[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t r1 = comp_size[c1], r2 = comp_size[c2];
        if (1 + r1 + r2 > max_cap) {
            if (r1 >= COMPONENT_MIN_NODES && r2 >= COMPONENT_MIN_NODES) {
                is_comp_root[c1] = 1;
                is_comp_root[c2] = 1;
                r1 = 0;
                r2 = 0;
            } else if (r1 >= r2) {
                is_comp_root[c1] = 1;
                r1 = 0;
            } else {
                is_comp_root[c2] = 1;
                r2 = 0;
            }
        }
        comp_size[u] = 1 + r1 + r2;
    }

    // Step 2: a root component under ROOT_COMPONENT_MIN_NODES nodes absorbs the smallest component just below it
    // (found by walking down from the root to the first component roots).
    if (comp_size[root_r] < ROOT_COMPONENT_MIN_NODES) {
        std::vector<int32_t> st = {root_r};
        int32_t best_c = -1;
        while (!st.empty()) {
            int32_t u = st.back();
            st.pop_back();
            int32_t ch[2] = {curr_c1[u], curr_c2[u]};
            for (int k = 0; k < 2; ++k) {
                int32_t c = ch[k];
                if (c >= N) {
                    if (is_comp_root[c]) {
                        if (best_c == -1 || comp_size[c] < comp_size[best_c])
                            best_c = c;
                    } else {
                        st.push_back(c);
                    }
                }
            }
        }
        if (best_c != -1) {
            is_comp_root[best_c] = 0;
            comp_size[root_r] += comp_size[best_c];
        }
    }
    is_comp_root[root_r] = 1;

    // Step 3, bottom up: the level of each component, and the leaves of its own.
    // max_ch_lvl[u]: the highest level among the components below u's component, reached from u (-1 = none).
    // direct_leaves[u]: the leaves below u that are in u's component (not inside a component below).
    std::vector<int32_t> comp_lvl(n_nodes_r, 0);
    std::vector<int32_t> max_ch_lvl(n_nodes_r, -1);
    std::vector<int32_t> direct_leaves(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x)
        direct_leaves[x] = 1;
    int32_t max_lvl = 0;
    for (int32_t idx = (int32_t)summary.order.size() - 1; idx >= 0; --idx) {
        int32_t u = summary.order[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t l1 = (c1 >= N && is_comp_root[c1]) ? (comp_lvl[c1] + 1) : max_ch_lvl[c1];
        int32_t l2 = (c2 >= N && is_comp_root[c2]) ? (comp_lvl[c2] + 1) : max_ch_lvl[c2];
        int32_t dl1 = (c1 >= N && is_comp_root[c1]) ? 0 : direct_leaves[c1];
        int32_t dl2 = (c2 >= N && is_comp_root[c2]) ? 0 : direct_leaves[c2];
        direct_leaves[u] = dl1 + dl2;
        max_ch_lvl[u] = std::max(l1, l2);
        if (is_comp_root[u]) {
            comp_lvl[u] = std::max(0, max_ch_lvl[u]);
            if (comp_lvl[u] > max_lvl)
                max_lvl = comp_lvl[u];
        }
    }

    // Step 4: each component with a leaf of its own goes into the wave of its level.
    waves.resize(max_lvl + 1);
    for (int32_t idx = (int32_t)summary.order.size() - 1; idx >= 0; --idx) {
        int32_t u = summary.order[idx];
        if (is_comp_root[u] && direct_leaves[u] > 0) {
            waves[comp_lvl[u]].push_back(u);
        }
    }
    while (!waves.empty() && waves.back().empty()) {
        waves.pop_back();
    }
}

void apply_unrooted_spr_host(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_u, int32_t dest_v) {
    int32_t n0 = -1, n1 = -1;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb = adj_u[3 * anchor_node + k];
        if (nb != prune_node) {
            if (n0 == -1)
                n0 = nb;
            else
                n1 = nb;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * n0 + k] == anchor_node) {
            adj_u[3 * n0 + k] = n1;
            break;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * n1 + k] == anchor_node) {
            adj_u[3 * n1 + k] = n0;
            break;
        }
    }
    if (dest_u == anchor_node)
        dest_u = n0;
    if (dest_v == anchor_node)
        dest_v = n0;

    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_u + k] == dest_v) {
            adj_u[3 * dest_u + k] = anchor_node;
            break;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_v + k] == dest_u) {
            adj_u[3 * dest_v + k] = anchor_node;
            break;
        }
    }
    adj_u[3 * anchor_node + 0] = prune_node;
    adj_u[3 * anchor_node + 1] = dest_u;
    adj_u[3 * anchor_node + 2] = dest_v;
}

void relabel_from_adj_u(const int32_t *adj_u, int32_t K, int32_t root_edge_u, int32_t root_edge_v, int32_t flip_mode,
                        std::vector<int64_t> &v_cf, std::vector<int32_t> &perm) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1_r(R_sk, 0), c2_r(R_sk, 0);
    c1_r[root_r] = root_edge_u;
    c2_r[root_r] = root_edge_v;

    std::vector<int32_t> q_u;
    std::vector<int32_t> q_p;
    q_u.reserve(R_sk);
    q_p.reserve(R_sk);
    if (root_edge_u >= K) {
        q_u.push_back(root_edge_u);
        q_p.push_back(root_edge_v);
    }
    if (root_edge_v >= K) {
        q_u.push_back(root_edge_v);
        q_p.push_back(root_edge_u);
    }

    size_t head = 0;
    while (head < q_u.size()) {
        int32_t u = q_u[head];
        int32_t pr = q_p[head];
        head++;
        int32_t b0 = -1, b1 = -1;
        for (int32_t k = 0; k < 3; ++k) {
            int32_t nb = adj_u[3 * u + k];
            if (nb != -1 && nb != pr) {
                if (b0 == -1)
                    b0 = nb;
                else
                    b1 = nb;
            }
        }
        c1_r[u] = b0;
        c2_r[u] = b1;
        if (b0 >= K) {
            q_u.push_back(b0);
            q_p.push_back(u);
        }
        if (b1 >= K) {
            q_u.push_back(b1);
            q_p.push_back(u);
        }
    }

    std::vector<int32_t> step_perm(K);
    queue_shuffle_rooted(c1_r.data(), c2_r.data(), K, flip_mode, v_cf.data(), step_perm.data());
    std::vector<int32_t> new_perm(K);
    for (int32_t a = 0; a < K; ++a)
        new_perm[a] = perm[step_perm[a]];
    perm = new_perm;
}

// Compute all-pairs unrooted topological edge distances dist_out[a * K + b] between original super-leaves a, b in
// {0, ..., K-1} for skeleton (v_cf, perm) in O(K^2) time by visiting each leaf pair once at its rooted LCA.
void compute_skeleton_leaf_dists(const int64_t *v_cf, const int32_t *perm, int32_t K, std::vector<int32_t> &dist_out) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1(R_sk), c2(R_sk), uf(R_sk);
    for (int32_t x = 0; x < K; ++x)
        uf[x] = x;
    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        c1[m] = uf[y];
        c2[m] = uf[x];
        uf[x] = m;
    }
    std::vector<int32_t> depth(R_sk, 0);
    for (int32_t u = root_r; u >= K; --u) {
        depth[c1[u]] = depth[u] + 1;
        depth[c2[u]] = depth[u] + 1;
    }
    dist_out.assign((size_t)K * K, 0);
    std::vector<std::vector<int32_t>> sub_leaves(R_sk);
    for (int32_t x = 0; x < K; ++x)
        sub_leaves[x].push_back(x);
    for (int32_t u = K; u <= root_r; ++u) {
        int32_t b0 = c1[u], b1 = c2[u];
        const auto &L0 = sub_leaves[b0];
        const auto &L1 = sub_leaves[b1];
        int32_t sub = (u == root_r) ? 1 : (2 * depth[u]);
        for (int32_t x : L0) {
            int32_t ox = perm[x];
            int32_t dx = depth[x] - sub;
            for (int32_t y : L1) {
                int32_t oy = perm[y];
                int32_t d = dx + depth[y];
                dist_out[(size_t)ox * K + oy] = d;
                dist_out[(size_t)oy * K + ox] = d;
            }
        }
        if (u < root_r) {
            auto &Lu = sub_leaves[u];
            Lu.reserve(L0.size() + L1.size());
            Lu.insert(Lu.end(), L0.begin(), L0.end());
            Lu.insert(Lu.end(), L1.begin(), L1.end());
            std::vector<int32_t>().swap(sub_leaves[b0]);
            std::vector<int32_t>().swap(sub_leaves[b1]);
        }
    }
}

namespace {
// A candidate move of the window solve: row i regrafted onto edge j, BME change d.
struct MoveCand {
    int32_t i;
    int32_t j;
    double d;
};
} // namespace

// Exact acceptance check (exact_check_eta).
// A window's skeleton update is re-scored on complete leaf profiles only when its capped-profile gain is small,
// dL_capped > -eta; larger gains are accepted as proposed.
// eta = inf checks every improving window; eta = 0 (default) none.

// Optimize one K-leaf skeleton with path-disjoint SPR batching, a monotonic condensed-skeleton BME check, and (when
// exact_builder is supplied) an exact full-objective acceptance check on all changed super-leaf pairs involving capped
// clades or capped outside profiles.
// The device does the arithmetic.
double solve_window_host(int32_t K, int32_t max_iters, int32_t max_batch_cap, double exact_check_eta,
                         const int32_t *prof_offsets, const int32_t *prof_leaves, const float *prof_weights,
                         std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                         const ExactProfileBuilder *exact_builder, WindowKernels &kernels) {
    // complete profiles, from exact_builder when the check needs them
    const int32_t *exact_prof_offsets = NULL;
    const int32_t *exact_prof_leaves = NULL;
    const float *exact_prof_weights = NULL;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;
    const std::vector<int64_t> init_v_cf = v_cf;
    const std::vector<int32_t> init_perm = perm;

    kernels.distances(K, prof_offsets, prof_leaves, prof_weights);

    SkeletonMeta sk;
    std::vector<double> h_best_delta(K);
    std::vector<int32_t> h_best_j(K);

    double init_L = 0.0, curr_L = 0.0, best_L = 1e300, prev_L = 1e300;
    std::vector<int64_t> best_v_cf = v_cf, prev_v_cf = v_cf;
    std::vector<int32_t> best_perm = perm, prev_perm = perm;
    int32_t flipped_already = 0;
    int32_t cur_batch_cap = max_batch_cap; // the batch cap B of Algorithm S1
    bool batch_pending = false;            // a batch was applied in the previous sweep and is checked now

    for (int32_t iter = 0; iter <= max_iters; ++iter) {
        build_skeleton_metadata(v_cf.data(), K, sk.c1, sk.c2, sk.v_map, sk.m_lmin, sk.next_k, sk.parent_u, sk.sub_size,
                                sk.tin, sk.adj_u, &sk.other_rc, &sk.rc0, &sk.rc1, &sk.rc2);
        const int32_t other_rc = sk.other_rc;
        std::vector<int32_t> &v_map_cf = sk.v_map, &parent_u_cf = sk.parent_u, &sub_size_cf = sk.sub_size;
        std::vector<int32_t> &tin_cf = sk.tin, &adj_u_cf = sk.adj_u;

        curr_L = kernels.score(sk, perm);
        if (iter == 0) {
            init_L = curr_L;
            best_L = curr_L;
            prev_L = curr_L;
        } else if (batch_pending && curr_L >= prev_L - 1e-8) {
            // the last batch did not lower the objective: restore the tree before it and use smaller batches (Algorithm
            // S1, line 4).
            // A single move that fails is below the resolution of the scores, so the window is converged.
            v_cf = prev_v_cf;
            perm = prev_perm;
            batch_pending = false;
            if (cur_batch_cap <= 1)
                break;
            cur_batch_cap = std::max(1, cur_batch_cap / 2);
            continue;
        } else if (batch_pending) {
            cur_batch_cap = std::min(max_batch_cap, cur_batch_cap + 16); // it fell: allow larger batches
        }
        batch_pending = false;
        prev_L = curr_L;
        prev_v_cf = v_cf;
        prev_perm = perm;
        if (curr_L < best_L) {
            best_L = curr_L;
            best_v_cf = v_cf;
            best_perm = perm;
        }
        if (iter == max_iters)
            break;
        kernels.best_moves(sk, h_best_delta, h_best_j);

        std::vector<MoveCand> cands;
        // A move is applied only if its single-precision score is below this numerical zero; any larger cutoff leaves
        // real improvements on the table (checked against double-precision scores and FastME's SPR optimum).
        // Termination is guaranteed by max_iters and by the double-precision length check that reverts a batch which
        // does not decrease L.
        const float eps_move_thresh = -1e-6f;
        for (int32_t i = 1; i < K - 1; ++i) {
            if (h_best_delta[i] < eps_move_thresh && h_best_j[i] >= 0 && h_best_j[i] < root_u_sk) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        if (cands.empty()) {
            if (flipped_already == 0) {
                flipped_already = 1;
                relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, other_rc, 1, v_cf, perm);
                continue;
            }
            break;
        }
        flipped_already = 0;
        std::sort(cands.begin(), cands.end(), [](const MoveCand &a, const MoveCand &b) { return a.d < b.d; });

        std::vector<uint8_t> tin_busy(M_sk, 0);
        int32_t n_applied = 0;
        std::vector<int32_t> mark_tins;
        mark_tins.reserve(256);
        const int32_t eff_batch_cap = cur_batch_cap;

        for (const auto &mc : cands) {
            int32_t i = mc.i, dst_u = mc.j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            if (pr == root_u_sk || dst_u < 0 || dst_u >= root_u_sk)
                continue;
            int32_t dst_v = parent_u_cf[dst_u];
            if (dst_v < 0 || dst_v == root_u_sk)
                continue;

            int32_t t0 = tin_cf[tr], t1 = t0 + sub_size_cf[tr];
            if (t0 <= tin_cf[dst_u] && tin_cf[dst_u] < t1)
                continue;
            if (t0 <= tin_cf[pr] && tin_cf[pr] < t1)
                continue;

            mark_tins.clear();
            bool conflict = false;
            {
                // the move's prune and regraft paths must be node-disjoint from every move already in the batch
                // (Algorithm S1, line 10)
                mark_tins.push_back(tin_cf[tr]);
                mark_tins.push_back(tin_cf[dst_v]);
                int32_t u_anc = pr;
                int32_t t_dst = tin_cf[dst_u];
                int32_t hops = 0;
                while (u_anc != -1 && !(tin_cf[u_anc] <= t_dst && t_dst < tin_cf[u_anc] + sub_size_cf[u_anc])) {
                    mark_tins.push_back(tin_cf[u_anc]);
                    u_anc = parent_u_cf[u_anc];
                    if (++hops > M_sk)
                        break;
                }
                if (u_anc == -1 || hops > M_sk)
                    continue;
                mark_tins.push_back(tin_cf[u_anc]);
                int32_t v_anc = dst_u;
                while (v_anc != u_anc && v_anc != -1) {
                    mark_tins.push_back(tin_cf[v_anc]);
                    v_anc = parent_u_cf[v_anc];
                    if (++hops > M_sk)
                        break;
                }
                if (v_anc != u_anc || hops > M_sk)
                    continue;
                for (int32_t tq : mark_tins) {
                    if (tin_busy[tq]) {
                        conflict = true;
                        break;
                    }
                }
                if (conflict)
                    continue;
            }

            for (int32_t tq : mark_tins)
                tin_busy[tq] = 1;
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
            n_applied++;
            if (n_applied >= eff_batch_cap)
                break;
        }
        if (n_applied == 0) {
            int32_t i = cands[0].i, dst_u = cands[0].j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            int32_t dst_v = parent_u_cf[dst_u];
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
        }
        batch_pending = true;

        int32_t r_nb = adj_u_cf[3 * root_u_sk + 0];
        relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, r_nb, iter & 1, v_cf, perm);
    }

    if (best_L >= init_L - p2vbme::WINDOW_MIN_GAIN) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }

    const bool have_exact = (exact_prof_offsets != NULL) || (exact_builder != NULL && *exact_builder);
    if (have_exact && best_L - init_L > -exact_check_eta) {
        if (exact_prof_offsets == NULL)
            (*exact_builder)(&exact_prof_offsets, &exact_prof_leaves, &exact_prof_weights);
        std::vector<uint8_t> is_capped(K, 0);
        bool any_capped = false;
        for (int32_t a = 0; a < K; ++a) {
            int32_t cap_len = prof_offsets[a + 1] - prof_offsets[a];
            int32_t ex_len = exact_prof_offsets[a + 1] - exact_prof_offsets[a];
            if (ex_len != cap_len) {
                is_capped[a] = 1;
                any_capped = true;
            }
        }
        if (any_capped) {
            std::vector<int32_t> dist_init, dist_best;
            compute_skeleton_leaf_dists(init_v_cf.data(), init_perm.data(), K, dist_init);
            compute_skeleton_leaf_dists(best_v_cf.data(), best_perm.data(), K, dist_best);

            std::vector<int32_t> pair_a, pair_b;
            std::vector<int32_t> uncap_a, uncap_b;
            for (int32_t a = 0; a < K; ++a) {
                for (int32_t b = a + 1; b < K; ++b) {
                    int32_t db = dist_best[(size_t)a * K + b];
                    int32_t di = dist_init[(size_t)a * K + b];
                    if (db != di) {
                        if (is_capped[a] || is_capped[b]) {
                            pair_a.push_back(a);
                            pair_b.push_back(b);
                        } else {
                            uncap_a.push_back(a);
                            uncap_b.push_back(b);
                        }
                    }
                }
            }
            if (pair_a.empty() && uncap_a.empty()) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }
            if (pair_a.size() > 1) {
                std::vector<std::pair<int32_t, int32_t>> sorted_pairs(pair_a.size());
                for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                    int32_t u = pair_a[pid], v = pair_b[pid];
                    int32_t lu = exact_prof_offsets[u + 1] - exact_prof_offsets[u];
                    int32_t lv = exact_prof_offsets[v + 1] - exact_prof_offsets[v];
                    if (lu > lv)
                        std::swap(u, v);
                    sorted_pairs[pid] = std::make_pair(v, u);
                }
                std::sort(sorted_pairs.begin(), sorted_pairs.end());
                for (size_t pid = 0; pid < sorted_pairs.size(); ++pid) {
                    pair_a[pid] = sorted_pairs[pid].second;
                    pair_b[pid] = sorted_pairs[pid].first;
                }
            }

            std::vector<double> h_pair_exact_D(pair_a.size(), 0.0);
            if (!pair_a.empty())
                kernels.exact_distances(pair_a, pair_b, exact_prof_offsets, exact_prof_leaves, exact_prof_weights,
                                        h_pair_exact_D);

            std::vector<double> h_D_cond;
            if (!uncap_a.empty())
                kernels.read_distances(h_D_cond);

            double exact_delta_L = 0.0;
            for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                int32_t a = pair_a[pid], b = pair_b[pid];
                double w_diff =
                    ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * h_pair_exact_D[pid];
            }
            for (size_t uid = 0; uid < uncap_a.size(); ++uid) {
                int32_t a = uncap_a[uid], b = uncap_b[uid];
                double w_diff =
                    ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * (double)h_D_cond[(size_t)a * K + b];
            }

            if (exact_delta_L >= -p2vbme::WINDOW_MIN_GAIN) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }
            v_cf = best_v_cf;
            perm = best_perm;
            return exact_delta_L;
        }
    }

    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}

static inline void update_one_node_summary(int32_t u, int32_t b0, int32_t b1, TreeSummary &summary) {
    summary.parent[b0] = u;
    summary.parent[b1] = u;
    summary.leaf_count[u] = summary.leaf_count[b0] + summary.leaf_count[b1];
    summary.stand_in[u] =
        (summary.leaf_count[b0] <= summary.leaf_count[b1]) ? summary.stand_in[b0] : summary.stand_in[b1];
}

// splitmix64's finalizer: a bijection on 64 bits that spreads every input bit over the output.
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// A 64-bit hash of what a window solve reads: K, max_iters, the skeleton (v, perm) and the profiles (offsets, leaves
// and the bits of the weights).
// The rest is fixed within one search: the devices' alignment, distance model and precision, and the solve's
// constants.
// Two different inputs share a hash with probability about 2^-64; such a collision would skip a solve that might have
// improved the tree.
static uint64_t window_input_hash(int32_t K, int32_t max_iters, const std::vector<int64_t> &v,
                                  const std::vector<int32_t> &perm, const std::vector<int32_t> &offsets,
                                  const std::vector<int32_t> &leaves, const std::vector<float> &weights) {
    uint64_t h = 0;
    auto add = [&h](uint64_t x) { h = mix64(h ^ (x + 0x9e3779b97f4a7c15ULL)); };
    add((uint64_t)(uint32_t)K | ((uint64_t)(uint32_t)max_iters << 32));
    for (int64_t x : v)
        add((uint64_t)x);
    for (int32_t x : perm)
        add((uint32_t)x);
    for (int32_t x : offsets)
        add((uint32_t)x);
    for (size_t i = 0; i < leaves.size(); ++i) {
        uint32_t weight_bits;
        std::memcpy(&weight_bits, &weights[i], sizeof weight_bits);
        add((uint64_t)(uint32_t)leaves[i] | ((uint64_t)weight_bits << 32));
    }
    return h;
}

// Optimize the window below sub_root and write the result into tree and summary.
// The window collapses clades into K taxa, each with a dyadic-weighted leaf profile, and optimizes them as an unrooted
// BME tree.
// A subtree window (sub_root != root_r) also sees the rest of the tree, as one extra taxon, the outgroup, because BME
// sums over all leaf pairs: moves inside the subtree change its distances to the leaves outside.
// Its result is rooted where the outgroup attaches, so the subtree keeps its place, and the outgroup is dropped.
// The whole-tree window (sub_root == root_r) has no outgroup and BME does not fix a root, so it re-roots on the
// centroid edge.
// Windows of one wave run in parallel: they pass a snapshot of the tree and its summary (snapshot, snapshot_summary),
// read their profiles from it, and leave summary.parent alone (the caller recomputes the summary after the wave).
// fruitless: the input hashes (window_input_hash) of this search's solves that found no improvement.
// The solve is deterministic, so a window with one of these inputs would again find none: it is skipped.
// fruitless_hash: set to the input hash when the solve runs and finds no improvement, for the caller to add to
// fruitless.

static double run_subtree_minibatch_window(int32_t sub_root, int32_t max_iters, const SearchSettings &settings,
                                           const std::vector<int32_t> &outgroup_leaves,
                                           const std::vector<float> &outgroup_weights, Device &dev, Tree &tree,
                                           TreeSummary &summary, const std::unordered_set<uint64_t> &fruitless,
                                           std::optional<uint64_t> &fruitless_hash,
                                           const uint8_t *stop_at_comp_root = NULL, const Tree *snapshot = NULL,
                                           const TreeSummary *snapshot_summary = NULL) {
    const int32_t N = tree.n_leaves();
    std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;
    const int32_t root_r = 2 * N - 2;
    const int32_t max_prof_leaves = settings.profile_cap;
    std::vector<int32_t> super_roots, split_internals, skel_left, skel_right;
    int32_t root_skel_id = -1;
    extract_active_skeleton(sub_root, tree, summary, super_roots, split_internals, skel_left, skel_right, &root_skel_id,
                            stop_at_comp_root);

    // A subtree window also sees the rest of the tree, as one extra taxon (the outgroup).
    const bool has_outgroup = sub_root != root_r;
    // The root is alone in its wave, so the whole-tree window never gets a snapshot.
    if (!has_outgroup && snapshot != NULL)
        throw std::logic_error("whole-tree window: unexpected snapshot (the root is alone in its wave)");
    const int32_t K_in = (int32_t)super_roots.size();
    const int32_t K = has_outgroup ? K_in + 1 : K_in;
    if (K < p2vbme::WINDOW_MIN_K || K > dev.max_window())
        return 0.0;

    int32_t parent_of_sub_root = summary.parent[sub_root];
    std::vector<int32_t> sk_c1(2 * K - 1, 0), sk_c2(2 * K - 1, 0);
    for (int32_t s = 0; s < K_in - 1; ++s) {
        int32_t l_raw = skel_left[s], r_raw = skel_right[s];
        sk_c1[K + s] = (l_raw >= 0) ? l_raw : (K + (-1 - l_raw));
        sk_c2[K + s] = (r_raw >= 0) ? r_raw : (K + (-1 - r_raw));
    }
    if (has_outgroup) {
        sk_c1[2 * K - 2] = K + (K_in - 2);
        sk_c2[2 * K - 2] = K_in;
    }

    const Tree &otree = (snapshot != NULL) ? *snapshot : tree;
    const TreeSummary &osum = (snapshot_summary != NULL) ? *snapshot_summary : summary;

    std::vector<int32_t> prof_offsets(K + 1, 0), prof_leaves;
    std::vector<float> prof_weights;
    prof_leaves.reserve(4 * K);
    prof_weights.reserve(4 * K);
    bool need_exact_prof = false;
    for (int32_t a = 0; a < K_in; ++a) {
        prof_offsets[a] = (int32_t)prof_leaves.size();
        compute_clade_dyadic_profile(super_roots[a], otree, osum, max_prof_leaves, prof_leaves, prof_weights);
        if (max_prof_leaves > 0 && summary.leaf_count[super_roots[a]] > max_prof_leaves) {
            need_exact_prof = true;
        }
    }
    prof_offsets[K_in] = (int32_t)prof_leaves.size();
    int32_t outgroup_true_leaves = N - summary.leaf_count[sub_root];
    if (has_outgroup) {
        prof_leaves.insert(prof_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
        prof_weights.insert(prof_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
        prof_offsets[K] = (int32_t)prof_leaves.size();
        if (max_prof_leaves > 0 && outgroup_true_leaves > (int32_t)outgroup_leaves.size()) {
            need_exact_prof = true;
        }
    }

    std::vector<int32_t> exact_prof_offsets, exact_prof_leaves;
    std::vector<float> exact_prof_weights;
    auto build_exact_prof = [&]() {
        const bool cap_exact = N >= p2vbme::EXACT_PROFILE_CAP_MIN_N;
        const int32_t exact_cap = cap_exact ? p2vbme::EXACT_PROFILE_CAP : -1;
        exact_prof_offsets.resize(K + 1, 0);
        exact_prof_leaves.reserve(cap_exact ? (size_t)(K * 256) : (size_t)N);
        exact_prof_weights.reserve(cap_exact ? (size_t)(K * 256) : (size_t)N);
        for (int32_t a = 0; a < K_in; ++a) {
            exact_prof_offsets[a] = (int32_t)exact_prof_leaves.size();
            if (max_prof_leaves <= 0 || summary.leaf_count[super_roots[a]] <= max_prof_leaves) {
                exact_prof_leaves.insert(exact_prof_leaves.end(), prof_leaves.begin() + prof_offsets[a],
                                         prof_leaves.begin() + prof_offsets[a + 1]);
                exact_prof_weights.insert(exact_prof_weights.end(), prof_weights.begin() + prof_offsets[a],
                                          prof_weights.begin() + prof_offsets[a + 1]);
            } else {
                compute_clade_dyadic_profile(super_roots[a], otree, osum, exact_cap, exact_prof_leaves,
                                             exact_prof_weights);
            }
        }
        exact_prof_offsets[K_in] = (int32_t)exact_prof_leaves.size();
        if (has_outgroup) {
            if (outgroup_true_leaves <= (int32_t)outgroup_leaves.size()) {
                exact_prof_leaves.insert(exact_prof_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
                exact_prof_weights.insert(exact_prof_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
            } else {
                compute_outgroup_dyadic_profile(sub_root, otree, osum, exact_cap, exact_prof_leaves,
                                                exact_prof_weights);
            }
            exact_prof_offsets[K] = (int32_t)exact_prof_leaves.size();
        }
    };
    ExactProfileBuilder exact_builder = [&](const int32_t **o, const int32_t **l, const float **w) {
        if (exact_prof_offsets.empty())
            build_exact_prof();
        *o = exact_prof_offsets.data();
        *l = exact_prof_leaves.data();
        *w = exact_prof_weights.data();
    };

    std::vector<int64_t> v_mb(K - 1);
    std::vector<int32_t> perm_mb(K);
    queue_shuffle_rooted(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

    // Skip a solve whose input already found no improvement.
    // Not with the exact check, which also reads the complete profiles.
    std::optional<uint64_t> input_hash;
    if (!(need_exact_prof && settings.exact_check_eta > 0.0)) {
        input_hash = window_input_hash(K, max_iters, v_mb, perm_mb, prof_offsets, prof_leaves, prof_weights);
        if (fruitless.count(*input_hash) != 0)
            return 0.0;
    }
    double dL = dev.solve_window(K, max_iters, p2vbme::MOVE_BATCH_CAP, settings.exact_check_eta, prof_offsets.data(),
                                 prof_leaves.data(), prof_weights.data(), v_mb, perm_mb,
                                 need_exact_prof ? &exact_builder : NULL);
    if (dL == 0.0) {
        fruitless_hash = input_hash;
        return 0.0;
    }

    std::vector<int32_t> c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf;
    int32_t other_rc, rc0, rc1, rc2;
    build_skeleton_metadata(v_mb.data(), K, c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_sz_cf,
                            tin_cf, adj_u_cf, &other_rc, &rc0, &rc1, &rc2);

    // Root the optimized skeleton and orient it away from the root.
    // A subtree window keeps its place: the root is the node where the outgroup attaches, and the outgroup is dropped.
    // The whole-tree window re-roots on its exact centroid edge (closest to N/2 leaves).
    int32_t a_out = -1, u_best = -1, p_best = -1;
    std::vector<int32_t> st_u, st_p;
    st_u.reserve(K - 2);
    st_p.reserve(K - 2);
    if (has_outgroup) {
        for (int32_t a = 0; a < K; ++a) {
            if (perm_mb[a] == K_in) {
                a_out = a;
                break;
            }
        }
        st_u.push_back(adj_u_cf[3 * a_out + 0]);
        st_p.push_back(a_out);
    } else {
        const int32_t root_u_sk = 2 * K - 3;
        std::vector<int32_t> w_sub(2 * K - 2, 0);
        for (int32_t a = 0; a < K; ++a)
            w_sub[a] = summary.leaf_count[super_roots[perm_mb[a]]];
        for (int32_t u = K; u < root_u_sk; ++u)
            w_sub[u] = w_sub[c1_cf[u]] + w_sub[c2_cf[u]];
        u_best = 0;
        int32_t best_diff = std::abs(2 * w_sub[0] - N);
        for (int32_t u = 1; u < root_u_sk; ++u) {
            int32_t diff = std::abs(2 * w_sub[u] - N);
            if (diff < best_diff) {
                best_diff = diff;
                u_best = u;
            }
        }
        p_best = parent_u_cf[u_best];
        if (u_best >= K) {
            st_u.push_back(u_best);
            st_p.push_back(p_best);
        }
        if (p_best >= K) {
            st_u.push_back(p_best);
            st_p.push_back(u_best);
        }
    }
    std::vector<int32_t> child0(2 * K - 2, -1), child1(2 * K - 2, -1);
    for (size_t idx = 0; idx < st_u.size(); ++idx) {
        int32_t u = st_u[idx];
        int32_t p = st_p[idx];
        int32_t b0 = -1, b1 = -1;
        for (int32_t k = 0; k < 3; ++k) {
            int32_t nb = adj_u_cf[3 * u + k];
            if (nb != -1 && nb != p) {
                if (b0 == -1)
                    b0 = nb;
                else
                    b1 = nb;
            }
        }
        child0[u] = b0;
        child1[u] = b1;
        if (b0 >= K) {
            st_u.push_back(b0);
            st_p.push_back(u);
        }
        if (b1 >= K) {
            st_u.push_back(b1);
            st_p.push_back(u);
        }
    }

    std::vector<int32_t> sk_to_full(2 * K - 2, -1);
    for (int32_t a = 0; a < K; ++a) {
        if (a != a_out)
            sk_to_full[a] = super_roots[perm_mb[a]];
    }
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = st_u[K - 3 - s];
        sk_to_full[u_sk] = split_internals[s];
    }
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = st_u[K - 3 - s];
        int32_t full_u = sk_to_full[u_sk];
        int32_t b0 = sk_to_full[child0[u_sk]];
        int32_t b1 = sk_to_full[child1[u_sk]];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
        if (snapshot == NULL) {
            update_one_node_summary(full_u, b0, b1, summary);
        } else {
            summary.leaf_count[full_u] = summary.leaf_count[b0] + summary.leaf_count[b1];
            summary.stand_in[full_u] =
                (summary.leaf_count[b0] <= summary.leaf_count[b1]) ? summary.stand_in[b0] : summary.stand_in[b1];
        }
    }
    if (has_outgroup) {
        if (snapshot == NULL) {
            summary.parent[sub_root] = parent_of_sub_root;
        }
    } else {
        int32_t full_u = split_internals[K - 2]; // == root_r
        int32_t b0 = sk_to_full[u_best];
        int32_t b1 = sk_to_full[p_best];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
        update_one_node_summary(full_u, b0, b1, summary);
        summary.parent[root_r] = -1;
    }
    return dL;
}

namespace {
// The state of one search (run_minibatch_bme), shared by its passes.
struct SearchState {
    Tree &tree; // improved in place
    int32_t root_r;
    int32_t level; // 0 = the main search; k >= 1 = a search inside the sample-and-place start, k levels down
    const SearchSettings &settings;
    Devices &devices;
    TreeSummary summary;                    // of the current tree, rebuilt after windows change it
    double total_mb_delta_L = 0.0;          // how much the search has changed the tree's BME length so far
    std::unordered_set<uint64_t> fruitless; // input hashes of the window solves that found no improvement
};
} // namespace

// Rebuilds the summary of the current tree.
static void rebuild_summary(SearchState &s) { compute_tree_summary(s.tree, s.summary); }

// Progress event of a stage; with_tree: the event carries the current tree (round ends).
static void emit_stage(const SearchState &s, int32_t round, const char *stage, int64_t detail, bool with_tree = false) {
    if (!s.settings.on_event)
        return;
    SearchEvent e;
    e.kind = SearchEvent::Stage;
    e.level = s.level;
    e.n_leaves = s.tree.n_leaves();
    e.round = round;
    e.stage = stage;
    e.detail = detail;
    e.total_dL = s.total_mb_delta_L;
    if (with_tree) {
        e.tree = &s.tree;
    }
    s.settings.on_event(e);
}

// One window on each of window_roots (disjoint clades); stop_at_comp_root: the component roots of the waves, or NULL.
// Several windows always take the snapshot path, whatever the number of devices, so the tree does not depend on it:
// every window reads the tree as it was before them and writes only its own subtree; the gains are summed in root
// order.
// They also only read s.fruitless: their fruitless inputs are added after the loop.
// A single window runs on the first device.
// The search changes the tree only here, so the summary is rebuilt here, whenever a window rewrote the tree.
static void run_windows(SearchState &s, const std::vector<int32_t> &window_roots, const uint8_t *stop_at_comp_root) {
    const int32_t max_prof_leaves = s.settings.profile_cap;
    if (window_roots.size() > 1) {
        const Tree snapshot = s.tree;
        const TreeSummary snapshot_summary = s.summary;
        std::vector<double> window_dL(window_roots.size(), 0.0);
        std::vector<std::optional<uint64_t>> window_fruitless(window_roots.size());
        ParallelErrors errors;
#pragma omp parallel for num_threads((int)s.devices.size()) schedule(dynamic, 1)
        for (int32_t idx = 0; idx < (int32_t)window_roots.size(); ++idx) {
            if (errors.failed())
                continue;
            try {
                int d = omp_get_thread_num();
                int32_t sr = window_roots[idx];
                std::vector<int32_t> out_leaves;
                std::vector<float> out_weights;
                if (sr != s.root_r) {
                    compute_outgroup_dyadic_profile(sr, snapshot, snapshot_summary, max_prof_leaves, out_leaves,
                                                    out_weights);
                }
                window_dL[idx] = run_subtree_minibatch_window(
                    sr, p2vbme::WINDOW_MAX_SWEEPS, s.settings, out_leaves, out_weights, *s.devices[d], s.tree,
                    s.summary, s.fruitless, window_fruitless[idx], stop_at_comp_root, &snapshot, &snapshot_summary);
            } catch (...) {
                errors.capture();
            }
        }
        errors.rethrow();
        for (const std::optional<uint64_t> &h : window_fruitless)
            if (h)
                s.fruitless.insert(*h);
        double sum_dL = 0.0;
        for (double dL : window_dL)
            sum_dL += dL;
        s.total_mb_delta_L += sum_dL;
        if (std::any_of(window_dL.begin(), window_dL.end(), [](double dL) { return dL != 0.0; }))
            rebuild_summary(s); // a window rewrote the tree
    } else {
        std::vector<int32_t> out_leaves;
        std::vector<float> out_weights;
        for (int32_t idx = 0; idx < (int32_t)window_roots.size(); ++idx) {
            int32_t sr = window_roots[idx];
            out_leaves.clear();
            out_weights.clear();
            if (sr != s.root_r) {
                compute_outgroup_dyadic_profile(sr, s.tree, s.summary, max_prof_leaves, out_leaves, out_weights);
            }
            std::optional<uint64_t> fruitless_hash;
            double dL = run_subtree_minibatch_window(sr, p2vbme::WINDOW_MAX_SWEEPS, s.settings, out_leaves, out_weights,
                                                     *s.devices[0], s.tree, s.summary, s.fruitless, fruitless_hash,
                                                     stop_at_comp_root);
            if (fruitless_hash)
                s.fruitless.insert(*fruitless_hash);
            if (dL != 0.0) { // the window rewrote the tree
                s.total_mb_delta_L += dL;
                rebuild_summary(s);
            }
        }
    }
}

// A backbone window: the top of the whole tree as one window (WINDOW_MAX_INTERNALS splits, so K <= 1,500).
// Only in rounds >= 1 and from BACKBONE_MIN_N leaves.
static void run_backbone(SearchState &s, int32_t round) {
    if (round == 0 || s.tree.n_leaves() < p2vbme::BACKBONE_MIN_N)
        return;
    run_windows(s, {s.root_r}, NULL);
    emit_stage(s, round, "backbone", 0);
}

// One pass of the leaf-covering component waves (every leaf is in one window; K <= 1,056).
// Its progress event gives the number of wave levels.
static void run_wave_pass(SearchState &s, int32_t round) {
    std::vector<uint8_t> is_comp_root;
    std::vector<std::vector<int32_t>> waves;
    build_component_waves(s.tree, s.summary, p2vbme::WAVE_COMPONENT_CAP, is_comp_root, waves);
    for (const std::vector<int32_t> &wave : waves)
        run_windows(s, wave, is_comp_root.data());
    emit_stage(s, round, "comp_waves", (int64_t)waves.size());
}

// The multi-scale V-cycle: Full Bottom-Up elevator + Global Backbone + Top-Down cascade.
// Scales, in leaves: up from VCYCLE_BASE_SCALE while below N, then N (the whole tree: a scale >= N selects the root
// alone), then down from N / VCYCLE_SCALE_FACTOR while above VCYCLE_FINEST_SCALE, then VCYCLE_FINEST_SCALE.
// At each scale, one window on each of the largest clades with at most that many leaves (if they are not too small).
static void run_vcycle(SearchState &s, int32_t round) {
    const int32_t n_leaves = s.tree.n_leaves();
    std::vector<int32_t> max_clade_sizes;
    for (int32_t sc = p2vbme::VCYCLE_BASE_SCALE; sc < n_leaves; sc *= p2vbme::VCYCLE_SCALE_FACTOR)
        max_clade_sizes.push_back(sc);
    max_clade_sizes.push_back(n_leaves);
    for (int32_t sc = n_leaves / p2vbme::VCYCLE_SCALE_FACTOR; sc > p2vbme::VCYCLE_FINEST_SCALE;
         sc /= p2vbme::VCYCLE_SCALE_FACTOR)
        max_clade_sizes.push_back(sc);
    if (n_leaves > p2vbme::VCYCLE_FINEST_SCALE)
        max_clade_sizes.push_back(p2vbme::VCYCLE_FINEST_SCALE);
    for (int32_t max_clade_size : max_clade_sizes) {
        // The floor: smaller clades are left to the waves (steps up to 4,096) or to a smaller step.
        const int32_t min_clade_size = (max_clade_size <= p2vbme::VCYCLE_BASE_SCALE)
                                           ? p2vbme::WAVE_COMPONENT_CAP
                                           : (max_clade_size / p2vbme::VCYCLE_SCALE_FACTOR);
        // The window roots: walking down from the root, the largest clades that fit, if above the floor.
        std::vector<int32_t> window_roots;
        std::vector<int32_t> st = {s.root_r};
        while (!st.empty()) {
            int32_t u = st.back();
            st.pop_back();
            if (u < n_leaves)
                continue;
            if (s.summary.leaf_count[u] <= max_clade_size) {
                if (s.summary.leaf_count[u] > min_clade_size)
                    window_roots.push_back(u);
            } else {
                st.push_back(s.tree.c2[u]);
                st.push_back(s.tree.c1[u]);
            }
        }
        if (window_roots.empty())
            continue;
        // One window on each (disjoint clades, so in parallel), then the step's progress event.
        run_windows(s, window_roots, NULL);
        emit_stage(s, round, "scale", max_clade_size);
    }
}

double run_minibatch_bme(Tree &tree, Devices &devices, const SearchSettings &settings, int32_t level,
                         SearchStats *stats) {
    const int32_t n_leaves = tree.n_leaves();
    SearchStats stats_local;
    SearchStats &run_stats = stats ? *stats : stats_local;
    if (settings.rounds <= 0) { // no search, not even the final cleanup pass
        run_stats = SearchStats();
        run_stats.stop_reason = StopReason::None;
        return 0.0;
    }
    // Search state: the tree, its summary, the total change in BME length, and the fruitless input hashes.
    SearchState s{tree, 2 * n_leaves - 2, level, settings, devices, TreeSummary(), 0.0};
    rebuild_summary(s);

    const int n_rounds = settings.rounds;
    double round0_gain = 0.0, round_start_dL = 0.0;
    if (settings.on_event) {
        SearchEvent e;
        e.kind = SearchEvent::Start;
        e.level = level;
        e.n_leaves = n_leaves;
        e.n_devices = (int32_t)devices.size();
        e.rounds_planned = n_rounds;
        settings.on_event(e);
    }
    run_stats.rounds_planned = n_rounds;
    run_stats.rounds_run = 0;
    for (int round = 0; round < n_rounds; ++round) {
        round_start_dL = s.total_mb_delta_L;

        // Pass 1: a backbone window
        run_backbone(s, round);

        // Pass 2A: the leaf-covering component waves
        run_wave_pass(s, round);

        // Pass 2B: a backbone window again, then the multi-scale V-cycle
        if (n_leaves > p2vbme::WAVE_COMPONENT_CAP) {
            run_backbone(s, round);
            run_vcycle(s, round);
        }

        emit_stage(s, round, "round_end", -1, true);
        run_stats.rounds_run = round + 1;
        const double round_gain = round_start_dL - s.total_mb_delta_L; // positive = improvement
        if (round == 0)
            round0_gain = round_gain;
        // Accepted windows always lower L by > WINDOW_MIN_GAIN, so a zero gain means the round changed nothing;
        // rounds >= 1 are deterministic in the tree, so every later regular round would also change nothing.
        // The <= stops there even when stop_tol = 0.
        std::optional<StopReason> reason; // empty = no reason to stop
        if (round >= 1 && round_gain <= settings.stop_tol * (-s.total_mb_delta_L))
            reason = StopReason::Converged;
        else if (settings.deadline > 0.0 && wall_sec() > settings.deadline)
            reason = StopReason::TimeLimit;
        const bool stop = reason.has_value();
        if (stop && round + 1 < n_rounds) {
            run_stats.stop_reason = *reason;
            if (settings.on_event) {
                SearchEvent e;
                e.kind = SearchEvent::Stop;
                e.level = level;
                e.n_leaves = n_leaves;
                e.round = round;
                e.round_gain = round_gain;
                e.round0_gain = round0_gain;
                e.total_dL = s.total_mb_delta_L;
                settings.on_event(e);
            }
            break;
        }
    }

    // Final cleanup pass (N > WAVE_COMPONENT_CAP): the component waves once more, after the last round.
    if (n_leaves > p2vbme::WAVE_COMPONENT_CAP) {
        const int32_t round = run_stats.rounds_run;
        run_wave_pass(s, round);
        emit_stage(s, round, "round_end", -1, true);
    }
    return s.total_mb_delta_L;
}

} // namespace p2vbme
