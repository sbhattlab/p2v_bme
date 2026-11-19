/*
 * search.cpp
 *
 * In brief: each pass of a round picks window roots.
 * A window is a connected set of internal nodes under its root.
 * Its skeleton is a small tree whose tips are pieces (each clade the window does not open, and the outgroup), each
 * summarised by a profile.
 * The waves cover the tree with components, one window per component; the V-cycle and the backbone grow their
 * windows from a root.
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
                             std::vector<int32_t> &v_map_cf, std::vector<int32_t> &parent_u_cf,
                             std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
                             std::vector<int32_t> &adj_u_cf, int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1,
                             int32_t *out_rc2) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;

    c1_cf.assign(R_sk, 0);
    c2_cf.assign(R_sk, 0);
    v_map_cf.assign(K, 0);
    parent_u_cf.assign(M_sk, -1);
    sub_sz_cf.assign(M_sk, 1);
    tin_cf.assign(M_sk, 0);
    adj_u_cf.assign(3 * M_sk, -1);

    std::vector<int32_t> scratch_uf(R_sk);
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
        v_map_cf[y] = b0;
        scratch_uf[x] = m;
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
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;

    out.order.resize(n_leaves - 1);
    out.parent.assign(n_nodes_r, -1);
    int32_t ord_len = 0;
    out.order[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = out.order[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        out.parent[b0] = u;
        out.parent[b1] = u;
        if (b0 >= n_leaves)
            out.order[ord_len++] = b0;
        if (b1 >= n_leaves)
            out.order[ord_len++] = b1;
    }

    out.leaf_count.assign(n_nodes_r, 1);
    out.stand_in.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);
    for (int32_t x = 0; x < n_leaves; ++x) {
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

// The neighbour of u towards the root in the unrooted tree: its parent, or its sibling when the parent is the root
// (the root has two neighbours, so the unrooted tree suppresses it); -1 for the root.
static inline int32_t up_neighbour(int32_t u, const Tree &tree, const TreeSummary &summary) {
    const int32_t root_r = (int32_t)tree.c1.size() - 1;
    const int32_t p = summary.parent[u];
    if (p != root_r)
        return p;
    return (tree.c1[root_r] == u) ? tree.c2[root_r] : tree.c1[root_r];
}

// The dyadic profile of one side of an edge of the unrooted tree: the part that start reaches without passing through
// from (a neighbour of start), each leaf x weighted 2^-d(start, x), so the weights sum to 1.
// profile_cap < 0: every leaf, exact (a plain stack).
// Otherwise the walk opens the heaviest nodes first (a priority queue); once it holds profile_cap entries (if
// > 0), each part not yet opened collapses onto one stand-in leaf with its whole weight: the node's stand_in, or for a
// part reached from below, the stand_in of the other child.
static void directed_dyadic_profile(int32_t start, int32_t from, const Tree &tree, const TreeSummary &summary,
                                    int32_t profile_cap, std::vector<int32_t> &profile_leaves_out,
                                    std::vector<float> &profile_weights_out) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    // A leaf: the side is that leaf alone.
    if (start < n_leaves) {
        profile_leaves_out.push_back(start);
        profile_weights_out.push_back(1.0f);
        return;
    }
    // Exact: visit every node of the side (a stack) and emit every leaf.
    if (profile_cap < 0) {
        struct DirectedStackEntry {
            int32_t node;
            int32_t from;
            double weight;
        };
        std::vector<DirectedStackEntry> st;
        st.push_back({start, from, 1.0});
        while (!st.empty()) {
            DirectedStackEntry top = st.back();
            st.pop_back();
            const int32_t node = top.node;
            // n0, n1: node's two children
            // n2: neighbour above
            int32_t n0 = c1_r[node];
            int32_t n1 = c2_r[node];
            int32_t n2 = up_neighbour(node, tree, summary);
            int32_t nbs[3] = {n0, n1, n2};
            // Each node passes half its weight on to each of its neighbours except the one the walk came from.
            // The order (up, c2, c1) sets the order of the leaves in the profile.
            double hw = 0.5 * top.weight;
            for (int k = 2; k >= 0; --k) {
                int32_t nb = nbs[k];
                if (nb == top.from || nb < 0)
                    continue;
                if (nb < n_leaves) {
                    profile_leaves_out.push_back(nb);
                    profile_weights_out.push_back((float)hw);
                } else {
                    st.push_back({nb, node, hw});
                }
            }
        }
        return;
    }
    // Capped: open the heaviest node first (ties: the smaller id), while the leaves emitted plus the nodes waiting in
    // the queue are fewer than profile_cap (each waiting node becomes at least one entry).
    struct DirectedWeightedNode {
        double weight;
        int32_t node;
        int32_t from;
        bool operator<(const DirectedWeightedNode &o) const {
            if (weight != o.weight)
                return weight < o.weight;
            return node > o.node;
        }
    };
    std::priority_queue<DirectedWeightedNode> pq;
    pq.push({1.0, start, from});
    int32_t start_leaves = (int32_t)profile_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)profile_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (profile_cap > 0 && curr_total >= profile_cap)
            break;
        DirectedWeightedNode top = pq.top();
        pq.pop();
        const int32_t node = top.node;
        // n0, n1: node's two children
        // n2: neighbour above
        int32_t n0 = c1_r[node];
        int32_t n1 = c2_r[node];
        int32_t n2 = up_neighbour(node, tree, summary);
        int32_t nbs[3] = {n0, n1, n2};
        // As above, half the weight to each neighbour except the one the walk came from, here in the order c1, c2, up.
        double hw = 0.5 * top.weight;
        for (int k = 0; k < 3; ++k) {
            int32_t nb = nbs[k];
            if (nb == top.from || nb < 0)
                continue;
            if (nb < n_leaves) {
                profile_leaves_out.push_back(nb);
                profile_weights_out.push_back((float)hw);
            } else {
                pq.push({hw, nb, node});
            }
        }
    }
    // The cap is reached: each part still waiting collapses onto one stand-in leaf that carries its whole weight.
    // Reached from above, the part is the clade below node: its closest leaf.
    // Reached from a child, the part is node's other child and everything above node: the other child's closest leaf.
    while (!pq.empty()) {
        DirectedWeightedNode top = pq.top();
        pq.pop();
        const int32_t node = top.node;
        int32_t rep_leaf;
        if (top.from != c1_r[node] && top.from != c2_r[node]) {
            rep_leaf = summary.stand_in[node];
        } else {
            int32_t sib = (top.from == c1_r[node]) ? c2_r[node] : c1_r[node];
            rep_leaf = summary.stand_in[sib];
        }
        profile_leaves_out.push_back(rep_leaf);
        profile_weights_out.push_back((float)top.weight);
    }
}

// The dyadic profile of the clade below u: each leaf x below u weighted 2^-depth_u(x).
// u is never the root (a window always opens the root).
static void compute_clade_dyadic_profile(int32_t u, const Tree &tree, const TreeSummary &summary, int32_t profile_cap,
                                         std::vector<int32_t> &profile_leaves_out,
                                         std::vector<float> &profile_weights_out) {
    if (u == (int32_t)tree.c1.size() - 1)
        throw std::logic_error("clade profile of the root: a window always opens the root");
    directed_dyadic_profile(u, up_neighbour(u, tree, summary), tree, summary, profile_cap, profile_leaves_out,
                            profile_weights_out);
}

// The dyadic profile of everything outside the clade below window_root (the window's outgroup), seen from the node
// window_root hangs from; empty for the root.
static void compute_outgroup_dyadic_profile(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                                            int32_t profile_cap, std::vector<int32_t> &profile_leaves_out,
                                            std::vector<float> &profile_weights_out) {
    if (window_root == (int32_t)tree.c1.size() - 1 || summary.parent[window_root] == -1)
        return;
    directed_dyadic_profile(up_neighbour(window_root, tree, summary), window_root, tree, summary, profile_cap,
                            profile_leaves_out, profile_weights_out);
}

WindowSkeleton extract_skeleton(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                                const uint8_t *stop_at_comp_root, bool with_outgroup) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    WindowSkeleton skel;
    if (window_root < n_leaves) {
        // a leaf: the window is that single piece
        skel.piece_roots.push_back(window_root);
        const int32_t K = with_outgroup ? 2 : 1;
        skel.c1.assign(2 * K - 1, 0);
        skel.c2.assign(2 * K - 1, 0);
        if (with_outgroup) {
            skel.c1[2 * K - 2] = 0;
            skel.c2[2 * K - 2] = 1;
        }
        return skel;
    }

    // Step 1: open internal nodes from the window's root, largest clades first, up to WINDOW_MAX_INTERNALS, never a
    // leaf or a node marked in stop_at_comp_root.
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
    exp_nodes.reserve(std::min(p2vbme::WINDOW_MAX_INTERNALS, std::max(16, summary.leaf_count[window_root])));
    exp_nodes.push_back({window_root, -1, -1});

    auto is_eligible = [&](int32_t w) -> bool {
        if (w < n_leaves)
            return false;
        if (stop_at_comp_root && stop_at_comp_root[w])
            return false;
        return true;
    };

    std::priority_queue<PqCand> pq;
    int32_t b0_root = c1_r[window_root], b1_root = c2_r[window_root];
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

    // Step 2: the skeleton, by a post-order walk over the opened nodes (left before right, children before parents).
    // A binary tree has one tip more than internal nodes, so the clade pieces number the opened nodes + 1.
    const int32_t K_in = (int32_t)exp_nodes.size() + 1;
    const int32_t K = with_outgroup ? K_in + 1 : K_in;
    skel.c1.assign(2 * K - 1, 0);
    skel.c2.assign(2 * K - 1, 0);
    skel.opened.reserve(K_in - 1);
    skel.piece_roots.reserve(K_in);
    // each finished subtree's skeleton node: a piece i (tip i) or opened node s (internal node K + s)
    struct Frame {
        int32_t exp_idx;
        int32_t state;
        int32_t left_node;
    };
    std::vector<Frame> st;
    st.push_back({0, 0, -1});
    int32_t last_node = -1;
    while (!st.empty()) {
        Frame &f = st.back();
        const ExpNode &en = exp_nodes[f.exp_idx];
        if (f.state == 0) {
            f.state = 1;
            if (en.left_exp == -1) {
                last_node = (int32_t)skel.piece_roots.size();
                skel.piece_roots.push_back(c1_r[en.u]);
            } else {
                st.push_back({en.left_exp, 0, -1});
            }
        } else if (f.state == 1) {
            f.left_node = last_node;
            f.state = 2;
            if (en.right_exp == -1) {
                last_node = (int32_t)skel.piece_roots.size();
                skel.piece_roots.push_back(c2_r[en.u]);
            } else {
                st.push_back({en.right_exp, 0, -1});
            }
        } else {
            const int32_t node = K + (int32_t)skel.opened.size();
            skel.opened.push_back(en.u);
            skel.c1[node] = f.left_node;
            skel.c2[node] = last_node;
            last_node = node;
            st.pop_back();
        }
    }
    // the outgroup: one more tip, joined to the window's root under a new root
    if (with_outgroup) {
        skel.c1[2 * K - 2] = last_node;
        skel.c2[2 * K - 2] = K_in;
    }
    return skel;
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
    const int32_t n_leaves = tree.n_leaves();
    const std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
    is_comp_root.assign(n_nodes_r, 0);
    waves.clear();
    if (n_leaves < p2vbme::WINDOW_MIN_K)
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
                if (c >= n_leaves) {
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
    for (int32_t x = 0; x < n_leaves; ++x)
        direct_leaves[x] = 1;
    int32_t max_lvl = 0;
    for (int32_t idx = (int32_t)summary.order.size() - 1; idx >= 0; --idx) {
        int32_t u = summary.order[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t l1 = (c1 >= n_leaves && is_comp_root[c1]) ? (comp_lvl[c1] + 1) : max_ch_lvl[c1];
        int32_t l2 = (c2 >= n_leaves && is_comp_root[c2]) ? (comp_lvl[c2] + 1) : max_ch_lvl[c2];
        int32_t dl1 = (c1 >= n_leaves && is_comp_root[c1]) ? 0 : direct_leaves[c1];
        int32_t dl2 = (c2 >= n_leaves && is_comp_root[c2]) ? 0 : direct_leaves[c2];
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

// Compute all-pairs unrooted topological edge distances dist_out[a * K + b] between the pieces a, b in
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
// exact_builder is supplied) an exact full-objective acceptance check on all changed piece pairs involving capped
// clades or capped outside profiles.
// The device does the arithmetic.
double solve_window_host(int32_t K, int32_t max_iters, int32_t max_batch_cap, double exact_check_eta,
                         const int32_t *profile_offsets, const int32_t *profile_leaves, const float *profile_weights,
                         std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                         const ExactProfileBuilder *exact_builder, WindowKernels &kernels) {
    // The complete (uncapped) profiles, in CSR form, filled by exact_builder only if the exact check runs.
    const int32_t *exact_profile_offsets = NULL; // where each piece's entries start
    const int32_t *exact_profile_leaves = NULL;  // the entries' leaves
    const float *exact_profile_weights = NULL;   // the entries' weights
    const int32_t M_sk = 2 * K - 2;              // nodes of the unrooted skeleton (the rooted one without its root)
    const int32_t root_u_sk = M_sk - 1;          // the node the unrooted skeleton hangs from (2K - 3)
    const std::vector<int64_t> init_v_cf = v_cf; // the skeleton before the solve: kept if nothing improves
    const std::vector<int32_t> init_perm = perm; // its leaf permutation

    // The K x K distances between the pieces, from their profiles: computed once per window, kept for every sweep.
    kernels.distances(K, profile_offsets, profile_leaves, profile_weights);

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
        // Step 1: decode the skeleton (v_cf) into the arrays the kernels and the moves read.
        build_skeleton_metadata(v_cf.data(), K, sk.c1, sk.c2, sk.v_map, sk.parent_u, sk.sub_size, sk.tin, sk.adj_u,
                                &sk.other_rc, &sk.rc0, &sk.rc1, &sk.rc2);
        const int32_t other_rc = sk.other_rc;
        std::vector<int32_t> &v_map_cf = sk.v_map, &parent_u_cf = sk.parent_u, &sub_size_cf = sk.sub_size;
        std::vector<int32_t> &tin_cf = sk.tin, &adj_u_cf = sk.adj_u;

        // Step 2: score the skeleton: its BME length (Pauplin's formula), from the K x K distances through the average
        // distances between subtrees, which the kernels keep for step 4.
        curr_L = kernels.score(sk, perm);
        // Step 3: judge the last batch: undo it and halve the batch cap if L did not fall, else raise the cap; keep the
        // best skeleton so far.
        if (iter == 0) {
            // the first sweep: the window's starting length
            init_L = curr_L;
            best_L = curr_L;
            prev_L = curr_L;
        } else if (batch_pending && curr_L >= prev_L - 1e-8) {
            // the last batch did not lower L (Algorithm S1, line 4): restore the previous tree and use smaller batches.
            // A single move that fails is below the resolution of the scores, so the window is converged.
            v_cf = prev_v_cf;
            perm = prev_perm;
            batch_pending = false;
            if (cur_batch_cap <= 1)
                break;
            cur_batch_cap = std::max(1, cur_batch_cap / 2);
            continue;
        } else if (batch_pending) {
            // the last batch lowered L: allow larger batches
            cur_batch_cap = std::min(max_batch_cap, cur_batch_cap + 16);
        }
        // (no batch pending: the skeleton was only relabelled, after a flip or an undo, so there is nothing to judge)
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
        // Step 4: find the best regraft of every pruned subtree, with its BME change.
        kernels.best_moves(sk, h_best_delta, h_best_j);

        // Step 5: keep the improving moves, best first.
        std::vector<MoveCand> cands;
        // A row's best move is a candidate only if its change is below this numerical zero (the scores may be single
        // precision).
        // A batch that does not lower L is undone in step 3, and max_iters bounds the sweeps.
        const float eps_move_thresh = -1e-6f;
        for (int32_t i = 1; i < K - 1; ++i) {
            if (h_best_delta[i] < eps_move_thresh && h_best_j[i] >= 0 && h_best_j[i] < root_u_sk) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        // None: relabel the skeleton once with the other labelling of its cherries (its rows give other moves), then
        // stop.
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

        // Step 6: apply a batch of moves to the unrooted skeleton (adj_u).
        // Going down the candidates, best first, a move joins the batch if its path (from the pruned subtree to the
        // regraft edge) shares no node with the moves already in it; at most cur_batch_cap moves.
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
        // No candidate fit the checks: apply the single best move.
        if (n_applied == 0) {
            int32_t i = cands[0].i, dst_u = cands[0].j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            int32_t dst_v = parent_u_cf[dst_u];
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
        }
        batch_pending = true;

        // Step 7: re-encode the moved skeleton as a cherry-first vector (the labelling alternates each sweep); the next
        // sweep scores it.
        int32_t r_nb = adj_u_cf[3 * root_u_sk + 0];
        relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, r_nb, iter & 1, v_cf, perm);
    }

    if (best_L >= init_L - p2vbme::WINDOW_MIN_GAIN) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }

    const bool have_exact = (exact_profile_offsets != NULL) || (exact_builder != NULL && *exact_builder);
    if (have_exact && best_L - init_L > -exact_check_eta) {
        if (exact_profile_offsets == NULL)
            (*exact_builder)(&exact_profile_offsets, &exact_profile_leaves, &exact_profile_weights);
        std::vector<uint8_t> is_capped(K, 0);
        bool any_capped = false;
        for (int32_t a = 0; a < K; ++a) {
            int32_t cap_len = profile_offsets[a + 1] - profile_offsets[a];
            int32_t ex_len = exact_profile_offsets[a + 1] - exact_profile_offsets[a];
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
                    int32_t lu = exact_profile_offsets[u + 1] - exact_profile_offsets[u];
                    int32_t lv = exact_profile_offsets[v + 1] - exact_profile_offsets[v];
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
                kernels.exact_distances(pair_a, pair_b, exact_profile_offsets, exact_profile_leaves,
                                        exact_profile_weights, h_pair_exact_D);

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

// Writes a window's solved skeleton (v, perm) back into the tree: decodes it, roots it (a subtree window where its
// outgroup attaches, the whole-tree window on its centroid edge), and gives the window's opened nodes their new
// children.
// It also patches the summary of those nodes (in_snapshot: only leaf_count and stand_in, as parallel windows must
// leave summary.parent alone).
static void write_back_skeleton(const std::vector<int64_t> &v_mb, const std::vector<int32_t> &perm_mb,
                                const WindowSkeleton &skel, int32_t window_root, int32_t parent_of_window_root,
                                bool in_snapshot, Tree &tree, TreeSummary &summary) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t root_r = 2 * n_leaves - 2;
    const int32_t K = (int32_t)v_mb.size() + 1;
    const std::vector<int32_t> &piece_roots = skel.piece_roots;
    const int32_t K_in = (int32_t)piece_roots.size();
    const bool has_outgroup = window_root != root_r;
    std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;

    // Decode the vector back to a skeleton + kernel extras
    std::vector<int32_t> c1_cf, c2_cf, v_map_cf, parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf;
    int32_t other_rc, rc0, rc1, rc2;
    build_skeleton_metadata(v_mb.data(), K, c1_cf, c2_cf, v_map_cf, parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf, &other_rc,
                            &rc0, &rc1, &rc2);

    // Step 1: choose the root.
    // A subtree window: the node its outgroup tip hangs from, so the subtree keeps its place.
    // The whole-tree window: its centroid edge, the split closest to N/2 leaves (pieces weighed by their leaf counts).
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
            w_sub[a] = summary.leaf_count[piece_roots[perm_mb[a]]];
        for (int32_t u = K; u < root_u_sk; ++u)
            w_sub[u] = w_sub[c1_cf[u]] + w_sub[c2_cf[u]];
        u_best = 0;
        int32_t best_diff = std::abs(2 * w_sub[0] - n_leaves);
        for (int32_t u = 1; u < root_u_sk; ++u) {
            int32_t diff = std::abs(2 * w_sub[u] - n_leaves);
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
    // Step 2: orient the skeleton away from that root, breadth first: each internal node's two children.
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

    // Step 3: map the skeleton's nodes to tree nodes: each tip to its piece's root, and the internal nodes to the
    // window's opened nodes, children before parents (so window_root stays at the top).
    std::vector<int32_t> sk_to_full(2 * K - 2, -1);
    for (int32_t a = 0; a < K; ++a) {
        if (a != a_out)
            sk_to_full[a] = piece_roots[perm_mb[a]];
    }
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = st_u[K - 3 - s];
        sk_to_full[u_sk] = skel.opened[s];
    }
    // Step 4: give the opened nodes their new children (and patch their summary).
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = st_u[K - 3 - s];
        int32_t full_u = sk_to_full[u_sk];
        int32_t b0 = sk_to_full[child0[u_sk]];
        int32_t b1 = sk_to_full[child1[u_sk]];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
        if (!in_snapshot) {
            update_one_node_summary(full_u, b0, b1, summary);
        } else {
            summary.leaf_count[full_u] = summary.leaf_count[b0] + summary.leaf_count[b1];
            summary.stand_in[full_u] =
                (summary.leaf_count[b0] <= summary.leaf_count[b1]) ? summary.stand_in[b0] : summary.stand_in[b1];
        }
    }
    // Step 5: the top.
    // A subtree window: window_root gets its old parent back (the outgroup is dropped).
    // The whole-tree window: the spare opened node (the tree root) joins the centroid edge's two ends.
    if (has_outgroup) {
        if (!in_snapshot) {
            summary.parent[window_root] = parent_of_window_root;
        }
    } else {
        int32_t full_u = skel.opened[K - 2]; // == root_r
        int32_t b0 = sk_to_full[u_best];
        int32_t b1 = sk_to_full[p_best];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
        update_one_node_summary(full_u, b0, b1, summary);
        summary.parent[root_r] = -1;
    }
}

// Optimize the window below window_root and write the result into tree and summary.
// The window collapses clades into K pieces, each with a dyadic-weighted leaf profile, and optimizes them as an
// unrooted BME tree.
// A subtree window (window_root != root_r) also sees the rest of the tree, as one extra piece, the outgroup,
// because BME sums over all leaf pairs: moves inside the subtree change its distances to the leaves outside.
// Its result is rooted where the outgroup attaches, so the subtree keeps its place, and the outgroup is dropped.
// The whole-tree window (window_root == root_r) has no outgroup and BME does not fix a root, so it re-roots on the
// centroid edge.
// Windows of one wave run in parallel: they pass a snapshot of the tree and its summary (snapshot, snapshot_summary),
// read their profiles from it, and leave summary.parent alone (the caller recomputes the summary after the wave).
// fruitless: the input hashes (window_input_hash) of this search's solves that found no improvement.
// The solve is deterministic, so a window with one of these inputs would again find none: it is skipped.
// fruitless_hash: set to the input hash when the solve runs and finds no improvement, for the caller to add to
// fruitless.

static double run_subtree_minibatch_window(int32_t window_root, int32_t max_iters, const SearchSettings &settings,
                                           const std::vector<int32_t> &outgroup_leaves,
                                           const std::vector<float> &outgroup_weights, Device &dev, Tree &tree,
                                           TreeSummary &summary, const std::unordered_set<uint64_t> &fruitless,
                                           std::optional<uint64_t> &fruitless_hash,
                                           const uint8_t *stop_at_comp_root = NULL, const Tree *snapshot = NULL,
                                           const TreeSummary *snapshot_summary = NULL) {
    const int32_t n_leaves = tree.n_leaves();
    std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;
    const int32_t root_r = 2 * n_leaves - 2;
    const int32_t profile_cap = settings.profile_cap;
    // A subtree window also sees the rest of the tree, as one extra piece (the outgroup).
    const bool has_outgroup = window_root != root_r;
    // Decide what a window contains and return its skeleton
    const WindowSkeleton skel = extract_skeleton(window_root, tree, summary, stop_at_comp_root, has_outgroup);
    const std::vector<int32_t> &piece_roots = skel.piece_roots;
    // The root is alone in its wave, so the whole-tree window never gets a snapshot.
    if (!has_outgroup && snapshot != NULL)
        throw std::logic_error("whole-tree window: unexpected snapshot (the root is alone in its wave)");
    const int32_t K_in = (int32_t)piece_roots.size();
    const int32_t K = has_outgroup ? K_in + 1 : K_in;
    // Fewer than 4 pieces: one possible shape, so nothing to move
    // > max_window: device cannot handle the window
    if (K < p2vbme::WINDOW_MIN_K || K > dev.max_window())
        return 0.0;

    int32_t parent_of_window_root = summary.parent[window_root];

    const Tree &otree = (snapshot != NULL) ? *snapshot : tree;
    const TreeSummary &osum = (snapshot_summary != NULL) ? *snapshot_summary : summary;

    // Build the window's K profiles in one CSR layout
    std::vector<int32_t> profile_offsets(K + 1, 0), profile_leaves;
    std::vector<float> profile_weights;
    profile_leaves.reserve(4 * K);
    profile_weights.reserve(4 * K);
    bool need_exact_profiles = false;
    for (int32_t a = 0; a < K_in; ++a) {
        profile_offsets[a] = (int32_t)profile_leaves.size();
        compute_clade_dyadic_profile(piece_roots[a], otree, osum, profile_cap, profile_leaves, profile_weights);
        if (profile_cap > 0 && summary.leaf_count[piece_roots[a]] > profile_cap) {
            need_exact_profiles = true;
        }
    }
    profile_offsets[K_in] = (int32_t)profile_leaves.size();
    int32_t outgroup_true_leaves = n_leaves - summary.leaf_count[window_root];
    if (has_outgroup) {
        profile_leaves.insert(profile_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
        profile_weights.insert(profile_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
        profile_offsets[K] = (int32_t)profile_leaves.size();
        if (profile_cap > 0 && outgroup_true_leaves > (int32_t)outgroup_leaves.size()) {
            need_exact_profiles = true;
        }
    }

    // Exact profile builder
    std::vector<int32_t> exact_profile_offsets, exact_profile_leaves;
    std::vector<float> exact_profile_weights;
    auto build_exact_prof = [&]() {
        const bool cap_exact = n_leaves >= p2vbme::EXACT_PROFILE_CAP_MIN_N;
        const int32_t exact_cap = cap_exact ? p2vbme::EXACT_PROFILE_CAP : -1;
        exact_profile_offsets.resize(K + 1, 0);
        exact_profile_leaves.reserve(cap_exact ? (size_t)(K * 256) : (size_t)n_leaves);
        exact_profile_weights.reserve(cap_exact ? (size_t)(K * 256) : (size_t)n_leaves);
        for (int32_t a = 0; a < K_in; ++a) {
            exact_profile_offsets[a] = (int32_t)exact_profile_leaves.size();
            if (profile_cap <= 0 || summary.leaf_count[piece_roots[a]] <= profile_cap) {
                exact_profile_leaves.insert(exact_profile_leaves.end(), profile_leaves.begin() + profile_offsets[a],
                                            profile_leaves.begin() + profile_offsets[a + 1]);
                exact_profile_weights.insert(exact_profile_weights.end(), profile_weights.begin() + profile_offsets[a],
                                             profile_weights.begin() + profile_offsets[a + 1]);
            } else {
                compute_clade_dyadic_profile(piece_roots[a], otree, osum, exact_cap, exact_profile_leaves,
                                             exact_profile_weights);
            }
        }
        exact_profile_offsets[K_in] = (int32_t)exact_profile_leaves.size();
        if (has_outgroup) {
            if (outgroup_true_leaves <= (int32_t)outgroup_leaves.size()) {
                exact_profile_leaves.insert(exact_profile_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
                exact_profile_weights.insert(exact_profile_weights.end(), outgroup_weights.begin(),
                                             outgroup_weights.end());
            } else {
                compute_outgroup_dyadic_profile(window_root, otree, osum, exact_cap, exact_profile_leaves,
                                                exact_profile_weights);
            }
            exact_profile_offsets[K] = (int32_t)exact_profile_leaves.size();
        }
    };
    ExactProfileBuilder exact_builder = [&](const int32_t **o, const int32_t **l, const float **w) {
        if (exact_profile_offsets.empty())
            build_exact_prof();
        *o = exact_profile_offsets.data();
        *l = exact_profile_leaves.data();
        *w = exact_profile_weights.data();
    };

    // Convert skeleton into phylo2vec form
    std::vector<int64_t> v_mb(K - 1);
    std::vector<int32_t> perm_mb(K);
    queue_shuffle_rooted(skel.c1.data(), skel.c2.data(), K, 0, v_mb.data(), perm_mb.data());

    // Skip a solve whose input already found no improvement.
    // Not with the exact check, which also reads the complete profiles.
    std::optional<uint64_t> input_hash;
    if (!(need_exact_profiles && settings.exact_check_eta > 0.0)) {
        input_hash = window_input_hash(K, max_iters, v_mb, perm_mb, profile_offsets, profile_leaves, profile_weights);
        if (fruitless.count(*input_hash) != 0)
            return 0.0;
    }
    // Core device-powered search through the skeleton with SPR sweeps
    // Returns the gain, and the new new skeleton in v_mb, perm_mb
    double dL = solve_window_host(K, max_iters, p2vbme::MOVE_BATCH_CAP, settings.exact_check_eta,
                                  profile_offsets.data(), profile_leaves.data(), profile_weights.data(), v_mb, perm_mb,
                                  need_exact_profiles ? &exact_builder : NULL, *dev.window_kernels());
    if (dL == 0.0) {
        fruitless_hash = input_hash;
        return 0.0;
    }

    // Write the solved skeleton back into the tree.
    write_back_skeleton(v_mb, perm_mb, skel, window_root, parent_of_window_root, snapshot != NULL, tree, summary);
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
    const int32_t profile_cap = s.settings.profile_cap;
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
                int32_t window_root = window_roots[idx];
                std::vector<int32_t> out_leaves;
                std::vector<float> out_weights;
                if (window_root != s.root_r) {
                    // Build the profile of everything outside the window's clade
                    // so the window can include the rest of the trree as one extra piece (the outgroup).
                    compute_outgroup_dyadic_profile(window_root, snapshot, snapshot_summary, profile_cap, out_leaves,
                                                    out_weights);
                }
                // Turn the window into a skeleton of pieces with profiles
                // Optimize it
                // Write results back into the tree if it got shorter
                window_dL[idx] = run_subtree_minibatch_window(
                    window_root, p2vbme::WINDOW_MAX_SWEEPS, s.settings, out_leaves, out_weights, *s.devices[d], s.tree,
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
            int32_t window_root = window_roots[idx];
            out_leaves.clear();
            out_weights.clear();
            if (window_root != s.root_r) {
                // Build the profile of everything outside the window's clade
                // so the window can include the rest of the trree as one extra piece (the outgroup).
                compute_outgroup_dyadic_profile(window_root, s.tree, s.summary, profile_cap, out_leaves, out_weights);
            }
            std::optional<uint64_t> fruitless_hash;
            // Turn the window into a skeleton of pieces with profiles
            // Optimize it
            // Write results back into the tree if it got shorter
            double dL = run_subtree_minibatch_window(window_root, p2vbme::WINDOW_MAX_SWEEPS, s.settings, out_leaves,
                                                     out_weights, *s.devices[0], s.tree, s.summary, s.fruitless,
                                                     fruitless_hash, stop_at_comp_root);
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
    // Find the largest clade sizes of the V-cycle's windows at each step
    // Ex. for N = 100,000: 4,096, 16,384, 65,536, 100,000, 25,000, 6,250, 3,200.
    // up from VCYCLE_BASE_SCALE by VCYCLE_SCALE_FACTOR while below n_leaves
    for (int32_t sc = p2vbme::VCYCLE_BASE_SCALE; sc < n_leaves; sc *= p2vbme::VCYCLE_SCALE_FACTOR)
        max_clade_sizes.push_back(sc);
    // Then n_leaves
    max_clade_sizes.push_back(n_leaves);
    // Down from n_leaves / VCYCLE_SCALE_FACTOR by VCYCLE_SCALE_FACTOR while above VCYCLE_FINEST_SCALE
    for (int32_t sc = n_leaves / p2vbme::VCYCLE_SCALE_FACTOR; sc > p2vbme::VCYCLE_FINEST_SCALE;
         sc /= p2vbme::VCYCLE_SCALE_FACTOR)
        max_clade_sizes.push_back(sc);
    // Then VCYCLE_FINEST_SCALE, if the tree has more leaves
    if (n_leaves > p2vbme::VCYCLE_FINEST_SCALE)
        max_clade_sizes.push_back(p2vbme::VCYCLE_FINEST_SCALE);

    // Main loop
    for (int32_t max_clade_size : max_clade_sizes) {
        // The floor: smaller clades are left to the waves (steps up to 4,096) or to a smaller step.
        const int32_t min_clade_size = (max_clade_size <= p2vbme::VCYCLE_BASE_SCALE)
                                           ? p2vbme::WAVE_COMPONENT_CAP
                                           : (max_clade_size / p2vbme::VCYCLE_SCALE_FACTOR);
        // Pick the step's window roots: walking down from the root, the largest clades that fit, if above the floor.
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
    // Main loop
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
