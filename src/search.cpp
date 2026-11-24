/*
 * search.cpp
 *
 * In brief: each pass of a round picks window roots.
 * A window is a connected set of internal nodes under its root.
 * Its skeleton is a small tree whose tips are pieces (each clade the window does not open, and the outgroup), each
 * summarised by its representatives.
 * The waves cover the tree with components, one window per component; the V-cycle and the backbone grow their
 * windows from a root.
 *
 * The minibatch BME search (host side): Phylo2Vec encoding of skeletons, tree summaries, clade and outside
 * representatives, window extraction, the leaf-covering waves and the rounds. Window solves run on the devices.
 */
#include "search.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parallel_errors.hpp"
#include "phylo2vec.hpp"
#include "random.hpp"
#include "timing.hpp"

namespace p2vbme {

// Decodes a skeleton's ordered vector into its tables, which the scoring, the moves and the write-back read.
void to_skeleton_tables(const std::vector<int64_t> &v, SkeletonTables &sk) {
    const int32_t K = (int32_t)v.size() + 1; // pieces = tips of the skeleton
    const int32_t R_sk = 2 * K - 1;          // nodes of the rooted skeleton (root 2K - 2)
    const int32_t M_sk = 2 * K - 2;          // nodes of the unrooted skeleton (the rooted one without its root)
    const int32_t root_u_sk = M_sk - 1;      // the node the unrooted skeleton hangs from (2K - 3)

    sk.c1.assign(R_sk, 0);
    sk.c2.assign(R_sk, 0);
    sk.v_map.assign(K, 0);
    sk.parent_u.assign(M_sk, -1);
    sk.sub_size.assign(M_sk, 1);
    sk.tin.assign(M_sk, 0);
    sk.adj_u.assign(3 * M_sk, -1);

    // Step 1: decode the ordered vector into the rooted skeleton (c1, c2).
    // v_map[i + 1] is the subtree row i merges in, the c1 side of the node it makes (2K - 2 - i): the subtree its
    // moves prune.
    to_skeleton(v.data(), K, sk.c1.data(), sk.c2.data());
    for (int32_t y = 1; y < K; ++y)
        sk.v_map[y] = sk.c1[2 * K - 1 - y];
    // Step 2: unroot it: drop the root 2K - 2 and join its two children by an edge.
    // One child is node 2K - 3 (made by row 1), which becomes the node the unrooted skeleton hangs from; the other
    // (the smaller id) becomes its third neighbour.
    // The hanging node's neighbours: its two rooted children (rc0, rc1) and that third neighbour (rc2).
    const int32_t root_a = sk.c1[R_sk - 1], root_b = sk.c2[R_sk - 1]; // the children of the root 2K - 2
    sk.rc0 = sk.c1[root_u_sk];
    sk.rc1 = sk.c2[root_u_sk];
    sk.rc2 = (root_a < root_b) ? root_a : root_b;

    // Step 3: set the parents and subtree sizes (in nodes), children before parents, and the internal nodes'
    // adjacency (two children, then the parent).
    for (int32_t u = K; u < root_u_sk; ++u) {
        int32_t b0 = sk.c1[u], b1 = sk.c2[u];
        sk.parent_u[b0] = u;
        sk.parent_u[b1] = u;
        sk.sub_size[u] = 1 + sk.sub_size[b0] + sk.sub_size[b1];
        sk.adj_u[3 * u + 0] = b0;
        sk.adj_u[3 * u + 1] = b1;
    }
    sk.sub_size[root_u_sk] = M_sk; // the root's Euler interval spans every node: it is an ancestor of all
    sk.parent_u[sk.rc0] = root_u_sk;
    sk.parent_u[sk.rc1] = root_u_sk;
    sk.parent_u[sk.rc2] = root_u_sk;
    sk.parent_u[root_u_sk] = -1;
    // Step 4: fill in the rest of the unrooted adjacency (3 slots per node, -1 if unused): a tip has only its parent,
    // an internal node gets its parent in the third slot, the hanging node its three neighbours.
    for (int32_t u = 0; u < K; ++u)
        sk.adj_u[3 * u + 0] = sk.parent_u[u];
    for (int32_t u = K; u < root_u_sk; ++u)
        sk.adj_u[3 * u + 2] = sk.parent_u[u];
    sk.adj_u[3 * root_u_sk + 0] = sk.rc0;
    sk.adj_u[3 * root_u_sk + 1] = sk.rc1;
    sk.adj_u[3 * root_u_sk + 2] = sk.rc2;

    // Step 5: number the nodes in preorder (tin), walking from the hanging node, parents before children (ids
    // decreasing).
    // With sub_size they make "x is under y" one comparison: tin[y] <= tin[x] < tin[y] + sub_size[y].
    sk.tin[root_u_sk] = 0;
    sk.tin[sk.rc0] = 1;
    sk.tin[sk.rc1] = 1 + sk.sub_size[sk.rc0];
    sk.tin[sk.rc2] = 1 + sk.sub_size[sk.rc0] + sk.sub_size[sk.rc1];
    for (int32_t u = root_u_sk - 1; u >= K; --u) {
        int32_t b0 = sk.c1[u], b1 = sk.c2[u];
        sk.tin[b0] = sk.tin[u] + 1;
        sk.tin[b1] = sk.tin[u] + 1 + sk.sub_size[b0];
    }
}

// Rebuilds the summary of the whole tree (BFS order, parents, leaf counts, stand-ins) from scratch.
void compute_tree_summary(const Tree &tree, TreeSummary &out) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;

    // Pass 1, top-down: walk from the root breadth first; order lists the internal nodes, parents before children, and
    // each node's parent is set on the way.
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

    // Pass 2, bottom-up: order read backwards, children before parents; each node's leaf count is its children's sum,
    // and its stand-in is the leaf nearest to it (the shallowest below it, c1 on ties).
    out.leaf_count.assign(n_nodes_r, 1);
    out.stand_in.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0); // depth of the nearest leaf below each node
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

// The dyadic representatives of one side of an edge of the unrooted tree: the part that start reaches without passing
// through from (a neighbour of start), each leaf x weighted 2^-d(start, x), so the weights sum to 1. representative_cap
// < 0: every leaf, exact (a plain stack). Otherwise the walk opens the heaviest nodes first (a priority queue); once it
// holds representative_cap entries (if > 0), each part not yet opened collapses onto one stand-in leaf with its whole
// weight: the node's stand_in, or for a part reached from below, the stand_in of the other child.
static void directed_dyadic_representatives(int32_t start, int32_t from, const Tree &tree, const TreeSummary &summary,
                                            int32_t representative_cap, std::vector<int32_t> &representative_leaves_out,
                                            std::vector<float> &representative_weights_out) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    // A leaf: the side is that leaf alone.
    if (start < n_leaves) {
        representative_leaves_out.push_back(start);
        representative_weights_out.push_back(1.0f);
        return;
    }
    // Exact: visit every node of the side (a stack) and emit every leaf.
    if (representative_cap < 0) {
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
            // The order (up, c2, c1) sets the order of the representatives.
            double hw = 0.5 * top.weight;
            for (int k = 2; k >= 0; --k) {
                int32_t nb = nbs[k];
                if (nb == top.from || nb < 0)
                    continue;
                if (nb < n_leaves) {
                    representative_leaves_out.push_back(nb);
                    representative_weights_out.push_back((float)hw);
                } else {
                    st.push_back({nb, node, hw});
                }
            }
        }
        return;
    }
    // Capped: open the heaviest node first (ties: the smaller id), while the leaves emitted plus the nodes waiting in
    // the queue are fewer than representative_cap (each waiting node becomes at least one entry).
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
    int32_t start_leaves = (int32_t)representative_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)representative_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (representative_cap > 0 && curr_total >= representative_cap)
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
                representative_leaves_out.push_back(nb);
                representative_weights_out.push_back((float)hw);
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
        representative_leaves_out.push_back(rep_leaf);
        representative_weights_out.push_back((float)top.weight);
    }
}

// The dyadic representatives of the clade below u: each leaf x below u weighted 2^-depth_u(x).
// u is never the root (a window always opens the root).
static void compute_clade_dyadic_representatives(int32_t u, const Tree &tree, const TreeSummary &summary,
                                                 int32_t representative_cap,
                                                 std::vector<int32_t> &representative_leaves_out,
                                                 std::vector<float> &representative_weights_out) {
    if (u == (int32_t)tree.c1.size() - 1)
        throw std::logic_error("clade representatives of the root: a window always opens the root");
    directed_dyadic_representatives(u, up_neighbour(u, tree, summary), tree, summary, representative_cap,
                                    representative_leaves_out, representative_weights_out);
}

// The dyadic representatives of everything outside the clade below window_root (the window's outgroup), seen from the
// node window_root hangs from; empty for the root.
static void compute_outgroup_dyadic_representatives(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                                                    int32_t representative_cap,
                                                    std::vector<int32_t> &representative_leaves_out,
                                                    std::vector<float> &representative_weights_out) {
    if (window_root == (int32_t)tree.c1.size() - 1 || summary.parent[window_root] == -1)
        return;
    directed_dyadic_representatives(up_neighbour(window_root, tree, summary), window_root, tree, summary,
                                    representative_cap, representative_leaves_out, representative_weights_out);
}

// Chooses a window's pieces and the nodes it opens, and encodes its skeleton as an ordered vector.
Skeleton extract_skeleton(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                          const uint8_t *stop_at_comp_root, bool with_outgroup) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t *c1_r = tree.c1.data(), *c2_r = tree.c2.data();
    Skeleton skel;
    if (window_root < n_leaves) {
        // a leaf: the window is that single piece
        skel.piece_roots.push_back(window_root);
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
    std::vector<int32_t> skel_c1(2 * K - 1, 0), skel_c2(2 * K - 1, 0); // the skeleton as a rooted tree (Tree layout)
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
            skel_c1[node] = f.left_node;
            skel_c2[node] = last_node;
            last_node = node;
            st.pop_back();
        }
    }
    // the outgroup: one more tip, joined to the window's root under a new root
    if (with_outgroup) {
        skel_c1[2 * K - 2] = last_node;
        skel_c2[2 * K - 2] = K_in;
    }
    // Step 3: encode the skeleton as an ordered vector (queue shuffle), for windows large enough to be solved.
    if (K >= p2vbme::WINDOW_MIN_K) {
        skel.v.resize(K - 1);
        skel.perm.resize(K);
        queue_shuffle_rooted(skel_c1.data(), skel_c2.data(), K, false, skel.v.data(), skel.perm.data());
    }
    return skel;
}

// Cuts the tree into components and schedules them as waves of windows.
// A component is a connected set of internal nodes: its root and the nodes below it, down to the roots of the
// components below.
// For example (sizes not to scale):
//
//              r                A = {r, x, y}: the root's component, level 1
//            /   \              x, y: A's other nodes
//           x     y
//          / \   / \            l1, l2: A's own leaves
//         b  l1 l2  c           b, c: the roots of components B and C, level 0
//        / \       / \          b's and c's children: in B and C
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

// Replaces node u's neighbour old_nb by new_nb in adj_u (3 slots per node).
static void replace_neighbour(int32_t *adj_u, int32_t u, int32_t old_nb, int32_t new_nb) {
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * u + k] == old_nb) {
            adj_u[3 * u + k] = new_nb;
            return;
        }
    }
}

// One SPR move on the unrooted adjacency adj_u: cut the subtree at prune_node off together with the node it hangs from
// (anchor_node), and regraft it into the edge between dest_a and dest_b; anchor_node becomes the new node on that edge.
//
//          n0                                 n0
//          |                                  |
//     anchor_node -- prune_node      ->       n1        dest_b
//          |                                            |
//          n1        dest_b                        anchor_node -- prune_node
//                    |                                  |
//                    dest_a                             dest_a
void apply_unrooted_spr(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_a, int32_t dest_b) {
    // Step 1: prune: join the anchor's two other neighbours (n0, n1) directly, which takes the anchor out of the tree.
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
    // n0's slot that held the anchor now holds n1, and n1's holds n0
    replace_neighbour(adj_u, n0, anchor_node, n1);
    replace_neighbour(adj_u, n1, anchor_node, n0);
    // A regraft edge that ended at the anchor is taken to end at n0 instead (never in the window solve: the kernels
    // leave out the edges at the pruning point).
    if (dest_a == anchor_node)
        dest_a = n0;
    if (dest_b == anchor_node)
        dest_b = n0;

    // Step 2: regraft: put the anchor into the edge between dest_a and dest_b (each end's slot that held the other end
    // now holds the anchor), and give the anchor its three neighbours: the pruned subtree, dest_a and dest_b.
    replace_neighbour(adj_u, dest_a, dest_b, anchor_node);
    replace_neighbour(adj_u, dest_b, dest_a, anchor_node);
    adj_u[3 * anchor_node + 0] = prune_node;
    adj_u[3 * anchor_node + 1] = dest_a;
    adj_u[3 * anchor_node + 2] = dest_b;
}

// Orients the unrooted skeleton adj_u (3 slots per node, tips 0..K-1) away from the start nodes, breadth first.
// starts: the internal nodes to begin with, each with the neighbour it is reached from.
// Each internal node reached gets its two other neighbours as children (child0, child1, indexed by node, in adj_u's
// slot order); tips end the walk.
// Returns the internal nodes in visiting order (parents before children).
static std::vector<int32_t> orient(const std::vector<int32_t> &adj_u, int32_t K,
                                   const std::vector<std::pair<int32_t, int32_t>> &starts, std::vector<int32_t> &child0,
                                   std::vector<int32_t> &child1) {
    std::vector<int32_t> to_visit;     // the internal nodes to visit, in order (head: the next one)
    std::vector<int32_t> reached_from; // the neighbour each was reached from
    to_visit.reserve(2 * K);
    reached_from.reserve(2 * K);
    for (const auto &[node, from] : starts) {
        to_visit.push_back(node);
        reached_from.push_back(from);
    }
    for (size_t head = 0; head < to_visit.size(); ++head) {
        const int32_t node = to_visit[head];
        const int32_t from = reached_from[head];
        // its two neighbours other than from
        int32_t b0 = -1, b1 = -1;
        for (int32_t k = 0; k < 3; ++k) {
            const int32_t nb = adj_u[3 * node + k];
            if (nb != -1 && nb != from) {
                if (b0 == -1)
                    b0 = nb;
                else
                    b1 = nb;
            }
        }
        child0[node] = b0;
        child1[node] = b1;
        if (b0 >= K) {
            to_visit.push_back(b0);
            reached_from.push_back(node);
        }
        if (b1 >= K) {
            to_visit.push_back(b1);
            reached_from.push_back(node);
        }
    }
    return to_visit;
}

// Re-encodes a skeleton after a batch of moves, or with the other labelling (the flip), for the next sweep.
void from_skeleton_tables(std::vector<int64_t> &v, const SkeletonTables &sk, int32_t root_c1, int32_t root_c2,
                          bool flip, std::vector<int32_t> &perm) {
    const int32_t K = (int32_t)sk.v_map.size(); // pieces = tips of the skeleton
    const int32_t R_sk = 2 * K - 1;             // nodes of the rooted skeleton
    const int32_t root_r = R_sk - 1;            // its root (2K - 2)
    // Step 1: root the unrooted skeleton on the edge between root_c1 and root_c2: a new root with those two as its
    // children.
    std::vector<int32_t> c1_r(R_sk, 0), c2_r(R_sk, 0);
    c1_r[root_r] = root_c1;
    c2_r[root_r] = root_c2;

    // Step 2: orient the unrooted skeleton away from that edge: each internal node's two rooted children.
    std::vector<std::pair<int32_t, int32_t>> starts; // the edge's internal ends, each reached from the other end
    if (root_c1 >= K)
        starts.push_back({root_c1, root_c2});
    if (root_c2 >= K)
        starts.push_back({root_c2, root_c1});
    orient(sk.adj_u, K, starts, c1_r, c2_r);

    // Step 3: encode the rooted tree as an ordered vector (queue shuffle, labelling flip), then compose perm with
    // the new tip labels (step_perm, the label mapping of this encoding) so each tip still points to its piece.
    std::vector<int32_t> step_perm(K);
    queue_shuffle_rooted(c1_r.data(), c2_r.data(), K, flip, v.data(), step_perm.data());
    std::vector<int32_t> new_perm(K);
    for (int32_t a = 0; a < K; ++a)
        new_perm[a] = perm[step_perm[a]];
    perm = new_perm;
}

namespace {
// A candidate move of the window solve: row i regrafted onto edge j, BME change dL.
struct CandidateMove {
    int32_t i;
    int32_t j;
    double dL;
};
} // namespace

// Optimize one K-leaf skeleton with path-disjoint SPR batching and a monotonic condensed-skeleton BME check.
// The device does the arithmetic.
double solve_window(int32_t K, int32_t max_iters, int32_t max_batch_cap, const int32_t *representative_offsets,
                    const int32_t *representative_leaves, const float *representative_weights, std::vector<int64_t> &v,
                    std::vector<int32_t> &perm, WindowKernels &kernels) {
    const int32_t M_sk = 2 * K - 2;              // nodes of the unrooted skeleton (the rooted one without its root)
    const int32_t root_u_sk = M_sk - 1;          // the node the unrooted skeleton hangs from (2K - 3)
    const std::vector<int64_t> init_v = v;       // the skeleton before the solve: kept if nothing improves
    const std::vector<int32_t> init_perm = perm; // its leaf permutation

    // The K x K distances between the pieces, from their representatives: computed once per window, kept for every
    // sweep.
    kernels.distances(K, representative_offsets, representative_leaves, representative_weights);

    SkeletonTables sk;
    std::vector<double> h_best_delta(K);
    std::vector<int32_t> h_best_j(K);

    double init_L = 0.0, curr_L = 0.0;
    double best_L = std::numeric_limits<double>::infinity(), prev_L = std::numeric_limits<double>::infinity();
    std::vector<int64_t> best_v = v, prev_v = v;
    std::vector<int32_t> best_perm = perm, prev_perm = perm;
    bool flipped_already = false;
    int32_t cur_batch_cap = max_batch_cap; // the batch cap B of Algorithm S1
    bool batch_pending = false;            // a batch was applied in the previous sweep and is checked now

    for (int32_t iter = 0; iter <= max_iters; ++iter) {
        // Step 1: decode the skeleton (v) into the arrays the kernels and the moves read.
        to_skeleton_tables(v, sk);
        std::vector<int32_t> &v_map = sk.v_map, &parent_u = sk.parent_u, &sub_size = sk.sub_size;
        std::vector<int32_t> &tin = sk.tin, &adj_u = sk.adj_u;

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
        } else if (batch_pending && curr_L >= prev_L - p2vbme::BATCH_MIN_GAIN) {
            // the last batch did not lower L (Algorithm S1, line 4): restore the previous tree and use smaller batches.
            // A single move that fails is below the resolution of the scores, so the window is converged.
            v = prev_v;
            perm = prev_perm;
            batch_pending = false;
            if (cur_batch_cap <= 1)
                break;
            cur_batch_cap = std::max(1, cur_batch_cap / 2);
            continue;
        } else if (batch_pending) {
            // the last batch lowered L: allow larger batches
            cur_batch_cap = std::min(max_batch_cap, cur_batch_cap + p2vbme::MOVE_BATCH_CAP_GROWTH);
        }
        // (no batch pending: the skeleton was only relabelled, after a flip or an undo, so there is nothing to judge)
        batch_pending = false;
        prev_L = curr_L;
        prev_v = v;
        prev_perm = perm;
        if (curr_L < best_L) {
            best_L = curr_L;
            best_v = v;
            best_perm = perm;
        }
        if (iter == max_iters)
            break;
        // Step 4: find the best regraft of every pruned subtree, with its BME change.
        kernels.best_moves(sk, h_best_delta, h_best_j);

        // Step 5: keep the improving moves, best first.
        std::vector<CandidateMove> cands;
        // A row's best move is a candidate only if its change is below this numerical zero (the scores may be single
        // precision).
        // A batch that does not lower L is undone in step 3, and max_iters bounds the sweeps.
        const float eps_move_thresh = -p2vbme::MOVE_MIN_GAIN;
        for (int32_t i = 1; i < K - 1; ++i) {
            // h_best_delta[i] < eps_move_thresh: the move improves the skeleton's BME score
            // h_best_j[i] >= 0: an edge was found to regraft onto (-1 otherwise)
            // h_best_j[i] < root_u_sk: the edge exists, i.e., isn't the root node.
            if (h_best_delta[i] < eps_move_thresh && h_best_j[i] >= 0 && h_best_j[i] < root_u_sk) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        // None: relabel the skeleton once with the other labelling of its cherries (its rows give other moves), then
        // stop.
        if (cands.empty()) {
            if (!flipped_already) {
                flipped_already = true;
                from_skeleton_tables(v, sk, root_u_sk, sk.rc2, true, perm);
                continue;
            }
            break;
        }
        flipped_already = false;
        std::sort(cands.begin(), cands.end(),
                  [](const CandidateMove &a, const CandidateMove &b) { return a.dL < b.dL; });

        // Step 6: apply a batch of at most cur_batch_cap moves to the unrooted skeleton (adj_u).
        // Going down the candidates, best first, a move joins the batch if its path (from the pruned subtree to the
        // regraft edge) shares no node with the moves already in it (Algorithm S1, line 10).
        // The paths are read on the skeleton before the batch (parent_u, tin, sub_size), while the moves rewire adj_u
        // one after another.
        // x is under y in the skeleton hung from root_u_sk (or is y)
        auto under = [&](int32_t x, int32_t y) { return tin[y] <= tin[x] && tin[x] < tin[y] + sub_size[y]; };
        std::vector<uint8_t> in_batch(M_sk, 0); // the nodes on the paths of the moves in the batch
        std::vector<int32_t> path;              // the nodes on the path of the move considered
        int32_t n_applied = 0;
        for (const CandidateMove &mc : cands) {
            // The move: prune the subtree under tr together with its attachment node pr, and regraft pr into the edge
            // between dst_child and its parent dst_parent (dst_child < root_u_sk from step 5, so dst_parent exists).
            const int32_t tr = v_map[mc.i + 1];
            const int32_t pr = (tr == root_u_sk) ? sk.rc2 : parent_u[tr];
            const int32_t dst_child = mc.j, dst_parent = parent_u[dst_child];
            // Left out: moves next to the hanging root, and regraft edges inside the pruned subtree.
            if (pr == root_u_sk || dst_parent == root_u_sk || under(dst_child, tr))
                continue;
            // Its path: tr and dst_parent, pr up to the lowest common ancestor with dst_child, and dst_child up to that
            // ancestor.
            path.assign({tr, dst_parent});
            int32_t lca = pr;
            for (; !under(dst_child, lca); lca = parent_u[lca])
                path.push_back(lca);
            for (int32_t x = dst_child; x != lca; x = parent_u[x])
                path.push_back(x);
            path.push_back(lca);
            // Skip it if its path meets a move already in the batch; otherwise mark the path and apply the move.
            if (std::any_of(path.begin(), path.end(), [&](int32_t x) { return in_batch[x] != 0; }))
                continue;
            for (int32_t x : path)
                in_batch[x] = 1;
            apply_unrooted_spr(adj_u.data(), tr, pr, dst_child, dst_parent);
            if (++n_applied >= cur_batch_cap)
                break;
        }
        // No candidate fit the checks: apply the single best move.
        if (n_applied == 0) {
            int32_t i = cands[0].i, dst_child = cands[0].j;
            int32_t tr = v_map[i + 1];
            int32_t pr = (tr == root_u_sk) ? sk.rc2 : parent_u[tr];
            int32_t dst_parent = parent_u[dst_child];
            apply_unrooted_spr(adj_u.data(), tr, pr, dst_child, dst_parent);
        }
        batch_pending = true;

        // Step 7: re-encode the moved skeleton as an ordered vector (the labelling alternates each sweep); the next
        // sweep scores it.
        int32_t r_nb = adj_u[3 * root_u_sk + 0];
        from_skeleton_tables(v, sk, root_u_sk, r_nb, (iter & 1) != 0, perm);
    }

    if (best_L >= init_L - p2vbme::WINDOW_MIN_GAIN) {
        v = init_v;
        perm = init_perm;
        return 0.0;
    }

    v = best_v;
    perm = best_perm;
    return best_L - init_L;
}

// A 64-bit hash of what a window solve reads: K, max_iters, the skeleton (v, perm) and the representatives (offsets,
// leaves and the bits of the weights). The rest is fixed within one search: the devices' alignment, distance model and
// precision, and the solve's constants. Two different inputs share a hash with probability about 2^-64; such a
// collision would skip a solve that might have improved the tree.
static uint64_t window_input_hash(int32_t K, int32_t max_iters, const std::vector<int64_t> &v,
                                  const std::vector<int32_t> &perm, const std::vector<int32_t> &offsets,
                                  const std::vector<int32_t> &leaves, const std::vector<float> &weights) {
    uint64_t h = 0;
    auto add = [&h](uint64_t x) { h = splitmix64_mix(h ^ (x + SPLITMIX64_GAMMA)); };
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

// Writes a window's solved skeleton (skel.v, skel.perm) back into the tree: decodes it, roots it (a subtree window
// where its outgroup attaches, the whole-tree window on its centroid edge), and gives the window's opened nodes their
// new children.
// It leaves the summary as it was: run_windows rebuilds it after every window that changed the tree.
static void write_back_skeleton(const Skeleton &skel, int32_t window_root, Tree &tree, const TreeSummary &summary) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t root_r = 2 * n_leaves - 2;
    const int32_t K = (int32_t)skel.v.size() + 1;
    const std::vector<int32_t> &piece_roots = skel.piece_roots;
    const int32_t K_in = (int32_t)piece_roots.size();
    const bool has_outgroup = window_root != root_r;
    std::vector<int32_t> &curr_c1 = tree.c1, &curr_c2 = tree.c2;

    // Decode the vector back to a skeleton + kernel extras
    SkeletonTables sk;
    to_skeleton_tables(skel.v, sk);

    // Step 1: choose the root.
    // A subtree window: the node its outgroup tip hangs from, so the subtree keeps its place.
    // The whole-tree window: its centroid edge, the split closest to N/2 leaves (pieces weighed by their leaf counts).
    int32_t a_out = -1, u_best = -1, p_best = -1;
    std::vector<std::pair<int32_t, int32_t>> starts; // where step 2 starts, each with the neighbour it is reached from
    if (has_outgroup) {
        for (int32_t a = 0; a < K; ++a) {
            if (skel.perm[a] == K_in) {
                a_out = a;
                break;
            }
        }
        starts.push_back({sk.adj_u[3 * a_out + 0], a_out});
    } else {
        const int32_t root_u_sk = 2 * K - 3;
        std::vector<int32_t> w_sub(2 * K - 2, 0);
        for (int32_t a = 0; a < K; ++a)
            w_sub[a] = summary.leaf_count[piece_roots[skel.perm[a]]];
        for (int32_t u = K; u < root_u_sk; ++u)
            w_sub[u] = w_sub[sk.c1[u]] + w_sub[sk.c2[u]];
        u_best = 0;
        int32_t best_diff = std::abs(2 * w_sub[0] - n_leaves);
        for (int32_t u = 1; u < root_u_sk; ++u) {
            int32_t diff = std::abs(2 * w_sub[u] - n_leaves);
            if (diff < best_diff) {
                best_diff = diff;
                u_best = u;
            }
        }
        p_best = sk.parent_u[u_best];
        if (u_best >= K)
            starts.push_back({u_best, p_best});
        if (p_best >= K)
            starts.push_back({p_best, u_best});
    }
    // Step 2: orient the skeleton away from that root, breadth first: each internal node's two children.
    // order: the skeleton's internal nodes, parents before children.
    std::vector<int32_t> child0(2 * K - 2, -1), child1(2 * K - 2, -1);
    const std::vector<int32_t> order = orient(sk.adj_u, K, starts, child0, child1);

    // Step 3: map the skeleton's nodes to tree nodes: each tip to its piece's root, and the internal nodes to the
    // window's opened nodes, children before parents (so window_root stays at the top).
    std::vector<int32_t> sk_to_full(2 * K - 2, -1);
    for (int32_t a = 0; a < K; ++a) {
        if (a != a_out)
            sk_to_full[a] = piece_roots[skel.perm[a]];
    }
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = order[K - 3 - s];
        sk_to_full[u_sk] = skel.opened[s];
    }
    // Step 4: give the opened nodes their new children.
    for (int32_t s = 0; s < K - 2; ++s) {
        int32_t u_sk = order[K - 3 - s];
        int32_t full_u = sk_to_full[u_sk];
        int32_t b0 = sk_to_full[child0[u_sk]];
        int32_t b1 = sk_to_full[child1[u_sk]];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
    }
    // Step 5: the top.
    // A subtree window: nothing more, as window_root is still its old parent's child (the outgroup is dropped).
    // The whole-tree window: the spare opened node (the tree root) joins the centroid edge's two ends.
    if (!has_outgroup) {
        int32_t full_u = skel.opened[K - 2]; // == root_r
        int32_t b0 = sk_to_full[u_best];
        int32_t b1 = sk_to_full[p_best];
        curr_c1[full_u] = b0;
        curr_c2[full_u] = b1;
    }
}

// Optimize the window below window_root and write the result into the tree.
// The window collapses clades into K pieces, each with dyadic-weighted representative leaves, and optimizes them as an
// unrooted BME tree.
// A subtree window (window_root != root_r) also sees the rest of the tree, as one extra piece, the outgroup,
// because BME sums over all leaf pairs: moves inside the subtree change its distances to the leaves outside.
// Its result is rooted where the outgroup attaches, so the subtree keeps its place, and the outgroup is dropped.
// The whole-tree window (window_root == root_r) has no outgroup and BME does not fix a root, so it re-roots on the
// centroid edge.
// Windows of one wave run in parallel: they pass a snapshot of the tree and its summary (snapshot, snapshot_summary),
// read their representatives from it; run_windows rebuilds the summary after the wave.
// fruitless: the input hashes (window_input_hash) of this search's solves that found no improvement.
// The solve is deterministic, so a window with one of these inputs would again find none: it is skipped.
// fruitless_hash: set to the input hash when the solve runs and finds no improvement, for the caller to add to
// fruitless.

static double run_subtree_minibatch_window(int32_t window_root, int32_t max_iters, const SearchSettings &settings,
                                           const std::vector<int32_t> &outgroup_leaves,
                                           const std::vector<float> &outgroup_weights, Device &dev, Tree &tree,
                                           const TreeSummary &summary, const std::unordered_set<uint64_t> &fruitless,
                                           std::optional<uint64_t> &fruitless_hash,
                                           const uint8_t *stop_at_comp_root = NULL, const Tree *snapshot = NULL,
                                           const TreeSummary *snapshot_summary = NULL) {
    const int32_t n_leaves = tree.n_leaves();
    const int32_t root_r = 2 * n_leaves - 2;
    const int32_t representative_cap = settings.representative_cap;
    // A subtree window also sees the rest of the tree, as one extra piece (the outgroup).
    const bool has_outgroup = window_root != root_r;
    // Decide what a window contains and return its skeleton
    Skeleton skel = extract_skeleton(window_root, tree, summary, stop_at_comp_root, has_outgroup);
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

    const Tree &otree = (snapshot != NULL) ? *snapshot : tree;
    const TreeSummary &osum = (snapshot_summary != NULL) ? *snapshot_summary : summary;

    // Build the representatives of the window's K pieces in one CSR layout
    std::vector<int32_t> representative_offsets(K + 1, 0), representative_leaves;
    std::vector<float> representative_weights;
    representative_leaves.reserve(4 * K);
    representative_weights.reserve(4 * K);
    for (int32_t a = 0; a < K_in; ++a) {
        representative_offsets[a] = (int32_t)representative_leaves.size();
        compute_clade_dyadic_representatives(piece_roots[a], otree, osum, representative_cap, representative_leaves,
                                             representative_weights);
    }
    representative_offsets[K_in] = (int32_t)representative_leaves.size();
    if (has_outgroup) {
        representative_leaves.insert(representative_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
        representative_weights.insert(representative_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
        representative_offsets[K] = (int32_t)representative_leaves.size();
    }

    // Skip a solve whose input already found no improvement.
    const uint64_t input_hash = window_input_hash(K, max_iters, skel.v, skel.perm, representative_offsets,
                                                  representative_leaves, representative_weights);
    if (fruitless.count(input_hash) != 0)
        return 0.0;
    // Core device-powered search through the skeleton with SPR sweeps
    // Returns the gain, and the new skeleton in skel.v, skel.perm
    double dL =
        solve_window(K, max_iters, p2vbme::MOVE_BATCH_CAP, representative_offsets.data(), representative_leaves.data(),
                     representative_weights.data(), skel.v, skel.perm, *dev.window_kernels());
    if (dL == 0.0) {
        fruitless_hash = input_hash;
        return 0.0;
    }

    // Write the solved skeleton back into the tree.
    write_back_skeleton(skel, window_root, tree, summary);
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
    const int32_t representative_cap = s.settings.representative_cap;
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
                    // Build the representatives of everything outside the window's clade
                    // so the window can include the rest of the tree as one extra piece (the outgroup).
                    compute_outgroup_dyadic_representatives(window_root, snapshot, snapshot_summary, representative_cap,
                                                            out_leaves, out_weights);
                }
                // Turn the window into a skeleton of pieces with representatives
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
                // Build the representatives of everything outside the window's clade
                // so the window can include the rest of the tree as one extra piece (the outgroup).
                compute_outgroup_dyadic_representatives(window_root, s.tree, s.summary, representative_cap, out_leaves,
                                                        out_weights);
            }
            std::optional<uint64_t> fruitless_hash;
            // Turn the window into a skeleton of pieces with representatives
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

// The schedule of a search: rounds of window passes (backbone, leaf-covering waves and, above WAVE_COMPONENT_CAP
// leaves, the V-cycle) until the rounds stop, then, above WAVE_COMPONENT_CAP leaves, a final cleanup wave pass.
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
