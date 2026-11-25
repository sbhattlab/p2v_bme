/*
 * support.cpp
 *
 * The supports of a reference tree's branches (support.hpp): a replicate tree's heavy-path segment tree, the reference
 * tree's schedule of leaf moves, and the counter that runs one against the other.
 * Trees are rooted and binary with leaves 0..K-1 and root 2K-2 (core.hpp); internal ids are in no particular order, so
 * every traversal starts at the root.
 */
#include "support.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vecme {

namespace {

// Moving a leaf into S changes W = |C| - 2 |C n S| of every subtree C that holds it by this.
constexpr int32_t INTO_S = -2;
constexpr int32_t OUT_OF_S = +2;

// A replicate tree prepared for transfer distances: W of each non-root node, in heavy-path order in a segment tree
// with range additions, and each leaf's root path as O(log K) ranges of that order.
struct ReplicateTree {
    int32_t K = 0, n_nodes = 0, root = 0, M = 0, seg_n = 1;
    std::vector<int32_t> parent, n_leaves_below, heavy, head, pos, inv_pos, path_offset;
    struct Range {
        int32_t l, r;
    };
    std::vector<Range> paths; // leaf l's root path: paths[path_offset[l] .. path_offset[l + 1])
    std::vector<int32_t> seg_min, seg_max, seg_lazy;

    // Builds the decomposition of a tree on K leaves: (c1, c2) -> W = |C| for every subtree (S empty).
    void init(int32_t n_leaves, const int32_t *c1, const int32_t *c2) {
        K = n_leaves;
        n_nodes = 2 * K - 1;
        root = n_nodes - 1;
        M = n_nodes - 1; // the non-root nodes
        parent.assign(n_nodes, -1);
        n_leaves_below.assign(n_nodes, 0);
        heavy.assign(n_nodes, -1);
        head.assign(n_nodes, -1);
        pos.assign(n_nodes, -1);
        inv_pos.assign(M, -1);

        std::vector<int32_t> order; // parents before children
        order.reserve(n_nodes);
        order.push_back(root);
        for (size_t i = 0; i < order.size(); ++i) {
            const int32_t u = order[i];
            if (u >= K) {
                parent[c1[u]] = u;
                parent[c2[u]] = u;
                order.push_back(c1[u]);
                order.push_back(c2[u]);
            }
        }
        if ((int32_t)order.size() != n_nodes)
            throw std::invalid_argument("support: a replicate tree does not reach all its nodes from the root");
        std::vector<int32_t> n_below(n_nodes, 1); // nodes in the subtree, for the heavy child
        for (int32_t i = 0; i < K; ++i)
            n_leaves_below[i] = 1;
        for (int32_t idx = (int32_t)order.size() - 1; idx >= 0; --idx) {
            const int32_t u = order[idx];
            if (u >= K) {
                const int32_t l = c1[u], r = c2[u];
                n_leaves_below[u] = n_leaves_below[l] + n_leaves_below[r];
                n_below[u] = 1 + n_below[l] + n_below[r];
                heavy[u] = (n_below[l] >= n_below[r]) ? l : r;
            }
        }
        // heavy paths, each contiguous in pos; the root is left out (it has no split)
        int32_t next_pos = 0;
        struct Frame {
            int32_t u, h;
        };
        std::vector<Frame> stack = {{c2[root], c2[root]}, {c1[root], c1[root]}};
        while (!stack.empty()) {
            const Frame f = stack.back();
            stack.pop_back();
            int32_t u = f.u;
            const int32_t h = f.h;
            while (u != -1) {
                head[u] = h;
                pos[u] = next_pos;
                inv_pos[next_pos++] = u;
                if (u >= K) {
                    const int32_t hv = heavy[u], light = (c1[u] == hv) ? c2[u] : c1[u];
                    if (light != -1)
                        stack.push_back({light, light});
                    u = hv;
                } else {
                    break;
                }
            }
        }
        path_offset.assign(K + 1, 0);
        paths.clear();
        paths.reserve((size_t)K * 8);
        for (int32_t l = 0; l < K; ++l) {
            path_offset[l] = (int32_t)paths.size();
            for (int32_t u = l; u != root && u != -1; u = parent[head[u]])
                paths.push_back({pos[head[u]], pos[u]});
        }
        path_offset[K] = (int32_t)paths.size();

        seg_n = 1;
        while (seg_n < M)
            seg_n <<= 1;
        seg_min.assign(2 * seg_n, std::numeric_limits<int32_t>::max() / 2);
        seg_max.assign(2 * seg_n, std::numeric_limits<int32_t>::min() / 2);
        seg_lazy.assign(2 * seg_n, 0);
        for (int32_t i = 0; i < M; ++i) {
            const int32_t w = n_leaves_below[inv_pos[i]];
            seg_min[seg_n + i] = w;
            seg_max[seg_n + i] = w;
        }
        for (int32_t i = seg_n - 1; i >= 1; --i) {
            seg_min[i] = std::min(seg_min[2 * i], seg_min[2 * i + 1]);
            seg_max[i] = std::max(seg_max[2 * i], seg_max[2 * i + 1]);
        }
    }

    // Adds delta to W at positions l..r: lazy tags on the covering nodes, then their ancestors refreshed.
    void range_add(int32_t l, int32_t r, int32_t delta) {
        const int32_t l0 = l + seg_n, r0 = r + seg_n;
        int32_t L = l0, R = r0;
        while (L <= R) {
            if (L & 1) {
                seg_lazy[L] += delta;
                seg_min[L] += delta;
                seg_max[L] += delta;
                L++;
            }
            if (!(R & 1)) {
                seg_lazy[R] += delta;
                seg_min[R] += delta;
                seg_max[R] += delta;
                R--;
            }
            L >>= 1;
            R >>= 1;
        }
        for (int32_t i = l0 >> 1; i >= 1; i >>= 1) {
            seg_min[i] = std::min(seg_min[2 * i], seg_min[2 * i + 1]) + seg_lazy[i];
            seg_max[i] = std::max(seg_max[2 * i], seg_max[2 * i + 1]) + seg_lazy[i];
        }
        for (int32_t i = r0 >> 1; i >= 1; i >>= 1) {
            seg_min[i] = std::min(seg_min[2 * i], seg_min[2 * i + 1]) + seg_lazy[i];
            seg_max[i] = std::max(seg_max[2 * i], seg_max[2 * i + 1]) + seg_lazy[i];
        }
    }

    // Moves leaf l into S (INTO_S) or out of it (OUT_OF_S): every subtree on its root path changes W by delta.
    void move_leaf(int32_t l, int32_t delta) {
        for (int32_t idx = path_offset[l]; idx < path_offset[l + 1]; ++idx)
            range_add(paths[idx].l, paths[idx].r, delta);
    }

    // Computes the transfer distance of the current S (s leaves) to this tree: s -> min over its splits.
    int32_t transfer_distance(int32_t s) const { return std::min(s + seg_min[1], (K - s) - seg_max[1]); }
};

// The reference tree as a schedule ("DSU on tree", heavy child kept): the leaf moves and, at each non-root node, the
// query of its split, so that S holds exactly that node's leaves when it is queried.
struct ReferenceSchedule {
    int32_t K = 0, n_nodes = 0, root = 0, second_root_child = -1;
    std::vector<int32_t> c1, c2, n_leaves_below, heavy_child, light_child, leaf_order, tin, tout;
    struct Step {
        bool query; // false: move leaf_order[a .. b) by delta; true: query node a with b leaves
        int32_t a, b, delta;
    };
    std::vector<Step> steps;

    // Builds the schedule of a tree on K leaves: (c1, c2) -> steps.
    void init(int32_t n_leaves, const int32_t *in_c1, const int32_t *in_c2) {
        K = n_leaves;
        n_nodes = 2 * K - 1;
        root = n_nodes - 1;
        c1.assign(in_c1, in_c1 + n_nodes);
        c2.assign(in_c2, in_c2 + n_nodes);
        second_root_child = std::max(c1[root], c2[root]);
        n_leaves_below.assign(n_nodes, 0);
        heavy_child.assign(n_nodes, -1);
        light_child.assign(n_nodes, -1);
        tin.assign(n_nodes, 0);
        tout.assign(n_nodes, 0);
        leaf_order.clear();
        leaf_order.reserve(K);

        std::vector<int32_t> order;
        order.reserve(n_nodes);
        order.push_back(root);
        for (size_t i = 0; i < order.size(); ++i)
            if (order[i] >= K) {
                order.push_back(c1[order[i]]);
                order.push_back(c2[order[i]]);
            }
        if ((int32_t)order.size() != n_nodes)
            throw std::invalid_argument("support: the reference tree does not reach all its nodes from the root");
        for (int32_t i = 0; i < K; ++i)
            n_leaves_below[i] = 1;
        for (int32_t idx = (int32_t)order.size() - 1; idx >= 0; --idx) {
            const int32_t u = order[idx];
            if (u >= K) {
                const int32_t l = c1[u], r = c2[u];
                n_leaves_below[u] = n_leaves_below[l] + n_leaves_below[r];
                heavy_child[u] = (n_leaves_below[l] >= n_leaves_below[r]) ? l : r;
                light_child[u] = (n_leaves_below[l] >= n_leaves_below[r]) ? r : l;
            }
        }
        // the leaves in depth-first order, light child first: a subtree's leaves are leaf_order[tin .. tout)
        struct Frame {
            int32_t u, stage;
        };
        std::vector<Frame> stack = {{root, 0}};
        while (!stack.empty()) {
            Frame &f = stack.back();
            const int32_t u = f.u;
            if (u < K) {
                tin[u] = (int32_t)leaf_order.size();
                leaf_order.push_back(u);
                tout[u] = (int32_t)leaf_order.size();
                stack.pop_back();
                continue;
            }
            if (f.stage == 0) {
                tin[u] = (int32_t)leaf_order.size();
                f.stage = 1;
                stack.push_back({light_child[u], 0});
            } else if (f.stage == 1) {
                f.stage = 2;
                stack.push_back({heavy_child[u], 0});
            } else {
                tout[u] = (int32_t)leaf_order.size();
                stack.pop_back();
            }
        }
        // DSU on tree: a node's light subtrees are done and emptied first; its heavy subtree's leaves stay in S, its
        // light ones are added back, the node is queried, and unless its parent keeps them all its leaves leave S
        steps.clear();
        steps.reserve(4 * (size_t)K);
        struct DsuFrame {
            int32_t u;
            bool keep;
            int32_t stage;
        };
        std::vector<DsuFrame> dstack = {{root, false, 0}};
        while (!dstack.empty()) {
            DsuFrame &f = dstack.back();
            const int32_t u = f.u;
            if (u < K) {
                if (f.keep)
                    steps.push_back({false, tin[u], tout[u], INTO_S});
                dstack.pop_back();
                continue;
            }
            if (f.stage == 0) {
                f.stage = 1;
                dstack.push_back({light_child[u], false, 0});
            } else if (f.stage == 1) {
                f.stage = 2;
                dstack.push_back({heavy_child[u], u != root, 0});
            } else {
                if (u != root) {
                    const int32_t lc = light_child[u];
                    steps.push_back({false, tin[lc], tout[lc], INTO_S});
                    steps.push_back({true, u, n_leaves_below[u], 0});
                    if (!f.keep)
                        steps.push_back({false, tin[u], tout[u], OUT_OF_S});
                }
                dstack.pop_back();
            }
        }
    }

    // Runs the schedule against a replicate: replicate -> the transfer distance of every non-root node's split.
    void run(ReplicateTree &replicate, std::vector<int32_t> &delta) const {
        delta.assign(n_nodes, 0);
        for (const Step &st : steps) {
            if (st.query)
                delta[st.a] = replicate.transfer_distance(st.b);
            else
                for (int32_t idx = st.a; idx < st.b; ++idx)
                    replicate.move_leaf(leaf_order[idx], st.delta);
        }
        // The root's two children have complementary splits, so the same distance; one value for both.
        const int32_t rc1 = c1[root], rc2 = c2[root];
        if (rc1 >= K && rc2 >= K)
            delta[second_root_child] = delta[(rc1 == second_root_child) ? rc2 : rc1];
    }

    // Leaves on the smaller side of node u's split (p).
    int32_t smaller_side(int32_t u) const { return std::min(n_leaves_below[u], K - n_leaves_below[u]); }
};

} // namespace

struct SupportCounter::Impl {
    ReferenceSchedule reference;
    ReplicateTree replicate; // reused across replicates
    std::vector<int32_t> delta;
    std::vector<int64_t> sum_delta;   // per node, over the replicates
    std::vector<int32_t> n_identical; // per node: replicates with delta = 0
    int32_t n_replicates = 0;
};

SupportCounter::SupportCounter(const Tree &reference) : impl_(new Impl) {
    const int32_t K = reference.n_leaves();
    if (K < 4)
        throw std::invalid_argument("support: the reference tree needs at least 4 leaves");
    impl_->reference.init(K, reference.c1.data(), reference.c2.data());
    impl_->sum_delta.assign(2 * K - 1, 0);
    impl_->n_identical.assign(2 * K - 1, 0);
}

SupportCounter::~SupportCounter() = default;

void SupportCounter::add(const Tree &replicate) {
    Impl &m = *impl_;
    const int32_t K = m.reference.K;
    if (replicate.n_leaves() != K || (int32_t)replicate.c2.size() != 2 * K - 1)
        throw std::invalid_argument("support: the replicate tree is not on the reference tree's leaves");
    m.replicate.init(K, replicate.c1.data(), replicate.c2.data());
    m.reference.run(m.replicate, m.delta);
    for (int32_t u = K; u < 2 * K - 2; ++u) {
        m.sum_delta[u] += m.delta[u];
        m.n_identical[u] += m.delta[u] == 0;
    }
    ++m.n_replicates;
}

int32_t SupportCounter::n_replicates() const { return impl_->n_replicates; }

std::vector<double> SupportCounter::exact() const {
    const Impl &m = *impl_;
    const int32_t K = m.reference.K;
    std::vector<double> out(2 * K - 1, NO_SUPPORT);
    if (m.n_replicates == 0)
        return out;
    for (int32_t u = K; u < 2 * K - 2; ++u)
        if (m.reference.smaller_side(u) >= 2)
            out[u] = 100.0 * (double)m.n_identical[u] / (double)m.n_replicates;
    return out;
}

std::vector<double> SupportCounter::tbe() const {
    const Impl &m = *impl_;
    const int32_t K = m.reference.K;
    std::vector<double> out(2 * K - 1, NO_SUPPORT);
    if (m.n_replicates == 0)
        return out;
    for (int32_t u = K; u < 2 * K - 2; ++u) {
        const int32_t p = m.reference.smaller_side(u);
        if (p < 2)
            continue;
        // p = 2: delta is 0 or 1, so the TBE is the exact support (as the paper's code computes it)
        const double exact = 100.0 * (double)m.n_identical[u] / (double)m.n_replicates;
        const double mean_delta = (double)m.sum_delta[u] / (double)m.n_replicates;
        out[u] = std::clamp((p == 2) ? exact : 100.0 * (1.0 - mean_delta / (double)(p - 1)), 0.0, 100.0);
    }
    return out;
}

namespace {

// Averages per-node supports over the internal branches of the unrooted tree: the root's second child repeats its
// sibling's branch and is left out.
double mean_over_branches(const std::vector<double> &support, int32_t second_root_child) {
    double sum = 0.0;
    int64_t n = 0;
    for (size_t u = 0; u < support.size(); ++u)
        if (support[u] != NO_SUPPORT && (int32_t)u != second_root_child) {
            sum += support[u];
            ++n;
        }
    return n > 0 ? sum / (double)n : NO_SUPPORT;
}

} // namespace

double SupportCounter::mean_tbe() const { return mean_over_branches(tbe(), impl_->reference.second_root_child); }

double SupportCounter::mean_exact() const { return mean_over_branches(exact(), impl_->reference.second_root_child); }

std::string format_support(double percent) {
    percent = std::clamp(percent, 0.0, 100.0);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", percent);
    size_t n = std::strlen(buf);
    while (n > 1 && buf[n - 1] == '0')
        buf[--n] = '\0';
    if (n > 1 && buf[n - 1] == '.')
        buf[--n] = '\0';
    return std::string(buf, n);
}

} // namespace vecme
