// Tests of the search's building blocks (src/search.hpp), on random trees:
//   - compute_tree_summary: parents, BFS order, leaf counts, stand-ins at the shallowest leaf;
//   - build_component_waves: components partition the internal nodes, within their size bounds, with their waves at
//     their levels;
//   - extract_skeleton: pieces and opened nodes describe the window, and its vector decodes into the window's clades;
//   - to_skeleton_tables: the tables agree with each other;
//   - from_skeleton_tables: re-encoding gives the same unrooted tree, from any root edge and labelling, also after a
//     move.
#include <gtest/gtest.h>

#include <stdint.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "config.hpp"
#include "phylo2vec.hpp"
#include "search.hpp"
#include "start_tree.hpp"

using namespace p2vbme;

namespace {

// Builds a random tree on n leaves with its internal ids shuffled (the root keeps 2n - 2), as the search leaves them.
Tree random_search_tree(int32_t n, uint64_t seed) {
    const Tree t = build_random_tree(n, seed);
    std::mt19937_64 rng(seed);
    std::vector<int32_t> id(2 * n - 1);
    std::iota(id.begin(), id.end(), 0);
    if (n > 2)
        std::shuffle(id.begin() + n, id.end() - 1, rng);
    Tree out;
    out.c1.assign(2 * n - 1, 0);
    out.c2.assign(2 * n - 1, 0);
    for (int32_t u = n; u < 2 * n - 1; ++u) {
        out.c1[id[u]] = id[t.c1[u]];
        out.c2[id[u]] = id[t.c2[u]];
    }
    return out;
}

// Draws a random ordered vector for k tips (0 <= v[i] <= i).
std::vector<int64_t> random_ordered_vector(int32_t k, std::mt19937_64 &rng) {
    std::vector<int64_t> v(k - 1);
    for (int32_t i = 0; i < k - 1; ++i)
        v[i] = (int64_t)(rng() % (uint64_t)(i + 1));
    return v;
}

// Lists the tips (label[tip]) under node u of a rooted tree (c1, c2; tips 0..n-1).
std::vector<int32_t> tips_under(const std::vector<int32_t> &c1, const std::vector<int32_t> &c2, int32_t n, int32_t u,
                                const std::vector<int32_t> &label) {
    std::vector<int32_t> tips, st = {u};
    while (!st.empty()) {
        const int32_t x = st.back();
        st.pop_back();
        if (x < n) {
            tips.push_back(label[x]);
        } else {
            st.push_back(c1[x]);
            st.push_back(c2[x]);
        }
    }
    std::sort(tips.begin(), tips.end());
    return tips;
}

// Lists the non-trivial splits of an unrooted tree, each as its side without label 0: rooted tree (c1, c2) -> splits.
std::set<std::vector<int32_t>> splits_of_rooted(const std::vector<int32_t> &c1, const std::vector<int32_t> &c2,
                                                int32_t n, const std::vector<int32_t> &label) {
    std::set<std::vector<int32_t>> out;
    for (int32_t u = n; u < 2 * n - 2; ++u) {
        std::vector<int32_t> side = tips_under(c1, c2, n, u, label);
        if (std::binary_search(side.begin(), side.end(), 0)) {
            std::vector<int32_t> other;
            for (int32_t a = 0; a < n; ++a)
                if (!std::binary_search(side.begin(), side.end(), a))
                    other.push_back(a);
            side = other;
        }
        if (side.size() >= 2 && (int32_t)side.size() <= n - 2)
            out.insert(side);
    }
    return out;
}

// Lists the same splits from an unrooted adjacency (3 slots per node, tips 0..k-1): adj_u -> splits.
std::set<std::vector<int32_t>> splits_of_adjacency(const std::vector<int32_t> &adj_u, int32_t k,
                                                   const std::vector<int32_t> &label) {
    // root at tip 0's neighbour, then read the rooted tree's splits
    const int32_t n_nodes = 2 * k - 2;
    std::vector<int32_t> c1(2 * k - 1, 0), c2(2 * k - 1, 0), parent(n_nodes, -1), order = {adj_u[0]};
    parent[adj_u[0]] = 0;
    for (size_t h = 0; h < order.size(); ++h) {
        const int32_t u = order[h];
        int32_t b[2] = {-1, -1}, nb_found = 0;
        for (int32_t s = 0; s < 3; ++s) {
            const int32_t nb = adj_u[3 * u + s];
            if (nb != -1 && nb != parent[u])
                b[nb_found++] = nb;
        }
        c1[u] = b[0];
        c2[u] = b[1];
        for (int32_t x : b)
            if (x >= k) {
                parent[x] = u;
                order.push_back(x);
            }
    }
    // a rooted tree with root 2k - 2 joining tip 0 and its neighbour, internal ids k .. 2k - 3 as in adj_u
    c1[2 * k - 2] = 0;
    c2[2 * k - 2] = adj_u[0];
    return splits_of_rooted(c1, c2, k, label);
}

std::vector<int32_t> identity(int32_t n) {
    std::vector<int32_t> id(n);
    std::iota(id.begin(), id.end(), 0);
    return id;
}

} // namespace

// compute_tree_summary: parents, the BFS order (parents first), leaf counts and stand-ins at minimal depth
TEST(Search, TreeSummary) {
    for (int32_t n : {4, 5, 50, 2000}) {
        for (uint64_t seed = 1; seed <= 5; ++seed) {
            const Tree tree = random_search_tree(n, seed);
            TreeSummary s;
            compute_tree_summary(tree, s);
            const int32_t root = 2 * n - 2;
            ASSERT_EQ((int32_t)s.order.size(), n - 1);
            EXPECT_EQ(s.order[0], root);
            EXPECT_EQ(s.parent[root], -1);
            std::vector<int32_t> pos(2 * n - 1, -1);
            for (int32_t i = 0; i < n - 1; ++i) {
                ASSERT_GE(s.order[i], n);
                ASSERT_EQ(pos[s.order[i]], -1) << "node listed twice";
                pos[s.order[i]] = i;
            }
            for (int32_t u = n; u < 2 * n - 1; ++u) {
                for (int32_t c : {tree.c1[u], tree.c2[u]}) {
                    EXPECT_EQ(s.parent[c], u);
                    if (c >= n) {
                        EXPECT_LT(pos[u], pos[c]) << "a child before its parent in order";
                    }
                }
            }
            for (int32_t u = 0; u < 2 * n - 1; ++u) {
                // brute force: the leaves under u and their depths below u
                std::vector<std::pair<int32_t, int32_t>> st = {{u, 0}};
                int32_t leaves = 0, min_depth = 1 << 30;
                while (!st.empty()) {
                    const auto [x, d] = st.back();
                    st.pop_back();
                    if (x < n) {
                        ++leaves;
                        min_depth = std::min(min_depth, d);
                    } else {
                        st.push_back({tree.c1[x], d + 1});
                        st.push_back({tree.c2[x], d + 1});
                    }
                }
                EXPECT_EQ(s.leaf_count[u], leaves);
                // the stand-in is a leaf under u at the minimal depth
                const int32_t l = s.stand_in[u];
                ASSERT_GE(l, 0);
                ASSERT_LT(l, n);
                int32_t depth = 0;
                for (int32_t x = l; x != u; x = s.parent[x]) {
                    ASSERT_NE(x, -1) << "stand-in not under its node";
                    ++depth;
                }
                EXPECT_EQ(depth, min_depth) << "node " << u;
            }
        }
    }
}

// build_component_waves: components partition the internal nodes within their size bounds; the waves list exactly the
// components with a leaf of their own, at their levels
TEST(Search, ComponentWaves) {
    const int32_t MIN_NODES = 15, ROOT_EXTRA = 4; // build_component_waves' bounds (see its comment)
    for (int32_t n : {10, 300, 3000, 20000}) {
        for (int32_t cap : {30, 100, WAVE_COMPONENT_CAP}) {
            const Tree tree = random_search_tree(n, 11 + (uint64_t)n + (uint64_t)cap);
            TreeSummary s;
            compute_tree_summary(tree, s);
            std::vector<uint8_t> is_root;
            std::vector<std::vector<int32_t>> waves;
            build_component_waves(tree, s, cap, is_root, waves);
            const int32_t root = 2 * n - 2;
            ASSERT_TRUE(is_root[root]);
            // each internal node's component: its nearest marked ancestor (or itself)
            std::vector<int32_t> comp(2 * n - 1, -1), size(2 * n - 1, 0);
            for (int32_t u = n; u < 2 * n - 1; ++u) {
                int32_t x = u;
                while (!is_root[x])
                    x = s.parent[x];
                comp[u] = x;
                ++size[x];
            }
            for (int32_t u = 0; u < n; ++u)
                EXPECT_FALSE(is_root[u]) << "a leaf marked as a component root";
            // sizes; levels (0 = no component below); a leaf of its own
            std::vector<int32_t> level(2 * n - 1, 0);
            std::vector<uint8_t> own_leaf(2 * n - 1, 0);
            for (int32_t i = n - 2; i >= 0; --i) { // children before parents
                const int32_t u = s.order[i];
                for (int32_t c : {tree.c1[u], tree.c2[u]}) {
                    if (c < n)
                        own_leaf[comp[u]] = 1;
                    else if (comp[c] != comp[u])
                        level[comp[u]] = std::max(level[comp[u]], level[c] + 1);
                }
            }
            std::set<int32_t> listed;
            for (size_t k = 0; k < waves.size(); ++k)
                for (int32_t r : waves[k]) {
                    EXPECT_TRUE(is_root[r]);
                    EXPECT_EQ(level[r], (int32_t)k) << "component " << r << " in the wrong wave";
                    EXPECT_TRUE(listed.insert(r).second) << "component listed twice";
                }
            for (int32_t u = n; u < 2 * n - 1; ++u) {
                if (!is_root[u])
                    continue;
                if (u == root)
                    EXPECT_LE(size[u], cap + ROOT_EXTRA);
                else {
                    EXPECT_GE(size[u], MIN_NODES) << "n " << n << " cap " << cap;
                    EXPECT_LE(size[u], cap);
                }
                EXPECT_EQ(listed.count(u) == 1, own_leaf[u] == 1) << "component " << u;
            }
        }
    }
    const Tree tree = random_search_tree(100, 1);
    TreeSummary s;
    compute_tree_summary(tree, s);
    std::vector<uint8_t> is_root;
    std::vector<std::vector<int32_t>> waves;
    EXPECT_THROW(build_component_waves(tree, s, 2 * MIN_NODES - 1, is_root, waves), std::invalid_argument);
}

// extract_skeleton: the opened nodes are a connected top part of the window, the pieces hang below them and share out
// its leaves, and the vector (with its label mapping) decodes into the window's clades
TEST(Search, ExtractSkeleton) {
    const int32_t n = 3000;
    const Tree tree = random_search_tree(n, 21);
    TreeSummary s;
    compute_tree_summary(tree, s);
    std::vector<uint8_t> stops;
    std::vector<std::vector<int32_t>> waves;
    build_component_waves(tree, s, WAVE_COMPONENT_CAP, stops, waves);
    const int32_t root = 2 * n - 2;
    struct Window {
        int32_t root;
        const uint8_t *stops;
    };
    std::vector<Window> windows = {{root, NULL}, {root, stops.data()}};
    for (const auto &wave : waves)
        for (int32_t r : wave)
            windows.push_back({r, stops.data()});
    std::mt19937_64 rng(5);
    for (int i = 0; i < 30; ++i)
        windows.push_back({n + (int32_t)(rng() % (uint64_t)(n - 1)), NULL}); // any internal node
    windows.push_back({(int32_t)(rng() % (uint64_t)n), NULL});               // a leaf
    for (int32_t u = n; u < root; ++u)                                       // a cherry: K = 3 with the outgroup
        if (tree.c1[u] < n && tree.c2[u] < n) {
            windows.push_back({u, NULL});
            break;
        }
    for (const Window &w : windows) {
        const bool with_outgroup = w.root != root;
        const Skeleton skel = extract_skeleton(w.root, tree, s, w.stops, with_outgroup);
        const int32_t k_in = (int32_t)skel.piece_roots.size();
        const int32_t k = with_outgroup ? k_in + 1 : k_in;
        if (w.root < n) { // a leaf: one piece
            EXPECT_EQ(k_in, 1);
            EXPECT_TRUE(skel.opened.empty());
            EXPECT_TRUE(skel.v.empty());
            continue;
        }
        ASSERT_EQ((int32_t)skel.opened.size(), k_in - 1);
        EXPECT_LE((int32_t)skel.opened.size(), WINDOW_MAX_INTERNALS);
        EXPECT_EQ(skel.opened.back(), w.root);
        const std::set<int32_t> opened(skel.opened.begin(), skel.opened.end());
        const std::set<int32_t> pieces(skel.piece_roots.begin(), skel.piece_roots.end());
        EXPECT_EQ((int32_t)opened.size(), k_in - 1) << "an opened node listed twice";
        EXPECT_EQ((int32_t)pieces.size(), k_in) << "a piece listed twice";
        int64_t leaves = 0;
        for (int32_t p : skel.piece_roots) {
            EXPECT_TRUE(opened.count(s.parent[p])) << "a piece not hanging from an opened node";
            leaves += s.leaf_count[p];
        }
        EXPECT_EQ(leaves, s.leaf_count[w.root]) << "the pieces do not share out the window's leaves";
        for (int32_t u : skel.opened) {
            EXPECT_GE(u, n);
            for (int32_t c : {tree.c1[u], tree.c2[u]})
                EXPECT_TRUE(opened.count(c) || pieces.count(c)) << "an opened node's child is neither";
            if (w.stops && u != w.root) {
                EXPECT_FALSE(w.stops[u]) << "opened a component root";
            }
        }
        if (k < WINDOW_MIN_K) {
            EXPECT_TRUE(skel.v.empty());
            continue;
        }
        // the vector decodes into the window's clades: tip a is piece perm[a] (the outgroup is piece k_in)
        ASSERT_EQ((int32_t)skel.v.size(), k - 1);
        std::vector<int32_t> sorted = skel.perm;
        std::sort(sorted.begin(), sorted.end());
        ASSERT_EQ(sorted, identity(k));
        std::vector<int32_t> d1(2 * k - 1), d2(2 * k - 1);
        to_skeleton(skel.v.data(), k, d1.data(), d2.data());
        std::set<std::vector<int32_t>> got;
        for (int32_t u = k; u < 2 * k - 1; ++u) {
            std::vector<int32_t> c = tips_under(d1, d2, k, u, skel.perm);
            if ((int32_t)c.size() < k || !with_outgroup) // with the outgroup, the skeleton's root is not a tree clade
                got.insert(c);
        }
        std::vector<int32_t> piece_index(2 * n - 1, -1);
        for (int32_t a = 0; a < k_in; ++a)
            piece_index[skel.piece_roots[a]] = a;
        std::set<std::vector<int32_t>> want;
        for (int32_t u : skel.opened) {
            std::vector<int32_t> c, st = {u};
            while (!st.empty()) {
                const int32_t x = st.back();
                st.pop_back();
                if (piece_index[x] >= 0) {
                    c.push_back(piece_index[x]);
                } else {
                    st.push_back(tree.c1[x]);
                    st.push_back(tree.c2[x]);
                }
            }
            std::sort(c.begin(), c.end());
            want.insert(c);
        }
        EXPECT_EQ(got, want) << "window " << w.root;
    }
}

// to_skeleton_tables: the rooted tree is to_skeleton's, the unrooted tables agree with each other, and
// tin/sub_size answer "x under y" as the parent chain does
TEST(Search, SkeletonTablesAreConsistent) {
    std::mt19937_64 rng(13);
    for (int32_t k : {4, 5, 6, 8, 33, 300, 1500}) {
        for (int rep = 0; rep < (k > 100 ? 5 : 50); ++rep) {
            const std::vector<int64_t> v = random_ordered_vector(k, rng);
            SkeletonTables sk;
            to_skeleton_tables(v, sk);
            std::vector<int32_t> c1(2 * k - 1), c2(2 * k - 1);
            to_skeleton(v.data(), k, c1.data(), c2.data());
            EXPECT_EQ(sk.c1, c1);
            EXPECT_EQ(sk.c2, c2);
            const int32_t n_nodes = 2 * k - 2, hang = n_nodes - 1;
            // adjacency: tips have one neighbour, internal nodes three, and every edge is listed from both ends
            for (int32_t u = 0; u < n_nodes; ++u) {
                int32_t deg = 0;
                for (int32_t s = 0; s < 3; ++s) {
                    const int32_t nb = sk.adj_u[3 * u + s];
                    if (nb == -1)
                        continue;
                    ++deg;
                    EXPECT_EQ(std::count(sk.adj_u.begin() + 3 * nb, sk.adj_u.begin() + 3 * nb + 3, u), 1);
                }
                EXPECT_EQ(deg, u < k ? 1 : 3) << "node " << u;
                if (u != hang) {
                    const int32_t p = sk.parent_u[u];
                    EXPECT_EQ(std::count(sk.adj_u.begin() + 3 * u, sk.adj_u.begin() + 3 * u + 3, p), 1);
                }
            }
            EXPECT_EQ(sk.parent_u[hang], -1);
            EXPECT_EQ(sk.adj_u[3 * hang + 0], sk.rc0);
            EXPECT_EQ(sk.adj_u[3 * hang + 1], sk.rc1);
            EXPECT_EQ(sk.adj_u[3 * hang + 2], sk.rc2);
            EXPECT_EQ(sk.other_rc, sk.rc2);
            EXPECT_EQ(sk.sub_size[hang], n_nodes);
            // v_map: the subtree row i merges in, the c1 side of the node it makes
            for (int32_t i = 0; i < k - 1; ++i)
                EXPECT_EQ(sk.v_map[i + 1], c1[2 * k - 2 - i]);
            // "x under y" by tin/sub_size equals y on x's parent chain
            const auto under = [&](int32_t x, int32_t y) {
                return sk.tin[y] <= sk.tin[x] && sk.tin[x] < sk.tin[y] + sk.sub_size[y];
            };
            const auto chain = [&](int32_t x, int32_t y) {
                for (; x != -1; x = sk.parent_u[x])
                    if (x == y)
                        return true;
                return false;
            };
            const int32_t pairs = k <= 33 ? n_nodes * n_nodes : 5000;
            for (int32_t q = 0; q < pairs; ++q) {
                const int32_t x = k <= 33 ? q / n_nodes : (int32_t)(rng() % (uint64_t)n_nodes);
                const int32_t y = k <= 33 ? q % n_nodes : (int32_t)(rng() % (uint64_t)n_nodes);
                EXPECT_EQ(under(x, y), chain(x, y)) << "x " << x << " y " << y;
            }
        }
    }
}

// from_skeleton_tables gives back the same unrooted tree (tips read through the new label mapping), from any root
// edge and in both labellings, and also after a move applied to adj_u
TEST(Search, SkeletonTablesRoundTrip) {
    std::mt19937_64 rng(17);
    long moved_trees_changed = 0;
    for (int32_t k : {4, 5, 8, 33, 300}) {
        for (int rep = 0; rep < (k > 100 ? 5 : 30); ++rep) {
            const std::vector<int64_t> v = random_ordered_vector(k, rng);
            SkeletonTables sk;
            to_skeleton_tables(v, sk);
            // tip a stands for piece start[a], as after earlier sweeps
            std::vector<int32_t> start = identity(k);
            std::shuffle(start.begin(), start.end(), rng);
            const auto want = splits_of_rooted(sk.c1, sk.c2, k, start);
            const int32_t n_nodes = 2 * k - 2;
            for (int32_t r1 = 0; r1 < n_nodes; r1 += (k > 33 ? 7 : 1)) {
                for (int32_t s = 0; s < (r1 < k ? 1 : 3); ++s) {
                    const int32_t r2 = sk.adj_u[3 * r1 + s];
                    for (int32_t flip : {0, 1}) {
                        std::vector<int64_t> v2(k - 1);
                        std::vector<int32_t> perm = start;
                        from_skeleton_tables(v2, sk, r1, r2, flip, perm);
                        std::vector<int32_t> d1(2 * k - 1), d2(2 * k - 1);
                        to_skeleton(v2.data(), k, d1.data(), d2.data());
                        EXPECT_EQ(splits_of_rooted(d1, d2, k, perm), want) << "root edge " << r1 << "-" << r2;
                    }
                }
            }
            // one move: prune a neighbour of an internal anchor, regraft on an edge on the anchor's side
            const int32_t anchor = k + (int32_t)(rng() % (uint64_t)(n_nodes - k));
            const int32_t prune = sk.adj_u[3 * anchor + (int32_t)(rng() % 3)];
            const auto under = [&](int32_t x, int32_t y) {
                return sk.tin[y] <= sk.tin[x] && sk.tin[x] < sk.tin[y] + sk.sub_size[y];
            };
            for (int attempt = 0; attempt < 100; ++attempt) {
                const int32_t du = (int32_t)(rng() % (uint64_t)(n_nodes - 1)), dv = sk.parent_u[du];
                if (du == anchor || dv == anchor)
                    continue;
                const bool on_anchor_side = sk.parent_u[prune] == anchor ? !under(du, prune) : under(du, anchor);
                if (!on_anchor_side)
                    continue;
                SkeletonTables moved = sk;
                apply_unrooted_spr(moved.adj_u.data(), prune, anchor, du, dv);
                const auto moved_splits = splits_of_adjacency(moved.adj_u, k, start);
                moved_trees_changed += moved_splits != want;
                const int32_t hang = n_nodes - 1;
                std::vector<int64_t> v2(k - 1);
                std::vector<int32_t> perm = start;
                from_skeleton_tables(v2, moved, hang, moved.adj_u[3 * hang], 0, perm);
                std::vector<int32_t> d1(2 * k - 1), d2(2 * k - 1);
                to_skeleton(v2.data(), k, d1.data(), d2.data());
                EXPECT_EQ(splits_of_rooted(d1, d2, k, perm), moved_splits) << "after a move";
                break;
            }
        }
    }
    EXPECT_GT(moved_trees_changed, 0) << "no move changed the tree: the move test checked nothing";
}
