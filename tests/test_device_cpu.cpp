// The CPU device's window kernels (make_cpu_window_kernels), on the CPU only (runs in CPU-only builds):
//   - each row's best move and its score equal the smallest exact BME change, each move applied and rescored;
//   - with identical sequences every score is exactly 0 and each row's best move is the smallest edge (the GPU
//     reduction's tie-break).
#include <gtest/gtest.h>

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "bme_test_helpers.hpp"
#include "eigen_dist.hpp"
#include "phylo2vec.hpp"
#include "search.hpp"
#include "simulations.hpp"

using namespace p2vbme;

// The move scores: for each pruned subtree (row) of a skeleton, the best regraft and its score equal the smallest
// exact BME change over every regraft the recursion covers (each move applied and the tree scored anew).
TEST(CpuDevice, MoveScoresAreExactBmeChanges) {
    const int32_t K = 14, M = 2 * K - 2, root_u = M - 1;
    const Tree tree = random_tree(K, 0.02, 0.3, 91);
    const Alignment aln = simulate_alignment(tree, 2000, truth(), 92);
    std::vector<int32_t> off(K + 1), leaves(K);
    std::vector<float> weights(K, 1.0f);
    for (int32_t a = 0; a <= K; ++a)
        off[a] = a;
    for (int32_t a = 0; a < K; ++a)
        leaves[a] = a;
    int compared = 0;
    for (Model model : {Model::JC69, Model::TN93, Model::GTR_GC})
        for (uint64_t seed : {1, 2, 3}) {
            auto kern = make_cpu_window_kernels(aln.packed.data(), aln.n_words, distance_model(model), true);
            kern->distances(K, off.data(), leaves.data(), weights.data());
            const Tree start = random_tree(K, 0.1, 0.1, 1000 + seed);
            std::vector<int64_t> v(K - 1);
            std::vector<int32_t> perm(K);
            queue_shuffle_rooted(start.c1.data(), start.c2.data(), K, 0, v.data(), perm.data());
            auto meta = [&](const std::vector<int64_t> &vv) {
                SkeletonTables s;
                to_skeleton_tables(vv, s);
                return s;
            };
            const SkeletonTables s = meta(v);
            const double L0 = kern->score(s, perm);
            std::vector<double> best_delta;
            std::vector<int32_t> best_j;
            kern->best_moves(s, best_delta, best_j);
            for (int32_t i = 1; i < K - 1; ++i) {
                const int32_t tr = s.v_map[i + 1];
                const int32_t pr = (tr == root_u) ? s.rc2 : s.parent_u[tr];
                /* the regraft edges: those of the two half trees beyond the pruning point's other neighbours */
                std::vector<int32_t> edges; /* by child node */
                for (int k = 0; k < 3; ++k) {
                    const int32_t start_node = s.adj_u[3 * pr + k];
                    if (start_node == -1 || start_node == tr)
                        continue;
                    std::vector<std::pair<int32_t, int32_t>> st = {{start_node, pr}};
                    while (!st.empty()) {
                        const auto [u, from] = st.back();
                        st.pop_back();
                        for (int m = 0; m < 3; ++m) {
                            const int32_t w = s.adj_u[3 * u + m];
                            if (w == -1 || w == from)
                                continue;
                            edges.push_back(s.parent_u[u] == w ? u : w);
                            st.push_back({w, u});
                        }
                    }
                }
                double best = 1e300;
                int32_t best_edge = -1;
                for (int32_t e : edges) {
                    SkeletonTables moved = s;
                    apply_unrooted_spr(moved.adj_u.data(), tr, pr, e, s.parent_u[e]);
                    std::vector<int64_t> v2 = v;
                    std::vector<int32_t> perm2 = perm;
                    from_skeleton_tables(v2, moved, root_u, moved.adj_u[3 * root_u + 0], 0, perm2);
                    const double d = kern->score(meta(v2), perm2) - L0;
                    if (d < best || (d == best && e < best_edge)) {
                        best = d;
                        best_edge = e;
                    }
                }
                if (edges.empty())
                    continue;
                ++compared;
                EXPECT_NEAR(best_delta[i], best, 1e-9 * L0) << model_name(model) << " seed " << seed << " row " << i;
                if (fabs(best_delta[i] - best) < 1e-9 * L0) {
                    EXPECT_EQ(best_j[i], best_edge) << model_name(model) << " seed " << seed << " row " << i;
                }
            }
        }
    EXPECT_GE(compared, 30);
}

// Ties: with identical sequences every regraft changes nothing (all scores exactly 0), and each row's best move is
// the smallest edge the recursion scores (the GPU reduction's tie-break).
TEST(CpuDevice, MoveTiesGoToTheSmallestEdge) {
    const int32_t K = 12, M = 2 * K - 2, root_u = M - 1;
    Alignment aln = simulate_alignment(random_tree(K, 0.1, 0.1, 111), 200, truth(), 112);
    for (int32_t x = 1; x < K; ++x) /* every sequence a copy of the first */
        std::copy(aln.packed.begin(), aln.packed.begin() + N_PLANES * aln.n_words,
                  aln.packed.begin() + (size_t)x * N_PLANES * aln.n_words);
    std::vector<int32_t> off(K + 1), leaves(K);
    std::vector<float> weights(K, 1.0f);
    for (int32_t a = 0; a <= K; ++a)
        off[a] = a;
    for (int32_t a = 0; a < K; ++a)
        leaves[a] = a;
    for (bool double_precision : {false, true}) {
        auto kern =
            make_cpu_window_kernels(aln.packed.data(), aln.n_words, distance_model(Model::JC69), double_precision);
        kern->distances(K, off.data(), leaves.data(), weights.data());
        const Tree start = random_tree(K, 0.1, 0.1, 113);
        std::vector<int64_t> v(K - 1);
        std::vector<int32_t> perm(K);
        queue_shuffle_rooted(start.c1.data(), start.c2.data(), K, 0, v.data(), perm.data());
        SkeletonTables s;
        to_skeleton_tables(v, s);
        kern->score(s, perm);
        std::vector<double> best_delta;
        std::vector<int32_t> best_j;
        kern->best_moves(s, best_delta, best_j);
        for (int32_t i = 1; i < K - 1; ++i) {
            const int32_t tr = s.v_map[i + 1];
            const int32_t pr = (tr == root_u) ? s.rc2 : s.parent_u[tr];
            /* the smallest edge (by child node) beyond the pruning point's other neighbours, not at the pruning point
             */
            int32_t smallest = INT32_MAX;
            for (int k = 0; k < 3; ++k) {
                const int32_t start_node = s.adj_u[3 * pr + k];
                if (start_node == -1 || start_node == tr)
                    continue;
                std::vector<std::pair<int32_t, int32_t>> st = {{start_node, pr}};
                while (!st.empty()) {
                    const auto [u, from] = st.back();
                    st.pop_back();
                    for (int m = 0; m < 3; ++m) {
                        const int32_t w = s.adj_u[3 * u + m];
                        if (w == -1 || w == from)
                            continue;
                        smallest = std::min(smallest, s.parent_u[u] == w ? u : w);
                        st.push_back({w, u});
                    }
                }
            }
            if (smallest == INT32_MAX)
                continue;
            EXPECT_EQ(best_delta[i], 0.0) << "row " << i;
            EXPECT_EQ(best_j[i], smallest) << "row " << i;
        }
    }
}
