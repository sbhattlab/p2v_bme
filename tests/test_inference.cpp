// infer() on the CPU (runs in CPU-only builds):
//   - brute force: at N = 4..8 the search finds the shortest tree of all (2N - 5)!! topologies, by Pauplin's
//     formula on the model's distances, for several models and seeds;
//   - the reference library (src/reference): the final tree of a search at N = 40 is SPR-optimal under the same
//     distance matrix (no coordinate move of either Phylo2Vec gauge lowers the BME length);
//   - the same tree, branch lengths and gains for 1, 2 or 3 CPU workers (the snapshot rule).
#include <gtest/gtest.h>

#include <stdint.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "bme_reference.h"
#include "bme_test_helpers.hpp"
#include "infer_test_helpers.hpp"
#include "inference.hpp"
#include "phylo2vec.hpp"
#include "simulations.hpp"

namespace {

using namespace p2vbme;

// The smallest BME length over all unrooted binary topologies on N leaves (stepwise insertion of each leaf into
// every edge).
double brute_force_min(int32_t N, const std::vector<double> &D) {
    std::vector<std::vector<int32_t>> adj(N); /* leaves 0..N-1, then internal nodes */
    const int32_t centre = N;
    adj.push_back({});
    for (int32_t x = 0; x < 3; ++x) {
        adj[x].push_back(centre);
        adj[centre].push_back(x);
    }
    double best = 1e300;
    std::function<void(int32_t)> add = [&](int32_t k) {
        if (k == N) {
            best = std::min(best, pauplin(adj, N, D));
            return;
        }
        std::vector<std::pair<int32_t, int32_t>> edges;
        for (int32_t u = 0; u < (int32_t)adj.size(); ++u)
            for (int32_t v : adj[u])
                if (u < v)
                    edges.push_back({u, v});
        for (const auto &e : edges) {
            const int32_t u = e.first, v = e.second, w = (int32_t)adj.size();
            adj.push_back({u, v, k});
            std::replace(adj[u].begin(), adj[u].end(), v, w);
            std::replace(adj[v].begin(), adj[v].end(), u, w);
            adj[k].push_back(w);
            add(k + 1);
            adj[k].clear();
            std::replace(adj[u].begin(), adj[u].end(), w, v);
            std::replace(adj[v].begin(), adj[v].end(), w, u);
            adj.pop_back();
        }
    };
    add(3);
    return best;
}

} // namespace

TEST(Inference, BruteForceAtSmallN) {
    for (int32_t N : {4, 5, 6, 7, 8})
        for (uint64_t seed : {1, 2, 3})
            for (Model model : {Model::JC69, Model::TN93, Model::LogDet, Model::GTR_GC}) {
                const Tree tree = random_tree(N, 0.02, 0.4, 100 * N + seed);
                const Alignment aln = simulate_alignment(tree, 400, truth(), 7 * seed);
                Settings opt = cpu_settings(model);
                opt.seed = seed;
                const Result res = run_infer(aln, opt);
                const std::vector<double> D = distance_matrix(aln, with_base_frequencies(aln, opt).substitution_model);
                const double found = pauplin(unrooted(res.tree, N), N, D), best = brute_force_min(N, D);
                EXPECT_LE(found, best + 1e-6 * best) << "N=" << N << " seed=" << seed << " " << model_name(model);
            }
}

// No coordinate move of the final tree (both gauges, every regraft j <= 2i) lowers the BME length, by the
// reference library's exact O(N^2) differences.
TEST(Inference, SprOptimalByTheReferenceLibrary) {
    const int32_t N = 40;
    for (Model model : {Model::JC69, Model::F84, Model::GTR_GC})
        for (Precision precision : {Precision::Single, Precision::Double}) {
            const Tree tree = random_tree(N, 0.01, 0.3, 41);
            const Alignment aln = simulate_alignment(tree, 1000, truth(), 42);
            const Settings opt = cpu_settings(model, precision);
            const Result res = run_infer(aln, opt);
            const std::vector<double> D = distance_matrix(aln, with_base_frequencies(aln, opt).substitution_model);
            double min_delta = 0.0;
            for (int32_t flip : {0, 1}) {
                std::vector<int64_t> v(N - 1);
                std::vector<int32_t> perm(N);
                queue_shuffle_rooted(res.tree.c1.data(), res.tree.c2.data(), N, flip, v.data(), perm.data());
                std::vector<double> Dp((size_t)N * N);
                for (int32_t a = 0; a < N; ++a)
                    for (int32_t b = 0; b < N; ++b)
                        Dp[(size_t)a * N + b] = D[(size_t)perm[a] * N + perm[b]];
                std::vector<double> delta((size_t)(N - 1) * (2 * N - 3));
                calculate_discrete_gradient_full_c(v.data(), Dp.data(), N, delta.data());
                for (int32_t i = 0; i < N - 1; ++i)
                    for (int32_t j = 0; j <= 2 * i; ++j)
                        min_delta = std::min(min_delta, delta[(size_t)i * (2 * N - 3) + j]);
            }
            const double L = pauplin(unrooted(res.tree, N), N, D);
            /* single precision: the move scores are floats (a move must gain more than 1e-6) */
            const double tol = precision == Precision::Double ? 1e-6 : 1e-6 + 1e-6 * L;
            EXPECT_GE(min_delta, -tol) << model_name(model) << (precision == Precision::Double ? " double" : " single")
                                       << ": an improving move remains (L = " << L << ")";
            free_phylo2vec_workspace_c();
        }
}

// Several CPU workers: the same tree, branch lengths and gains as one (the snapshot rule).
TEST(Inference, SameTreeForAnyWorkerCount) {
    const Tree tree = random_tree(2500, 0.005, 0.08, 61);
    const Alignment aln = simulate_alignment(tree, 600, truth(), 62);
    Result one;
    for (int32_t workers : {1, 2, 3}) {
        Settings opt = cpu_settings(Model::JC69, Precision::Single, workers);
        opt.tree_init = TreeInit::Sample;
        const Result res = run_infer(aln, opt);
        if (workers == 1) {
            one = res;
            continue;
        }
        EXPECT_EQ(res.tree.c1, one.tree.c1) << workers << " workers";
        EXPECT_EQ(res.tree.c2, one.tree.c2) << workers << " workers";
        EXPECT_EQ(res.tree.branch_lengths, one.tree.branch_lengths) << workers << " workers";
        EXPECT_EQ(res.delta_bme, one.delta_bme) << workers << " workers";
        EXPECT_EQ(res.rounds_run, one.rounds_run) << workers << " workers";
    }
}
