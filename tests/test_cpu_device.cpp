// The CPU device, on the CPU only (runs in CPU-only builds):
//   - brute force: at N = 6, 7, 8 the search finds the shortest tree of all (2N - 5)!! topologies, by Pauplin's
//     formula on the model's distances, for several models and seeds;
//   - the reference library (src/reference): the final tree of a search at N = 40 is SPR-optimal under the same
//     distance matrix (no coordinate move of either Phylo2Vec gauge lowers the BME length);
//   - its BME length of a fixed tree (complete profiles) equals Pauplin's formula, to float precision in single
//     and to 1e-9 in double precision, and the Newick output keeps double digits;
//   - the same tree for 1, 2 or 3 CPU workers (the snapshot rule), and the exact check's gain.
#include <gtest/gtest.h>

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "bme_reference.h"
#include "device.hpp"
#include "eigen_dist.hpp"
#include "infer_test_helpers.hpp"
#include "inference.hpp"
#include "newick.hpp"
#include "search.hpp"
#include "simulations.hpp"
#include "substitution_models.hpp"

namespace {

using namespace p2vbme;

SimulationModel truth() {
    SimulationModel m;
    const double r[6] = {0.5, 2.0, 0.5, 0.5, 2.0, 0.5}, f[4] = {0.3, 0.2, 0.2, 0.3};
    for (int k = 0; k < 6; ++k)
        m.rates[k] = r[k];
    for (int i = 0; i < 4; ++i)
        m.freqs[i] = f[i];
    m.alpha = 0.5;
    return m;
}

SubstitutionModel distance_model(Model model) {
    SubstitutionModel dm;
    dm.model = model;
    const SimulationModel t = truth();
    for (int k = 0; k < 6; ++k)
        dm.gtr_rates[k] = t.rates[k];
    dm.gtr_alpha = t.alpha;
    return dm;
}

std::vector<double> distance_matrix(const Alignment &aln, const SubstitutionModel &dm) {
    const int32_t N = aln.n_taxa, W = aln.n_words;
    std::vector<double> D((size_t)N * N, 0.0);
    for (int32_t i = 0; i < N; ++i)
        for (int32_t j = i + 1; j < N; ++j) {
            const PairCounts c = count_pair(&aln.packed[(size_t)i * 5 * W], &aln.packed[(size_t)j * 5 * W], W);
            D[(size_t)i * N + j] = D[(size_t)j * N + i] =
                pair_distance<double>(distance_params<double>(dm, EigenSettings()), c.n);
        }
    return D;
}

// Pauplin's BME length of an unrooted tree given as adjacency lists on leaves 0..N-1 and internal nodes.
double pauplin(const std::vector<std::vector<int32_t>> &adj, int32_t N, const std::vector<double> &D) {
    double total = 0.0;
    std::vector<int32_t> dist(adj.size());
    for (int32_t i = 0; i < N; ++i) {
        std::fill(dist.begin(), dist.end(), -1);
        std::queue<int32_t> q;
        dist[i] = 0;
        q.push(i);
        while (!q.empty()) {
            const int32_t u = q.front();
            q.pop();
            for (int32_t v : adj[u])
                if (dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
        }
        for (int32_t j = i + 1; j < N; ++j)
            total += ldexp(1.0, 1 - dist[j]) * D[(size_t)i * N + j];
    }
    return total;
}

// The unrooted tree of a rooted Tree (the root's two edges joined).
std::vector<std::vector<int32_t>> unrooted(const Tree &t, int32_t N) {
    const int32_t root = 2 * N - 2;
    std::vector<std::vector<int32_t>> adj(2 * N - 1);
    for (int32_t u = N; u < root; ++u)
        for (int32_t c : {t.c1[u], t.c2[u]}) {
            adj[u].push_back(c);
            adj[c].push_back(u);
        }
    adj[t.c1[root]].push_back(t.c2[root]);
    adj[t.c2[root]].push_back(t.c1[root]);
    return adj;
}

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

Settings cpu_settings(Model model, Precision precision = Precision::Single, int32_t threads = 2) {
    Settings opt;
    opt.tree_init = TreeInit::Random;
    opt.accelerator = Accelerator::Cpu;
    opt.threads = threads;
    opt.precision = precision;
    opt.substitution_model = distance_model(model);
    return opt;
}

} // namespace

TEST(CpuDevice, BruteForceAtSmallN) {
    for (int32_t N : {6, 7, 8})
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
TEST(CpuDevice, SprOptimalByTheReferenceLibrary) {
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
                relabel_from_rooted_tree(res.tree.c1.data(), res.tree.c2.data(), N, flip, v.data(), perm.data());
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

TEST(CpuDevice, BmeLengthOfAGivenTree) {
    const int32_t N = 48;
    const Tree tree = random_tree(N, 0.02, 0.2, 21);
    const Alignment aln = simulate_alignment(tree, 4000, truth(), 22);
    for (Model model : {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC})
        for (Precision precision : {Precision::Single, Precision::Double}) {
            Settings opt = cpu_settings(model, precision);
            opt.tree_init = TreeInit::Given;
            opt.max_rounds = 0;
            opt.bl_profile_cap = PROFILE_CAP_MAX;
            const Result res = run_infer(aln, opt, &tree);
            const double exact =
                pauplin(unrooted(tree, N), N, distance_matrix(aln, with_base_frequencies(aln, opt).substitution_model));
            const double tol = precision == Precision::Double ? 1e-9 : 1e-4;
            EXPECT_NEAR(res.bme_length, exact, tol * exact) << model_name(model);
            EXPECT_EQ(res.tree.double_precision_lengths, precision == Precision::Double);
        }
}

// --precision double: the Newick lengths keep double digits (shortest round trip of the double values).
TEST(CpuDevice, DoublePrecisionNewick) {
    const int32_t N = 30;
    const Tree tree = random_tree(N, 0.02, 0.2, 51);
    const Alignment aln = simulate_alignment(tree, 2000, truth(), 52);
    Settings opt = cpu_settings(Model::JC69, Precision::Double);
    const Result res = run_infer(aln, opt);
    const std::string path = ::testing::TempDir() + "cpu_double.nwk";
    write_newick_unrooted(path, aln.names, res.tree);
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    size_t long_numbers = 0;
    std::string s = text.str();
    for (size_t pos = s.find(':'); pos != std::string::npos; pos = s.find(':', pos + 1)) {
        const size_t end = s.find_first_of(",);", pos);
        if (end - pos - 1 > 10) /* more digits than a float's shortest round trip */
            ++long_numbers;
    }
    EXPECT_GT(long_numbers, (size_t)N); /* most lengths need double digits */
}

// Several CPU workers: the same tree, branch lengths and gains as one (the snapshot rule).
TEST(CpuDevice, SameTreeForAnyWorkerCount) {
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

// The exact check on the CPU: the returned gain is the gain on complete profiles (as the GPU test).
TEST(CpuDevice, ExactCheck) {
    const int32_t N = 400, K = 40, G = 10, CAPPED = 3;
    const Tree tree = random_tree(N, 0.01, 0.2, 71);
    const Alignment aln = simulate_alignment(tree, 1000, truth(), 72);
    std::vector<int32_t> cap_off(K + 1), cap_leaves, ex_off(K + 1), ex_leaves;
    std::vector<float> cap_w, ex_w;
    for (int32_t a = 0; a < K; ++a) {
        cap_off[a] = (int32_t)cap_leaves.size();
        ex_off[a] = (int32_t)ex_leaves.size();
        for (int32_t k = 0; k < G; ++k) {
            ex_leaves.push_back(G * a + k);
            ex_w.push_back((k + 1) / 55.0f);
            if (k < CAPPED) {
                cap_leaves.push_back(G * a + k);
                cap_w.push_back(1.0f / CAPPED);
            }
        }
    }
    cap_off[K] = (int32_t)cap_leaves.size();
    ex_off[K] = (int32_t)ex_leaves.size();
    const ExactProfileBuilder builder = [&](const int32_t **o, const int32_t **l, const float **w) {
        *o = ex_off.data();
        *l = ex_leaves.data();
        *w = ex_w.data();
    };
    int accepted = 0;
    for (Precision precision : {Precision::Single, Precision::Double})
        for (Model model : {Model::JC69, Model::TN93, Model::GTR_GC}) {
            auto dev = make_cpu_device(0, 2, precision == Precision::Double);
            dev->load(aln.packed.data(), N, aln.n_words);
            const SubstitutionModel dm = distance_model(model);
            dev->set_distance(dm, EigenSettings());
            std::vector<int64_t> v(K - 1, 0), v0 = v;
            std::vector<int32_t> perm(K), perm0;
            for (int32_t a = 0; a < K; ++a)
                perm[a] = a;
            perm0 = perm;
            const double dL = dev->solve_window(K, WINDOW_MAX_SWEEPS, MOVE_BATCH_CAP, 1e30, cap_off.data(),
                                                cap_leaves.data(), cap_w.data(), v, perm, &builder);
            if (dL == 0.0)
                continue;
            ++accepted;
            const std::vector<double> Dl = distance_matrix(aln, dm);
            std::vector<int32_t> before, after;
            compute_skeleton_leaf_dists(v0.data(), perm0.data(), K, before);
            compute_skeleton_leaf_dists(v.data(), perm.data(), K, after);
            double expected = 0.0, scale = 0.0;
            for (int32_t a = 0; a < K; ++a)
                for (int32_t b = a + 1; b < K; ++b) {
                    const double wd =
                        ldexp(1.0, 1 - after[(size_t)a * K + b]) - ldexp(1.0, 1 - before[(size_t)a * K + b]);
                    if (wd == 0.0)
                        continue;
                    double Dab = 0.0;
                    for (int32_t p = ex_off[a]; p < ex_off[a + 1]; ++p)
                        for (int32_t q = ex_off[b]; q < ex_off[b + 1]; ++q)
                            Dab += (double)ex_w[p] * ex_w[q] * Dl[(size_t)ex_leaves[p] * N + ex_leaves[q]];
                    expected += wd * Dab;
                    scale += fabs(wd * Dab);
                }
            const double tol = (precision == Precision::Double ? 1e-12 : model == Model::GTR_GC ? 1e-5 : 1e-6) * scale;
            EXPECT_NEAR(dL, expected, tol) << model_name(model);
        }
    EXPECT_GE(accepted, 4);
}

// Settings the CPU refuses.
TEST(CpuDevice, Errors) {
    const Tree tree = random_tree(20, 0.01, 0.2, 81);
    const Alignment aln = simulate_alignment(tree, 300, truth(), 82);
    Settings opt = cpu_settings(Model::JC69);
    opt.threads = -1;
    EXPECT_THROW(run_infer(aln, opt), std::runtime_error);
    opt = cpu_settings(Model::JC69, Precision::Double);
    opt.accelerator = Accelerator::Gpu;
    EXPECT_THROW(run_infer(aln, opt), std::runtime_error);
}

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
            auto kern =
                make_cpu_window_kernels(aln.packed.data(), aln.n_words, distance_model(model), EigenSettings(), true);
            kern->distances(K, off.data(), leaves.data(), weights.data());
            const Tree start = random_tree(K, 0.1, 0.1, 1000 + seed);
            std::vector<int64_t> v(K - 1);
            std::vector<int32_t> perm(K);
            relabel_from_rooted_tree(start.c1.data(), start.c2.data(), K, 0, v.data(), perm.data());
            auto meta = [&](const std::vector<int64_t> &vv) {
                SkeletonMeta s;
                build_skeleton_metadata(vv.data(), K, s.c1, s.c2, s.v_map, s.m_lmin, s.next_k, s.parent_u, s.sub_size,
                                        s.tin, s.adj_u, &s.other_rc, &s.rc0, &s.rc1, &s.rc2);
                return s;
            };
            const SkeletonMeta s = meta(v);
            const double L0 = kern->score(s, perm);
            std::vector<double> best_delta;
            std::vector<int32_t> best_j;
            kern->best_moves(s, best_delta, best_j);
            for (int32_t i = 1; i < K - 1; ++i) {
                const int32_t tr = s.v_map[i + 1];
                const int32_t pr = (tr == root_u) ? s.other_rc : s.parent_u[tr];
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
                    std::vector<int32_t> adj = s.adj_u;
                    apply_unrooted_spr_host(adj.data(), tr, pr, e, s.parent_u[e]);
                    std::vector<int64_t> v2 = v;
                    std::vector<int32_t> perm2 = perm;
                    relabel_from_adj_u(adj.data(), K, root_u, adj[3 * root_u + 0], 0, v2, perm2);
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
        std::copy(aln.packed.begin(), aln.packed.begin() + 5 * aln.n_words,
                  aln.packed.begin() + (size_t)x * 5 * aln.n_words);
    std::vector<int32_t> off(K + 1), leaves(K);
    std::vector<float> weights(K, 1.0f);
    for (int32_t a = 0; a <= K; ++a)
        off[a] = a;
    for (int32_t a = 0; a < K; ++a)
        leaves[a] = a;
    for (bool double_precision : {false, true}) {
        auto kern = make_cpu_window_kernels(aln.packed.data(), aln.n_words, distance_model(Model::JC69),
                                            EigenSettings(), double_precision);
        kern->distances(K, off.data(), leaves.data(), weights.data());
        const Tree start = random_tree(K, 0.1, 0.1, 113);
        std::vector<int64_t> v(K - 1);
        std::vector<int32_t> perm(K);
        relabel_from_rooted_tree(start.c1.data(), start.c2.data(), K, 0, v.data(), perm.data());
        SkeletonMeta s;
        build_skeleton_metadata(v.data(), K, s.c1, s.c2, s.v_map, s.m_lmin, s.next_k, s.parent_u, s.sub_size, s.tin,
                                s.adj_u, &s.other_rc, &s.rc0, &s.rc1, &s.rc2);
        kern->score(s, perm);
        std::vector<double> best_delta;
        std::vector<int32_t> best_j;
        kern->best_moves(s, best_delta, best_j);
        for (int32_t i = 1; i < K - 1; ++i) {
            const int32_t tr = s.v_map[i + 1];
            const int32_t pr = (tr == root_u) ? s.other_rc : s.parent_u[tr];
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
