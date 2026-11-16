// The GPU's distances under every model against the CPU formulas (substitution_models.hpp, eigen_dist.hpp) on the
// same pair counts. Label gpu.
//   - single pairs and profile-weighted sums through the edge-length pass (the JC69 tiled path and the
//     per-pair path of the other models), on an alignment with missing sites and a pair without shared sites;
//   - the BME length of a fixed tree (no search, complete profiles) against Pauplin's formula on the CPU;
//   - a search from a random start ends at a tree no longer than the true tree (both scored by the same branch-
//     length pass, profile cap 32) and near it in RF; the true model recovers it best (the window distances;
//     N = 3,000 gives multi-leaf profiles), also with the exact check on every window;
//   - the exact check's refinement kernel against a CPU sum on complete profiles (per-pair and JC69 tiled paths).
#include <gtest/gtest.h>

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include "device.hpp"
#include "eigen_dist.hpp"
#include "infer_test_helpers.hpp"
#include "inference.hpp"
#include "random.hpp"
#include "search.hpp"
#include "simulations.hpp"
#include "substitution_models.hpp"

namespace {

using namespace p2vbme;

const Model MODELS[] = {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC};

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

// Clears each site of each sequence with probability `rate` (missing data), and makes sequences 0 and 1 share no
// valid site (0 keeps the first half, 1 the second).
void punch_holes(Alignment &aln, double rate, uint64_t seed) {
    uint64_t st = seed;
    const int32_t W = aln.n_words;
    for (int32_t i = 0; i < aln.n_taxa; ++i) {
        uint64_t *row = &aln.packed[(size_t)i * 5 * W];
        for (int32_t s = 0; s < aln.n_sites; ++s) {
            bool clear = (double)(splitmix64(st) >> 11) * 0x1.0p-53 < rate;
            if (i == 0)
                clear = clear || s >= aln.n_sites / 2;
            if (i == 1)
                clear = clear || s < aln.n_sites / 2;
            if (clear)
                for (int p = 0; p < 5; ++p)
                    row[(size_t)p * W + s / 64] &= ~(1ULL << (s % 64));
        }
    }
}

double cpu_distance(const Alignment &aln, const SubstitutionModel &dm, int32_t a, int32_t b) {
    if (a == b)
        return 0.0;
    const int32_t W = aln.n_words;
    const PairCounts c = count_pair(&aln.packed[(size_t)a * 5 * W], &aln.packed[(size_t)b * 5 * W], W);
    return pair_distance<double>(distance_params<double>(dm, EigenSettings()), c.n);
}

// Exact BME length of a rooted binary tree (Tree layout) by Pauplin's formula: sum_{i<j} 2^(1 - edges(i, j)) d_ij,
// the root's two child edges counted as one.
double pauplin_length(const Alignment &aln, const SubstitutionModel &dm, const Tree &t) {
    const int32_t N = aln.n_taxa, n_nodes = 2 * N - 1;
    std::vector<std::vector<int32_t>> adj(n_nodes);
    for (int32_t u = N; u < n_nodes; ++u) {
        adj[u].push_back(t.c1[u]);
        adj[u].push_back(t.c2[u]);
        adj[t.c1[u]].push_back(u);
        adj[t.c2[u]].push_back(u);
    }
    const int32_t root = n_nodes - 1;
    double total = 0.0;
    for (int32_t i = 0; i < N; ++i) {
        std::vector<int32_t> dist(n_nodes, -1);
        std::queue<int32_t> q;
        dist[i] = 0;
        q.push(i);
        while (!q.empty()) {
            const int32_t u = q.front();
            q.pop();
            for (int32_t v : adj[u])
                if (dist[v] < 0) {
                    dist[v] = dist[u] + (v == root ? 0 : 1); /* the root's two edges are one edge */
                    q.push(v);
                }
        }
        for (int32_t j = i + 1; j < N; ++j)
            total += ldexp(1.0, 1 - dist[j]) * cpu_distance(aln, dm, i, j);
    }
    return total;
}

// Splits of a rooted tree as sorted leaf sets (each side once, by the side without leaf 0), for RF.
std::vector<std::vector<int32_t>> splits(const Tree &t, int32_t N) {
    const int32_t n_nodes = 2 * N - 1;
    std::vector<std::vector<int32_t>> below(n_nodes);
    std::vector<int32_t> order;
    std::vector<int32_t> stack = {n_nodes - 1};
    while (!stack.empty()) {
        const int32_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        if (u >= N) {
            stack.push_back(t.c1[u]);
            stack.push_back(t.c2[u]);
        }
    }
    std::vector<std::vector<int32_t>> out;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int32_t u = *it;
        if (u < N)
            below[u] = {u};
        else {
            below[u] = below[t.c1[u]];
            below[u].insert(below[u].end(), below[t.c2[u]].begin(), below[t.c2[u]].end());
        }
        std::vector<int32_t> s = below[u];
        if ((int32_t)s.size() < 2 || (int32_t)s.size() > N - 2)
            continue;
        if (std::find(s.begin(), s.end(), 0) != s.end()) { /* the complement */
            std::vector<bool> in(N, false);
            for (int32_t x : s)
                in[x] = true;
            s.clear();
            for (int32_t x = 0; x < N; ++x)
                if (!in[x])
                    s.push_back(x);
        }
        std::sort(s.begin(), s.end());
        out.push_back(s);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

int32_t rf_distance(const Tree &a, const Tree &b, int32_t N) {
    const auto sa = splits(a, N), sb = splits(b, N);
    std::vector<std::vector<int32_t>> common;
    std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(common));
    return (int32_t)(sa.size() + sb.size() - 2 * common.size());
}

} // namespace

// Single pairs (one leaf on each side) and weighted profiles (the JC69 tiled path; the per-pair path otherwise).
TEST(DeviceDistances, EdgeLengthPassMatchesCpu) {
    const int32_t N = 60;
    const Tree tree = random_tree(N, 0.01, 0.3, 11);
    Alignment aln = simulate_alignment(tree, 3000, truth(), 12);
    punch_holes(aln, 0.1, 13);
    auto dev = make_gpu_device(0);
    dev->load(aln.packed.data(), N, aln.n_words);

    // jobs 0..N-1: pairs (i, i + 1 mod N) and outside leaf i + 2 (job 0: leaves 0 and 1, which share no site)
    std::vector<int32_t> c1(N), c2(N), out_len(N, 1), out_leaves(N), in_len(2, 0);
    std::vector<float> out_w(N, 1.0f);
    for (int32_t i = 0; i < N; ++i) {
        c1[i] = i;
        c2[i] = (i + 1) % N;
        out_leaves[i] = (i + 2) % N;
    }
    // job N: two "internal nodes" with profiles of 40 and 30 leaves and an outside profile of 25 leaves
    const int32_t cap = 40;
    std::vector<int32_t> in_leaves(2 * cap, 0), out_leaves_p(cap, 0);
    std::vector<float> in_w(2 * cap, 0.0f), out_w_p(cap, 0.0f);
    in_len = {40, 30};
    for (int32_t k = 0; k < 40; ++k) {
        in_leaves[k] = k;
        in_w[k] = (k + 1) / 820.0f;
    }
    for (int32_t k = 0; k < 30; ++k) {
        in_leaves[cap + k] = 30 + k;
        in_w[cap + k] = 1.0f / 30.0f;
    }
    for (int32_t k = 0; k < 25; ++k) {
        out_leaves_p[k] = 2 * k + 1;
        out_w_p[k] = 1.0f / 25.0f;
    }

    for (Model model : MODELS) {
        const SubstitutionModel dm = distance_model(model);
        dev->set_distance(dm, EigenSettings());
        std::vector<double> dAB(N), dAO(N), dBO(N);
        const double tol = model == Model::GTR_GC ? 2e-4 : 2e-5; /* float on the GPU, double here */
        // single pairs: cap 1 for the outside profiles (in_* unused)
        dev->edge_lengths(N, 1, N, c1.data(), c2.data(), 2, in_len.data(), in_leaves.data(), in_w.data(),
                          out_len.data(), out_leaves.data(), out_w.data(), dAB.data(), dAO.data(), dBO.data());
        for (int32_t i = 0; i < N; ++i) {
            const double ab = cpu_distance(aln, dm, c1[i], c2[i]), ao = cpu_distance(aln, dm, c1[i], out_leaves[i]);
            EXPECT_NEAR(dAB[i], ab, tol * ab + 1e-6) << model_name(model) << " pair " << i;
            EXPECT_NEAR(dAO[i], ao, tol * ao + 1e-6) << model_name(model) << " pair " << i;
        }
        const double none = model == Model::JC69 ? 1.5 : 2.0;
        EXPECT_EQ(dAB[0], (float)none) << model_name(model) << ": leaves 0 and 1 share no site";

        // profiles
        const int32_t pc1 = N, pc2 = N + 1;
        double pAB, pAO, pBO;
        dev->edge_lengths(N, cap, 1, &pc1, &pc2, 2, in_len.data(), in_leaves.data(), in_w.data(),
                          std::vector<int32_t>{25}.data(), out_leaves_p.data(), out_w_p.data(), &pAB, &pAO, &pBO);
        double eAB = 0.0, eAO = 0.0, eBO = 0.0;
        for (int32_t p = 0; p < 40; ++p)
            for (int32_t q = 0; q < 30; ++q)
                eAB += (double)in_w[p] * in_w[cap + q] * cpu_distance(aln, dm, in_leaves[p], in_leaves[cap + q]);
        for (int32_t q = 0; q < 25; ++q) {
            for (int32_t p = 0; p < 40; ++p)
                eAO += (double)in_w[p] * out_w_p[q] * cpu_distance(aln, dm, in_leaves[p], out_leaves_p[q]);
            for (int32_t p = 0; p < 30; ++p)
                eBO += (double)in_w[cap + p] * out_w_p[q] * cpu_distance(aln, dm, in_leaves[cap + p], out_leaves_p[q]);
        }
        EXPECT_NEAR(pAB, eAB, 1e-4 * eAB) << model_name(model);
        EXPECT_NEAR(pAO, eAO, 1e-4 * eAO) << model_name(model);
        EXPECT_NEAR(pBO, eBO, 1e-4 * eBO) << model_name(model);
    }
}

// Branch lengths of a fixed tree from complete profiles sum to the tree's exact BME length.
TEST(DeviceDistances, BmeLengthOfAGivenTree) {
    const int32_t N = 48;
    const Tree tree = random_tree(N, 0.02, 0.2, 21);
    Alignment aln = simulate_alignment(tree, 4000, truth(), 22);
    punch_holes(aln, 0.05, 23);
    for (Model model : MODELS) {
        Settings opt;
        opt.start = Start::Given;
        opt.max_rounds = 0;
        opt.bl_profile_cap = PROFILE_CAP_MAX;
        opt.devices = {0};
        opt.substitution = distance_model(model);
        const Result res = run_infer(aln, opt, &tree);
        const double exact = pauplin_length(aln, res.substitution, tree); /* the model as used (base frequencies) */
        EXPECT_NEAR(res.bme_length, exact, 1e-4 * exact) << model_name(model);
    }
}

// A search ends no longer than the true tree (same branch-length pass, profile cap 32); near it in RF.
TEST(DeviceDistances, SearchUnderEachModel) {
    const int32_t N = 3000;
    const Tree tree = random_tree(N, 0.005, 0.08, 31);
    const Alignment aln = simulate_alignment(tree, 2000, truth(), 32);
    for (Model model : MODELS) {
        for (double eta : {0.0, 1e30}) {
            if (eta > 0.0 && model != Model::TN93 && model != Model::GTR_GC)
                continue; /* the exact check's per-pair path: two models are enough */
            Settings opt;
            opt.start = Start::Random;
            opt.devices = {0};
            opt.exact_check_eta = eta;
            if (eta > 0.0)
                opt.profile_cap = 8; /* capped profiles, so the check runs (on complete profiles) */
            opt.substitution = distance_model(model);
            const Result res = run_infer(aln, opt);
            EXPECT_LT(res.delta_bme, 0.0) << model_name(model); /* the tree got shorter */
            // the true tree, scored by the same pass
            Settings score = opt;
            score.start = Start::Given;
            score.max_rounds = 0;
            const Result true_score = run_infer(aln, score, &tree);
            EXPECT_LE(res.bme_length, true_score.bme_length * (1.0 + 1e-4))
                << model_name(model) << " eta=" << eta << ": found " << res.bme_length << ", true tree "
                << true_score.bme_length;
            const int32_t rf = rf_distance(res.tree, tree, N);
            EXPECT_LT(rf, N / 5) << model_name(model) << " eta=" << eta << " RF " << rf;
            /* the true model (GTR+GC with the true parameters) recovers the tree far better than the others (RF
             * about 160): this fails if the window solve does not use the model */
            if (model == Model::GTR_GC) {
                EXPECT_LT(rf, N / 30) << "GTR+GC eta=" << eta << " RF " << rf;
            }
            printf("%-7s eta=%g: BME %.6f (true tree %.6f), RF %d of %d, rounds %d\n", model_name(model), eta,
                   res.bme_length, true_score.bme_length, rf, 2 * (N - 3), res.rounds_run);
        }
    }
}

// The exact check's refinement kernel: a window whose profiles are capped (3 of each component's 10 leaves), solved
// with the check on every window, returns the gain on the complete profiles, sum over component pairs of
// (2^(1 - edges after) - 2^(1 - edges before)) x D_ab, D_ab the profile-weighted distance. Recomputed here on
// the CPU. n_words 16 takes the kernel's per-pair path (every model); 20 (a multiple of 5) JC69's tiled path.
TEST(DeviceDistances, ExactCheckRefinement) {
    const int32_t N = 400, K = 40, G = 10, CAPPED = 3;
    for (int32_t n_sites : {1000, 1280}) {
        const Tree tree = random_tree(N, 0.01, 0.2, 41);
        Alignment aln = simulate_alignment(tree, n_sites, truth(), 42);
        punch_holes(aln, 0.05, 43);
        auto dev = make_gpu_device(0);
        dev->load(aln.packed.data(), N, aln.n_words);
        std::vector<int32_t> cap_off(K + 1), cap_leaves, ex_off(K + 1), ex_leaves;
        std::vector<float> cap_w, ex_w;
        for (int32_t a = 0; a < K; ++a) {
            cap_off[a] = (int32_t)cap_leaves.size();
            ex_off[a] = (int32_t)ex_leaves.size();
            for (int32_t k = 0; k < G; ++k) {
                ex_leaves.push_back(G * a + k);
                ex_w.push_back((k + 1) / 55.0f); /* unequal weights (sum 1), so a wrong leaf shows */
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
        bool jc69_seen = false;
        for (Model model : MODELS) {
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
                continue; /* rejected or nothing found: no exact gain to compare */
            ++accepted;
            if (model == Model::JC69)
                jc69_seen = true;
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
                    double D = 0.0;
                    for (int32_t p = ex_off[a]; p < ex_off[a + 1]; ++p)
                        for (int32_t q = ex_off[b]; q < ex_off[b + 1]; ++q)
                            D += (double)ex_w[p] * ex_w[q] * cpu_distance(aln, dm, ex_leaves[p], ex_leaves[q]);
                    expected += wd * D;
                    scale += fabs(wd * D);
                }
            const double tol = (model == Model::GTR_GC ? 1e-5 : 1e-6) * scale; /* seen: 3e-7 and 1e-8 at scale 20-70 */
            EXPECT_NEAR(dL, expected, tol) << model_name(model) << " n_sites=" << n_sites;
            printf("%-7s n_sites=%d: exact gain %.9g, CPU %.9g (scale %.4g)\n", model_name(model), n_sites, dL,
                   expected, scale);
        }
        EXPECT_GE(accepted, 5) << "n_sites=" << n_sites << ": too few windows to compare";
        EXPECT_TRUE(jc69_seen) << "n_sites=" << n_sites << ": the JC69 path was not compared";
    }
}

// The CPU device against the GPU: the same search on the same data gives the same BME length up
// to single-precision rounding, under every model.
TEST(DeviceDistances, CpuAgreesWithGpu) {
    const int32_t N = 3000;
    const Tree tree = random_tree(N, 0.005, 0.08, 101);
    const Alignment aln = simulate_alignment(tree, 1000, truth(), 102);
    for (Model model : MODELS) {
        Settings opt;
        opt.devices = {0};
        opt.substitution = distance_model(model);
        opt.accelerator = Accelerator::Gpu;
        Devices gpu_devices = devices_for(opt);
        const Result gpu = infer(aln, opt, gpu_devices);
        opt.accelerator = Accelerator::Cpu;
        Devices cpu_devices = devices_for(opt);
        const Result cpu = infer(aln, opt, cpu_devices);
        /* measured: <= 8.4e-7 relative and RF 0 under every model */
        EXPECT_NEAR(cpu.bme_length, gpu.bme_length, 1e-5 * gpu.bme_length) << model_name(model);
        const int32_t rf = rf_distance(cpu.tree, gpu.tree, N);
        EXPECT_LE(rf, 10) << model_name(model);
        EXPECT_EQ(gpu_devices[0]->name().rfind("GPU ", 0), 0u) << gpu_devices[0]->name(); // the run used the GPU
        EXPECT_EQ(cpu_devices[0]->name().rfind("CPU ", 0), 0u) << cpu_devices[0]->name(); // and the CPU
        printf("%-7s GPU %.6f CPU %.6f (rel %.1e), RF %d\n", model_name(model), gpu.bme_length, cpu.bme_length,
               fabs(cpu.bme_length - gpu.bme_length) / gpu.bme_length, rf);
    }
}
