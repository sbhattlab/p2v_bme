// infer() on the CPU (runs in CPU-only builds):
//   - brute force: at N = 4..8 the search finds the shortest tree of all (2N - 5)!! topologies, by Pauplin's
//     formula on the model's distances, for several models and seeds;
//   - the reference library (src/reference): the final tree of a search at N = 40 is SPR-optimal under the same
//     distance matrix (no coordinate move of either Phylo2Vec gauge lowers the BME length);
//   - the same tree, branch lengths and gains for 1, 2 or 3 CPU workers (the snapshot rule);
//   - infer()'s checks: bad input, the sample start's fallback, GTR+GC's parameters, the base frequencies (a fake
//     device, so no device is used).
// InferenceGpu.* on GPU 0 (label gpu): a search under every model ends no longer than the true tree and near it in RF;
// the CPU device gives the same BME length as the GPU up to single-precision rounding; the start tree, search and
// branch lengths give the same tree on 1, 2 and 3 devices (several devices share one GPU, each with its own workspace
// and host thread, so the multi-device code paths run on a single card).
#include <gtest/gtest.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "bme_reference.h"
#include "bme_test_helpers.hpp"
#include "branch_lengths.hpp"
#include "config.hpp"
#include "fake_device_test_helpers.hpp"
#include "infer_test_helpers.hpp"
#include "inference.hpp"
#include "phylo2vec.hpp"
#include "search.hpp"
#include "simulations.hpp"
#include "start_tree.hpp"

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

const Model MODELS[] = {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC};

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

struct RunResult {
    Tree tree;
    double delta = 0.0, bme_length = 0.0;
    SearchStats stats;
};

RunResult run(const Alignment &aln, int n_devices, bool sample_start) {
    Devices devices;
    for (int d = 0; d < n_devices; ++d)
        devices.push_back(make_gpu_device(0));
    for (auto &dev : devices)
        dev->load(aln.packed.data(), aln.n_taxa, aln.n_words);
    const int32_t N = aln.n_taxa;
    SearchSettings settings; // the defaults
    RunResult r;
    if (sample_start) {
        int32_t n0 = N / START_SAMPLE_DIVISOR;
        r.tree = build_sample_and_place_tree(N, n0, 1, devices, settings);
    } else {
        r.tree = build_random_tree(N, 1);
    }
    r.delta = run_minibatch_bme(r.tree, devices, settings, 0, &r.stats);
    r.bme_length = compute_bme_branch_lengths(r.tree, devices, r.tree.branch_lengths, DEFAULT_BL_REPRESENTATIVE_CAP);
    return r;
}

// A short fingerprint of a result, printed so runs of different code versions can be compared by hand.
std::string fingerprint(const RunResult &r) {
    size_t h = 0;
    auto mix = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (int32_t x : r.tree.c1)
        mix(std::hash<int32_t>()(x));
    for (int32_t x : r.tree.c2)
        mix(std::hash<int32_t>()(x));
    for (float x : r.tree.branch_lengths)
        mix(std::hash<float>()(x));
    char buf[160];
    snprintf(buf, sizeof(buf), "tree %016zx  dL %.9f  BME %.9f  rounds %d", h, r.delta, r.bme_length,
             r.stats.rounds_run);
    return buf;
}

Alignment simulated(int32_t n, uint64_t seed) {
    const Tree t = random_tree(n, 0.005, 0.08, seed);
    return simulate_alignment(t, 800, SimulationModel(), seed + 1);
}

void expect_same_for_any_device_count(const Alignment &aln, bool sample_start) {
    const RunResult one = run(aln, 1, sample_start);
    printf("  1 device:  %s\n", fingerprint(one).c_str());
    for (int n : {2, 3}) {
        const RunResult more = run(aln, n, sample_start);
        printf("  %d devices: %s\n", n, fingerprint(more).c_str());
        EXPECT_EQ(more.tree.c1, one.tree.c1) << n << " devices";
        EXPECT_EQ(more.tree.c2, one.tree.c2) << n << " devices";
        EXPECT_EQ(more.tree.branch_lengths, one.tree.branch_lengths) << n << " devices";
        EXPECT_EQ(more.delta, one.delta) << n << " devices";
        EXPECT_EQ(more.bme_length, one.bme_length) << n << " devices";
        EXPECT_EQ(more.stats.rounds_run, one.stats.rounds_run) << n << " devices";
    }
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
            for (const bool flip : {false, true}) {
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

// A search ends no longer than the true tree (same branch-length pass, the default representative cap); near it in RF.
TEST(InferenceGpu, SearchUnderEachModel) {
    const int32_t N = 3000;
    const Tree tree = random_tree(N, 0.005, 0.08, 31);
    const Alignment aln = simulate_alignment(tree, 2000, truth(), 32);
    for (Model model : MODELS) {
        Settings opt;
        opt.tree_init = TreeInit::Random;
        opt.devices = {0};
        opt.substitution_model = distance_model(model);
        const Result res = run_infer(aln, opt);
        EXPECT_LT(res.delta_bme, 0.0) << model_name(model); /* the tree got shorter */
        // the true tree, scored by the same pass
        Settings score = opt;
        score.tree_init = TreeInit::Given;
        score.max_rounds = 0;
        const Result true_score = run_infer(aln, score, &tree);
        EXPECT_LE(res.bme_length, true_score.bme_length * (1.0 + 1e-4))
            << model_name(model) << ": found " << res.bme_length << ", true tree " << true_score.bme_length;
        const int32_t rf = rf_distance(res.tree, tree, N);
        EXPECT_LT(rf, N / 5) << model_name(model) << " RF " << rf;
        /* the true model (GTR+GC with the true parameters) recovers the tree far better than the others (RF
         * about 160): this fails if the window solve does not use the model */
        if (model == Model::GTR_GC) {
            EXPECT_LT(rf, N / 30) << "GTR+GC RF " << rf;
        }
        printf("%-7s: BME %.6f (true tree %.6f), RF %d of %d, rounds %d\n", model_name(model), res.bme_length,
               true_score.bme_length, rf, 2 * (N - 3), res.rounds_run);
    }
}

// The CPU device against the GPU: the same search on the same data gives the same BME length up
// to single-precision rounding, under every model.
TEST(InferenceGpu, CpuAgreesWithGpu) {
    const int32_t N = 3000;
    const Tree tree = random_tree(N, 0.005, 0.08, 101);
    const Alignment aln = simulate_alignment(tree, 1000, truth(), 102);
    for (Model model : MODELS) {
        Settings opt;
        opt.devices = {0};
        opt.substitution_model = distance_model(model);
        opt = with_base_frequencies(aln, opt);
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

// infer() checks its input before it uses the devices.
TEST(Inference, InferRejectsBadInput) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    Settings opt;
    opt.representative_cap = 0;
    EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("representative cap"), std::string::npos);
    opt.representative_cap = DEFAULT_REPRESENTATIVE_CAP;
    opt.tree_init = TreeInit::Given;
    EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("start tree"), std::string::npos);
    const Alignment three = simulate_alignment(random_tree(3, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    EXPECT_NE(thrown_message([&] { infer(three, Settings(), devices); }).find("at least 4 sequences"),
              std::string::npos);
}

// Below START_SAMPLE_MIN_N sequences, the sample-and-place start falls back to the random start, with a warning.
TEST(Inference, SampleStartFallsBackToRandom) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    const Result res = infer(aln, Settings(), devices);
    EXPECT_EQ(res.tree_init, TreeInit::Random);
    ASSERT_EQ(res.warnings.size(), 1u);
    EXPECT_NE(res.warnings[0].find("using the random start"), std::string::npos);
}

// GTR+GC needs alpha > 0 (a normal float), and exchangeabilities > 0 with a finite sum.
// Values outside the range checked in single precision run, with a warning (none in double precision).
TEST(Inference, GtrGcParameters) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    Settings opt;
    opt.tree_init = TreeInit::Random;
    opt.substitution_model.model = Model::GTR_GC;
    auto refused = [&](const char *what) {
        return thrown_message([&] { infer(aln, opt, devices); }).find(what) != std::string::npos;
    };
    auto warnings = [&] { return infer(aln, opt, devices).warnings; };
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();

    for (double a : {0.0, -1.0, 1e-40, 1e39, inf, nan}) {
        opt.substitution_model.alpha = a;
        EXPECT_TRUE(refused("alpha must be > 0")) << a;
    }
    opt.substitution_model.alpha = 1.0;
    double *s = opt.substitution_model.rates;
    for (double r : {0.0, -1.0, inf, nan}) {
        s[4] = r;
        EXPECT_TRUE(refused("exchangeabilities must be finite and > 0")) << r;
    }
    std::fill(s, s + N_GTR_RATES, 1e308);
    EXPECT_TRUE(refused("sum of the exchangeabilities"));
    std::fill(s, s + N_GTR_RATES, 1.0);
    EXPECT_TRUE(warnings().empty());

    // outside the checked range: a warning per parameter in single precision, none in double
    opt.substitution_model.alpha = 2.0 * GTR_FLOAT_ALPHA_MAX;
    s[1] = 2.0 * GTR_FLOAT_RATE_RATIO_MAX;
    const std::vector<std::string> w = warnings();
    ASSERT_EQ(w.size(), 2u);
    EXPECT_NE(w[0].find("alpha = 2000 is outside [0.05, 1000]"), std::string::npos) << w[0];
    EXPECT_NE(w[1].find("200 times the smallest, above 100"), std::string::npos) << w[1];
    opt.precision = Precision::Double;
    EXPECT_TRUE(warnings().empty());
    // only the ratios matter: tiny or large exchangeabilities of ratio 1 are fine
    opt.precision = Precision::Single;
    opt.substitution_model.alpha = 1.0;
    for (double scale : {1e-30, 1e30}) {
        std::fill(s, s + N_GTR_RATES, scale);
        EXPECT_TRUE(warnings().empty()) << scale;
    }
}

// Base frequencies must be finite and >= 0: NaN, inf and negative values are refused.
TEST(Inference, BaseFrequenciesMustBeFinite) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    Settings opt;
    opt.tree_init = TreeInit::Random;
    opt.substitution_model.model = Model::F81;
    const double freqs[N_BASES] = {0.1, 0.2, 0.3, 0.4};
    std::copy(freqs, freqs + N_BASES, opt.substitution_model.base_freqs);
    EXPECT_NO_THROW(infer(aln, opt, devices));
    for (double bad : {-0.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        opt.substitution_model.base_freqs[2] = bad;
        EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("base frequencies"), std::string::npos) << bad;
    }
}

// N = 6,000: several leaf-covering waves with more than one window, and V-cycle levels 4,096, N and 3,200.
TEST(InferenceGpu, SameTreeForAnyDeviceCountRandomStart) {
    expect_same_for_any_device_count(simulated(6000, 11), false);
}

// Sample-and-place: the nearest-sample step is split across the devices too.
TEST(InferenceGpu, SameTreeForAnyDeviceCountSampleStart) {
    expect_same_for_any_device_count(simulated(6000, 21), true);
}
