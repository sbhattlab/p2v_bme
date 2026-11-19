// The tree must not depend on the number of devices: the search, the sample-and-place start
// and the branch lengths on 1, 2 and 3 devices give the same result. Several devices share one GPU here (each
// with its own workspace and host thread), so the multi-device code paths run on a single card. Label gpu.
#include <gtest/gtest.h>

#include <omp.h>
#include <stdio.h>

#include <functional>
#include <string>
#include <vector>

#include "branch_lengths.hpp"
#include "config.hpp"
#include "device.hpp"
#include "search.hpp"
#include "simulations.hpp"
#include "start_tree.hpp"

namespace {

using namespace p2vbme;

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
    r.bme_length = compute_bme_branch_lengths(r.tree, devices, r.tree.branch_lengths, DEFAULT_BL_PROFILE_CAP);
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

// N = 6,000: several leaf-covering waves with more than one window, and V-cycle levels 4,096, N and 3,200.
TEST(SearchDevices, SameTreeForAnyDeviceCountRandomStart) {
    expect_same_for_any_device_count(simulated(6000, 11), false);
}

// Sample-and-place: the nearest-sample step is split across the devices too.
TEST(SearchDevices, SameTreeForAnyDeviceCountSampleStart) {
    expect_same_for_any_device_count(simulated(6000, 21), true);
}

// The edge-length pass hands one chunk to each device. Called from inside a parallel region, OpenMP gives it a
// single thread: every chunk must still be computed (it was split by thread number before).
TEST(SearchDevices, BranchLengthsInsideAParallelRegion) {
    const Alignment aln = simulated(2000, 31);
    Devices devices;
    for (int d = 0; d < 3; ++d)
        devices.push_back(make_gpu_device(0));
    for (auto &dev : devices)
        dev->load(aln.packed.data(), aln.n_taxa, aln.n_words);
    const Tree tree = build_random_tree(aln.n_taxa, 5);
    std::vector<double> outside, inside;
    const double total = compute_bme_branch_lengths(tree, devices, outside, 32);
    double total_nested = 0.0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        total_nested = compute_bme_branch_lengths(tree, devices, inside, 32);
    }
    EXPECT_EQ(inside, outside);
    EXPECT_EQ(total_nested, total);
}

} // namespace
