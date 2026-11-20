// Error paths: the library throws, never exits, also from inside the OpenMP regions
// that hand work to the devices. A failing fake device forces the device-step errors without a GPU; the
// GpuDevice tests (out of memory, an invalid GPU) need one and are named Gpu*.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "branch_lengths.hpp"
#include "config.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "search.hpp"
#include "simulations.hpp"
#include "start_tree.hpp"

namespace {

using namespace p2vbme;

// A device that does nothing, except throw from one chosen step.
// Window kernels that find no move, so a window solve returns 0 (a failing device that is not meant to fail here).
class NoMoveKernels : public WindowKernels {
  public:
    void distances(int32_t, const int32_t *, const int32_t *, const float *) override {}
    double score(const SkeletonTables &, const std::vector<int32_t> &) override { return 0.0; }
    void best_moves(const SkeletonTables &, std::vector<double> &best_delta, std::vector<int32_t> &best_j) override {
        std::fill(best_delta.begin(), best_delta.end(), 1e30);
        std::fill(best_j.begin(), best_j.end(), -1);
    }
};

class FailingDevice : public Device {
  public:
    explicit FailingDevice(const char *step) : step_(step) {}
    void load(const uint64_t *, int32_t, int32_t) override {}
    void set_distance(const SubstitutionModel &, const EigenSettings &) override {}
    void push_subset(const std::vector<int32_t> &) override {}
    void pop_subset() override {}
    int32_t max_window() const override { return WORKSPACE_MAX_K; }
    void synchronize() override {}
    std::string name() const override { return "failing device"; }
    double memory_in_use_mb() const override { return 0.0; }
    void nearest_sampled(const int32_t *, int32_t n_queries, const int32_t *, int32_t, int32_t *best_pos) override {
        fail("nearest_sampled");
        for (int32_t i = 0; i < n_queries; ++i)
            best_pos[i] = 0;
    }
    std::unique_ptr<WindowKernels> window_kernels() override {
        fail("window_kernels");
        return std::make_unique<NoMoveKernels>();
    }
    void edge_lengths(int32_t, int32_t, int32_t count, const int32_t *, const int32_t *, int32_t, const int32_t *,
                      const int32_t *, const float *, const int32_t *, const int32_t *, const float *, double *dAB,
                      double *dAO, double *dBO) override {
        fail("edge_lengths");
        for (int32_t i = 0; i < count; ++i)
            dAB[i] = dAO[i] = dBO[i] = 0.0;
    }

  private:
    void fail(const char *step) const {
        if (strcmp(step, step_) == 0)
            throw std::runtime_error(std::string("injected failure in ") + step);
    }
    const char *step_;
};

Devices failing_devices(int n, const char *step) {
    Devices d;
    for (int i = 0; i < n; ++i)
        d.push_back(std::unique_ptr<Device>(new FailingDevice(step)));
    return d;
}

// The message of the exception f throws (fails the test if it throws none).
template <class F> std::string thrown_message(F f) {
    try {
        f();
    } catch (const std::exception &e) {
        return e.what();
    }
    ADD_FAILURE() << "no exception";
    return "";
}

// A window solve that throws on one of several host threads: the search rethrows it after the parallel wave.
TEST(Errors, WindowSolveFailureLeavesTheSearch) {
    for (int n_devices : {1, 3}) {
        Devices devices = failing_devices(n_devices, "window_kernels");
        Tree tree = build_random_tree(3000, 1);
        const std::string msg = thrown_message([&] { run_minibatch_bme(tree, devices, SearchSettings()); });
        EXPECT_EQ(msg, "injected failure in window_kernels") << n_devices;
    }
}

TEST(Errors, NearestSampleFailureLeavesTheStartTree) {
    Devices devices = failing_devices(2, "nearest_sampled");
    int32_t n0 = 20;
    const std::string msg = thrown_message([&] { build_sample_and_place_tree(200, n0, 1, devices, SearchSettings()); });
    EXPECT_EQ(msg, "injected failure in nearest_sampled");
}

TEST(Errors, EdgeLengthFailureLeavesTheBranchLengths) {
    Devices devices = failing_devices(3, "edge_lengths");
    const Tree tree = build_random_tree(500, 1);
    std::vector<double> bl;
    const std::string msg =
        thrown_message([&] { compute_bme_branch_lengths(tree, devices, bl, DEFAULT_BL_PROFILE_CAP); });
    EXPECT_EQ(msg, "injected failure in edge_lengths");
}

// infer() checks its input before it uses the devices.
TEST(Errors, InferRejectsBadInput) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    Settings opt;
    opt.profile_cap = 0;
    EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("profile cap"), std::string::npos);
    opt.profile_cap = DEFAULT_PROFILE_CAP;
    opt.tree_init = TreeInit::Given;
    EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("start tree"), std::string::npos);
    const Alignment three = simulate_alignment(random_tree(3, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    EXPECT_NE(thrown_message([&] { infer(three, Settings(), devices); }).find("at least 4 sequences"),
              std::string::npos);
}

// Below START_SAMPLE_MIN_N sequences, the sample-and-place start falls back to the random start, with a warning.
TEST(Errors, SampleStartFallsBackToRandom) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    const Result res = infer(aln, Settings(), devices);
    EXPECT_EQ(res.tree_init, TreeInit::Random);
    ASSERT_EQ(res.warnings.size(), 1u);
    EXPECT_NE(res.warnings[0].find("using the random start"), std::string::npos);
}

// GTR+GC needs alpha > 0 (a normal float), exchangeabilities > 0 with a finite sum, and reweighting passes >= 0.
// Values outside the range checked in single precision run, with a warning (none in double precision).
TEST(Errors, GtrGcParameters) {
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
        opt.substitution_model.gtr_alpha = a;
        EXPECT_TRUE(refused("alpha must be > 0")) << a;
    }
    opt.substitution_model.gtr_alpha = 1.0;
    double *s = opt.substitution_model.gtr_rates;
    for (double r : {0.0, -1.0, inf, nan}) {
        s[4] = r;
        EXPECT_TRUE(refused("exchangeabilities must be finite and > 0")) << r;
    }
    std::fill(s, s + 6, 1e308);
    EXPECT_TRUE(refused("sum of the exchangeabilities"));
    std::fill(s, s + 6, 1.0);
    opt.eigen.reweight = -1;
    EXPECT_TRUE(refused("reweighting passes must be >= 0"));
    opt.eigen.reweight = 1000;
    EXPECT_TRUE(warnings().empty());

    // outside the checked range: a warning per parameter in single precision, none in double
    opt.substitution_model.gtr_alpha = 2.0 * GTR_FLOAT_ALPHA_MAX;
    s[1] = 2.0 * GTR_FLOAT_RATE_RATIO_MAX;
    const std::vector<std::string> w = warnings();
    ASSERT_EQ(w.size(), 2u);
    EXPECT_NE(w[0].find("alpha = 2000 is outside [0.05, 1000]"), std::string::npos) << w[0];
    EXPECT_NE(w[1].find("200 times the smallest, above 100"), std::string::npos) << w[1];
    opt.precision = Precision::Double;
    EXPECT_TRUE(warnings().empty());
    // only the ratios matter: tiny or large exchangeabilities of ratio 1 are fine
    opt.precision = Precision::Single;
    opt.substitution_model.gtr_alpha = 1.0;
    for (double scale : {1e-30, 1e30}) {
        std::fill(s, s + 6, scale);
        EXPECT_TRUE(warnings().empty()) << scale;
    }
}

// Base frequencies must be finite and >= 0: NaN, inf and negative values are refused.
TEST(Errors, BaseFrequenciesMustBeFinite) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Devices devices = failing_devices(1, "none");
    Settings opt;
    opt.tree_init = TreeInit::Random;
    opt.substitution_model.model = Model::F81;
    const double freqs[4] = {0.1, 0.2, 0.3, 0.4};
    std::copy(freqs, freqs + 4, opt.substitution_model.base_freqs);
    EXPECT_NO_THROW(infer(aln, opt, devices));
    for (double bad : {-0.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        opt.substitution_model.base_freqs[2] = bad;
        EXPECT_NE(thrown_message([&] { infer(aln, opt, devices); }).find("base frequencies"), std::string::npos) << bad;
    }
}

size_t gpu_free_bytes() {
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(0);
    cudaMemGetInfo(&free_b, &total_b);
    return free_b;
}

// A workspace far beyond any GPU: a clean out-of-memory error naming the GPU, the size and the free memory.
// The ~1.2 GB allocated before the failing allocation is freed again (GPU-wide free memory, so a 512 MB margin
// for other processes).
TEST(Errors, GpuOutOfMemoryIsReported) {
    make_gpu_device(0, 64); /* creates the CUDA context first */
    const size_t before = gpu_free_bytes();
    const std::string msg = thrown_message([] { make_gpu_device(0, 300000); });
    const size_t after = gpu_free_bytes();
    EXPECT_NE(msg.find("out of GPU memory on GPU 0"), std::string::npos) << msg;
    EXPECT_NE(msg.find("MiB requested"), std::string::npos) << msg;
    EXPECT_NE(msg.find("MiB free of"), std::string::npos) << msg;
    EXPECT_LT((double)before - (double)after, 512.0 * 1048576.0) << "memory left behind by the failed device";
}

TEST(Errors, GpuCudaErrorsThrow) {
    const std::string msg = thrown_message([] { make_gpu_device(97); });
    EXPECT_NE(msg.find("GPU 97 does not exist"), std::string::npos) << msg;
}

} // namespace
