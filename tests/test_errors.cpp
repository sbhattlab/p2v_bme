// Error paths: the library throws, never exits, also from inside the OpenMP regions
// that hand work to the devices. A failing fake device forces the device-step errors without a GPU; the
// GpuDevice tests (out of memory, an invalid GPU) need one and are named Gpu*.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string.h>

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
    double solve_window(int32_t, int32_t, int32_t, double, const int32_t *, const int32_t *, const float *,
                        std::vector<int64_t> &, std::vector<int32_t> &, const ExactProfileBuilder *) override {
        fail("solve_window");
        return 0.0;
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
        Devices devices = failing_devices(n_devices, "solve_window");
        std::vector<int32_t> c1, c2;
        make_uniform_random_tree_host(3000, 1, c1, c2);
        const std::string msg =
            thrown_message([&] { run_multi_gpu_minibatch_bme(3000, devices, c1, c2, SearchSettings()); });
        EXPECT_EQ(msg, "injected failure in solve_window") << n_devices;
    }
}

TEST(Errors, NearestSampleFailureLeavesTheStartTree) {
    Devices devices = failing_devices(2, "nearest_sampled");
    std::vector<int32_t> c1, c2;
    SampleInitStats stats;
    const std::string msg =
        thrown_message([&] { build_sample_init_tree(200, 20, 1, devices, c1, c2, stats, SearchSettings()); });
    EXPECT_EQ(msg, "injected failure in nearest_sampled");
}

TEST(Errors, EdgeLengthFailureLeavesTheBranchLengths) {
    Devices devices = failing_devices(3, "edge_lengths");
    std::vector<int32_t> c1, c2;
    make_uniform_random_tree_host(500, 1, c1, c2);
    std::vector<double> bl;
    const std::string msg = thrown_message(
        [&] { compute_bme_branch_lengths(500, c1.data(), c2.data(), devices, bl, DEFAULT_BL_PROFILE_CAP); });
    EXPECT_EQ(msg, "injected failure in edge_lengths");
}

TEST(Errors, SampleStartNeedsSixteenSequences) {
    Devices devices = failing_devices(1, "none");
    std::vector<int32_t> c1, c2;
    SampleInitStats stats;
    EXPECT_THROW(build_sample_init_tree(10, 8, 1, devices, c1, c2, stats, SearchSettings()), std::invalid_argument);
}

// infer() checks its input before it looks for a GPU.
TEST(Errors, InferRejectsBadInput) {
    const Alignment aln = simulate_alignment(random_tree(10, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    Settings opt;
    EXPECT_NE(thrown_message([&] { infer(aln, opt); }).find("at least 16 sequences"), std::string::npos);
    opt.start = Start::Random;
    opt.profile_cap = 0;
    EXPECT_NE(thrown_message([&] { infer(aln, opt); }).find("profile cap"), std::string::npos);
    opt.profile_cap = DEFAULT_PROFILE_CAP;
    opt.start = Start::Given;
    EXPECT_NE(thrown_message([&] { infer(aln, opt); }).find("start tree"), std::string::npos);
    const Alignment three = simulate_alignment(random_tree(3, 0.1, 0.1, 1), 50, SimulationModel(), 2);
    EXPECT_NE(thrown_message([&] { infer(three, Settings()); }).find("at least 4 sequences"), std::string::npos);
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
