// Helpers shared by the error-path tests (test_search, test_start_tree, test_branch_lengths, test_inference,
// test_device_gpu): a fake device that throws from one chosen step, so the device-step errors run without a GPU, and
// the message of an exception.
#ifndef VECME_TESTS_FAKE_DEVICE_TEST_HELPERS_HPP
#define VECME_TESTS_FAKE_DEVICE_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "search.hpp"

namespace {

using namespace vecme;

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
    void read_distances(std::vector<double> &D) override { D.clear(); }
};

class FailingDevice : public Device {
  public:
    explicit FailingDevice(const char *step) : step_(step) {}
    void load(const uint64_t *, int32_t, int32_t) override {}
    void set_distance(const SubstitutionModel &) override {}
    void push_subset(const std::vector<int32_t> &) override {}
    void pop_subset() override {}
    int32_t max_window() const override { return WORKSPACE_MAX_K; }
    void synchronize() override {}
    std::string name() const override { return "failing device"; }
    double memory_peak_mb() const override { return 0.0; }
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

inline Devices failing_devices(int n, const char *step) {
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

} // namespace

#endif // VECME_TESTS_FAKE_DEVICE_TEST_HELPERS_HPP
