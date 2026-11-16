/*
 * device.hpp
 *
 * The four steps of the method that run on a device (a GPU or a CPU worker), behind one interface.
 * A Device owns its memory: the alignment (a GPU copy; the CPU reads the caller's), the start tree's subsamples
 * and a workspace. The host code (search, start_tree, branch_lengths) hands work to one Device per worker, one host
 * thread each. Only this interface has more than one implementation (device_gpu.cu, device_cpu.cpp).
 */
#pragma once

#include <stdint.h>

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "config.hpp"
#include "eigen_dist.hpp"

namespace p2vbme {

/* Supplies complete (uncapped) leaf profiles to the exact acceptance check of a window, building them on first
 * use (CSR layout as the window's profiles). */
using ExactProfileBuilder = std::function<void(const int32_t **offsets, const int32_t **leaves, const float **weights)>;

class Device {
  public:
    virtual ~Device() = default;

    /* 1. Setup. Makes the alignment (n_taxa rows of 5 x n_words bitplanes, the Alignment layout) the device's,
     * replacing the one held: the GPU copies it; the CPU device reads it in place, so the caller keeps it alive. */
    virtual void load(const uint64_t *packed, int32_t n_taxa, int32_t n_words) = 0;
    /* The distance model of the window solve and the edge-length pass (JC69 until set). */
    virtual void set_distance(const DistanceModel &model) = 0;
    /* The current alignment becomes rows[] of the current one (a start-tree subsample), until pop_subset. */
    virtual void push_subset(const std::vector<int32_t> &rows) = 0;
    virtual void pop_subset() = 0;
    /* Largest window (components K) the workspace holds. */
    virtual int32_t max_window() const = 0;
    /* Waits for all work on the device. */
    virtual void synchronize() = 0;
    /* e.g. "GPU 0: NVIDIA RTX A6000" */
    virtual std::string name() const = 0;
    /* Memory in use on the device, MiB (for a GPU: GPU-wide, other processes included). */
    virtual double memory_in_use_mb() const = 0;

    /* 2. Nearest sampled sequence (start tree): for each query row, the position in samples[] of its nearest
     * sample in p-distance over shared valid sites (no shared site: p = NN_P_NO_OVERLAP); ties to the lower
     * position. */
    virtual void nearest_sampled(const int32_t *queries, int32_t n_queries, const int32_t *samples, int32_t n_samples,
                                 int32_t *best_pos) = 0;

    /* 3. Window solve. The window is a skeleton tree on K components (Phylo2Vec vector v_cf over the leaf
     * permutation perm), each component a weighted leaf profile (CSR: prof_offsets[K + 1], prof_leaves,
     * prof_weights). Runs up to max_sweeps sweeps of batched moves (batch cap max_batch_cap); updates v_cf and
     * perm and returns the BME change (0 and the input tree if nothing improved). With exact_builder, small
     * gains (> -exact_check_eta) are re-checked on complete profiles. */
    virtual double solve_window(int32_t K, int32_t max_sweeps, int32_t max_batch_cap, double exact_check_eta,
                                const int32_t *prof_offsets, const int32_t *prof_leaves, const float *prof_weights,
                                std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                                const ExactProfileBuilder *exact_builder = nullptr) = 0;

    /* 4. Edge-length pass (branch lengths). For `count` internal nodes with children c1[i], c2[i]: the
     * profile-weighted distances between the two child subtrees (dAB) and between each child and the rest of the
     * tree (dAO, dBO). Inside profiles of all n_int internal nodes (node n_leaves + j at j, cap entries each);
     * outside profiles of the `count` nodes. The sums are in the device's precision (the GPU: float). */
    virtual void edge_lengths(int32_t n_leaves, int32_t cap, int32_t count, const int32_t *c1, const int32_t *c2,
                              int32_t n_int, const int32_t *in_len, const int32_t *in_leaves, const float *in_weights,
                              const int32_t *out_len, const int32_t *out_leaves, const float *out_weights, double *dAB,
                              double *dAO, double *dBO) = 0;
};

using Devices = std::vector<std::unique_ptr<Device>>;

/* The first exception thrown by the device steps inside an OpenMP region (one host thread per device), kept
 * and rethrown after the region: an exception must not leave a parallel region. Once one is caught, the
 * remaining work items are skipped. */
class ParallelErrors {
  public:
    bool failed() const { return failed_.load(std::memory_order_relaxed); }
    void capture() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_)
            first_ = std::current_exception();
        failed_ = true;
    }
    void rethrow() const {
        if (first_)
            std::rethrow_exception(first_);
    }

  private:
    std::exception_ptr first_;
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
};

/* Number of visible GPUs (0 without CUDA or without a GPU). */
int visible_gpus();

/* One device per GPU number in gpu_ids (empty: all visible GPUs, at most MAX_GPUS), each with a workspace for
 * windows of up to WORKSPACE_MAX_K components. Throws std::runtime_error if a GPU does not exist or there is none. */
Devices make_devices(const std::vector<int> &gpu_ids);

/* The CUDA runtime and driver versions and the GPU architectures compiled for. */
std::string gpu_build_info();

/* One GPU with a workspace for windows of up to max_k components. */
std::unique_ptr<Device> make_gpu_device(int gpu, int32_t max_k = WORKSPACE_MAX_K);

/* The CPU device: n_workers devices (--accelerator cpu --devices N), each using up to n_workers
 * threads when it works alone; single precision (the GPU's) or double. */
Devices make_cpu_devices(int32_t n_workers, bool double_precision);
std::unique_ptr<Device> make_cpu_device(int32_t worker, int32_t threads, bool double_precision);

} // namespace p2vbme
