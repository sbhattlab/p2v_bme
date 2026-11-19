/*
 * device.hpp
 *
 * The four steps of the method that run on a device (a GPU or a CPU worker), behind one interface.
 * A Device owns its memory: the alignment (a GPU copy; the CPU reads the caller's), the start tree's subsamples
 * and a workspace. The host code (search, start_tree, branch_lengths) hands work to one Device per worker, one host
 * thread each. Only this interface has more than one implementation (device_gpu.cu, device_cpu.cpp).
 */
#ifndef P2VBME_DEVICE_HPP
#define P2VBME_DEVICE_HPP

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "accelerator.hpp"
#include "config.hpp"
#include "eigen_dist.hpp"

namespace p2vbme {

// Supplies complete (uncapped) leaf profiles to the exact acceptance check of a window, building them on first
// use (CSR layout as the window's profiles).
using ExactProfileBuilder = std::function<void(const int32_t **offsets, const int32_t **leaves, const float **weights)>;

class Device {
  public:
    virtual ~Device() = default;

    // 1. Setup: load the alignment, replacing any previous one.
    // `packed` is laid out as in Alignment (n_taxa rows of 5 x n_words bitplanes).
    // The GPU device copies it.
    // The CPU device reads it in place, so `packed` must stay valid until the next load() or the device's destruction.
    virtual void load(const uint64_t *packed, int32_t n_taxa, int32_t n_words) = 0;
    // The distance model of the window solve and the edge-length pass (JC69 until set).
    virtual void set_distance(const SubstitutionModel &model, const EigenSettings &eigen) = 0;
    // The current alignment becomes rows[] of the current one (a start-tree subsample), until pop_subset.
    virtual void push_subset(const std::vector<int32_t> &rows) = 0;
    virtual void pop_subset() = 0;
    // Largest window (components K) the workspace holds.
    virtual int32_t max_window() const = 0;
    // Waits for all work on the device.
    virtual void synchronize() = 0;
    // e.g. "GPU 0: NVIDIA RTX A6000"
    virtual std::string name() const = 0;
    // Memory in use on the device, MiB (for a GPU: GPU-wide, other processes included).
    virtual double memory_in_use_mb() const = 0;

    // 2. Nearest sampled sequence (start tree).
    // For each query row, the position in samples[] of its nearest sample in p-distance over shared valid sites
    // (no shared site: p = NN_P_NO_OVERLAP).
    // Ties to the lower position.
    virtual void nearest_sampled(const int32_t *queries, int32_t n_queries, const int32_t *samples, int32_t n_samples,
                                 int32_t *best_pos) = 0;

    // 3. Window solve.
    // A window is part of the tree cut into K pieces.
    // Each piece is summarised as a weighted leaf profile (CSR: profile_offsets[K + 1], profile_leaves,
    // profile_weights).
    // Its skeleton, the tree connecting the pieces, is the Phylo2Vec vector v_cf over the leaf permutation perm.
    // Runs up to max_sweeps sweeps of batched moves on the skeleton (batch cap max_batch_cap) to shorten its BME
    // length.
    // Updates v_cf and perm and returns the BME change (0 and the input tree if nothing improved).
    // The caller writes the new skeleton back into the full tree.
    // With exact_builder, small gains (> -exact_check_eta) are re-checked on complete profiles.
    virtual double solve_window(int32_t K, int32_t max_sweeps, int32_t max_batch_cap, double exact_check_eta,
                                const int32_t *profile_offsets, const int32_t *profile_leaves,
                                const float *profile_weights, std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                                const ExactProfileBuilder *exact_builder = nullptr) = 0;

    // 4. Edge-length pass (branch lengths).
    // For each of `count` internal nodes with children c1[i] and c2[i], computes the profile-weighted distances
    // between the two child subtrees (dAB[i]) and between each child and the rest of the tree (dAO[i], dBO[i]).
    // The inside profiles cover all n_int internal nodes (node n_leaves + j at position j, cap entries each).
    // The outside profiles cover the `count` nodes.
    // The sums are in the device's precision (GPU: float).
    virtual void edge_lengths(int32_t n_leaves, int32_t cap, int32_t count, const int32_t *c1, const int32_t *c2,
                              int32_t n_int, const int32_t *in_len, const int32_t *in_leaves, const float *in_weights,
                              const int32_t *out_len, const int32_t *out_leaves, const float *out_weights, double *dAB,
                              double *dAO, double *dBO) = 0;
};

using Devices = std::vector<std::unique_ptr<Device>>;

// Number of visible GPUs (0 without CUDA or without a GPU).
int visible_gpus();

// One device per GPU number in gpu_ids (empty: all visible GPUs), each with a workspace for windows of up to
// WORKSPACE_MAX_K pieces.
// Throws std::runtime_error if a GPU does not exist or there is none.
Devices make_devices(const std::vector<int> &gpu_ids);

// The CUDA runtime and driver versions and the GPU architectures compiled for.
std::string gpu_build_info();

// One GPU with a workspace for windows of up to max_k pieces.
std::unique_ptr<Device> make_gpu_device(int gpu, int32_t max_k = WORKSPACE_MAX_K);

// The CPU device: n_workers devices (--accelerator cpu --devices N), each using up to n_workers
// threads when it works alone; single precision (the GPU's) or double.
Devices make_cpu_devices(int32_t n_workers, bool double_precision);
std::unique_ptr<Device> make_cpu_device(int32_t worker, int32_t threads, bool double_precision);

// ---- The run's devices, chosen once (the command line, then infer() runs on them) ----

// The processor of a run: the GPU or the CPU as asked; auto: the GPU if one is visible and the precision is
// single, else the CPU, with the reason in `note` ("<why>: running on the CPU").
// The only place auto is resolved.
struct DeviceChoice {
    bool on_gpu = false;
    std::string note; // auto on the CPU: why (else empty)
};
DeviceChoice choose_devices(Accelerator accelerator, Precision precision);

// The devices of the choice: one per GPU number in gpu_ids (empty: all visible), or `threads` CPU workers (0: all,
// omp_get_max_threads) in the precision.
// First checks the device settings, the only place they are checked.
// Throws std::runtime_error if a GPU is listed twice, the number of threads is out of range, or the choice cannot run
// (the GPU in double precision, or no GPU visible).
Devices select_devices(const DeviceChoice &choice, Precision precision, const std::vector<int> &gpu_ids,
                       int32_t threads);

} // namespace p2vbme

#endif // P2VBME_DEVICE_HPP
