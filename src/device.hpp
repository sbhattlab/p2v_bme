/*
 * device.hpp
 *
 * The parts of the method that run on a device (a GPU or a CPU worker), behind one interface: setup, the
 * nearest-sample search, the window solve and the edge-length pass.
 * device_gpu.cu and device_cpu.cpp implement it.
 * The host code (search, start_tree, branch_lengths) hands work to one Device per worker, one host thread each.
 * A Device holds what its steps read: the alignment (the GPU copies it, the CPU reads the caller's in place), the
 * start tree's subsamples and a workspace.
 */
#ifndef P2VBME_DEVICE_HPP
#define P2VBME_DEVICE_HPP

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "accelerator.hpp"
#include "config.hpp"
#include "distances.hpp"

namespace p2vbme {

class WindowKernels; // the window solve's arithmetic (search.hpp)

// One GPU, or one CPU worker.
class Device {
  public:
    virtual ~Device() = default;

    // 1. Setup

    // Loads the alignment, replacing any previous one.
    // `packed` is laid out as in Alignment (n_taxa rows of 5 x n_words bitplanes).
    // The GPU device copies it and returns once the copy is complete.
    // The CPU device reads it in place, so `packed` must stay valid until the next load() or the device's destruction.
    virtual void load(const uint64_t *packed, int32_t n_taxa, int32_t n_words) = 0;
    // Sets the distance model of the window solve and the edge-length pass (JC69 until set).
    // The nearest-sample search always uses the p-distance.
    virtual void set_distance(const SubstitutionModel &model) = 0;
    // Narrows the current alignment to its rows[] (a start-tree subsample), until pop_subset.
    // Subsets nest: each push narrows the current alignment again.
    virtual void push_subset(const std::vector<int32_t> &rows) = 0;
    // Restores the alignment from before the last push_subset.
    virtual void pop_subset() = 0;
    // Returns the largest window the workspace holds, in pieces (K).
    virtual int32_t max_window() const = 0;
    // Waits for all work on the device.
    virtual void synchronize() = 0;
    // Returns the device's name for the report, e.g. "GPU 0: NVIDIA RTX A6000" or "CPU worker 0 (single precision)".
    virtual std::string name() const = 0;
    // Returns the most memory p2v-bme held on the device's GPU at once, in MiB.
    // GPU: what this process allocated there, without the CUDA context or other processes; CPU: 0.
    virtual double memory_peak_mb() const = 0;

    // 2. Nearest-sample search (start tree)

    // Finds each query's nearest sample: (queries, samples) -> best_pos.
    // best_pos[i] is the position in samples[] of the sample nearest to queries[i] in p-distance over shared valid
    // sites (no shared site: p = NN_P_NO_OVERLAP).
    // Ties go to the lower position.
    virtual void nearest_sampled(const int32_t *queries, int32_t n_queries, const int32_t *samples, int32_t n_samples,
                                 int32_t *best_pos) = 0;

    // 3. Window solve

    // Creates the kernels of one window solve, on the current alignment (a subset while the start tree is built).
    // solve_window (search.hpp) runs the sweeps and calls them; a new object for each window.
    virtual std::unique_ptr<WindowKernels> window_kernels() = 0;

    // 4. Edge-length pass (branch lengths)

    // Computes the edge-length distances of `count` internal nodes: representatives -> (dAB, dAO, dBO).
    // Node i has children c1[i] (A) and c2[i] (B); O is the rest of the tree, outside node i.
    // dAB[i], dAO[i], dBO[i]: the distances weighted by the representatives between A and B, A and O, B and O.
    // Inside representatives: all n_int internal nodes, node n_leaves + j in row j (in_len[j] entries, rows cap apart).
    // A leaf child represents itself (weight 1).
    // Outside representatives: node i in row i (out_len[i] entries, rows cap apart).
    // The sums are in the device's precision (GPU: float).
    virtual void edge_lengths(int32_t n_leaves, int32_t cap, int32_t count, const int32_t *c1, const int32_t *c2,
                              int32_t n_int, const int32_t *in_len, const int32_t *in_leaves, const float *in_weights,
                              const int32_t *out_len, const int32_t *out_leaves, const float *out_weights, double *dAB,
                              double *dAO, double *dBO) = 0;
};

using Devices = std::vector<std::unique_ptr<Device>>;

// Number of visible GPUs (0 without CUDA or without a GPU).
int visible_gpus();

// Says why no GPU is visible, for messages: "" if one is; else, e.g., "no GPU found", "no NVIDIA driver found", a
// driver too old for this build's CUDA, or "this p2v-bme was built without CUDA".
std::string no_gpu_reason();

// One device per GPU number in gpu_ids (empty: all visible GPUs), each with a workspace for windows of up to
// WORKSPACE_MAX_K pieces.
// Throws std::runtime_error if a GPU does not exist or there is none.
Devices make_devices(const std::vector<int> &gpu_ids);

// One GPU with a workspace for windows of up to max_k pieces.
std::unique_ptr<Device> make_gpu_device(int gpu, int32_t max_k = WORKSPACE_MAX_K);

// The CPU device: n_workers devices (--accelerator cpu --devices N), each using up to n_workers
// threads when it works alone; single precision (the GPU's) or double.
Devices make_cpu_devices(int32_t n_workers, bool double_precision);
std::unique_ptr<Device> make_cpu_device(int32_t worker, int32_t threads, bool double_precision);

// ---- The run's devices, chosen once (the command line, then infer() runs on them) ----

// Resolves the accelerator: (accelerator, precision) -> Gpu or Cpu, never Auto.
// gpu, cpu: as asked.
// auto: the GPU if one is visible and the precision is single, else the CPU.
// The only place auto is resolved.
Accelerator resolve_accelerator(Accelerator accelerator, Precision precision);

// Creates the run's devices after checking their settings: accelerator -> one Device per GPU or per CPU worker.
// The accelerator is Gpu or Cpu (resolve_accelerator).
// GPU: one device per GPU number in gpu_ids (empty: all visible).
// CPU: `threads` workers (0: all, omp_get_max_threads) in the precision.
// The only place the device settings are checked.
// Throws std::runtime_error if a GPU is listed twice, the number of threads is out of range, or the accelerator
// cannot run (the GPU in double precision, or no GPU visible).
// Throws std::logic_error if the accelerator is Auto.
Devices select_devices(Accelerator accelerator, Precision precision, const std::vector<int> &gpu_ids, int32_t threads);

} // namespace p2vbme

#endif // P2VBME_DEVICE_HPP
