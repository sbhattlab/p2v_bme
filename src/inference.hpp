/*
 * inference.hpp
 *
 * Public interface of the Phylo2Vec BME tree search, with core.hpp (the alignment and tree types), the
 * input/output headers (fasta.hpp, maple.hpp, newick.hpp, io.hpp), processing.hpp (before and after inference)
 * and device.hpp (the devices): Settings, Result and infer() (balanced minimum evolution, distances under a
 * substitution model, GPU minibatch search).
 */
#ifndef P2VBME_INFERENCE_HPP
#define P2VBME_INFERENCE_HPP

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "accelerator.hpp"
#include "config.hpp"
#include "core.hpp"
#include "device.hpp"
#include "eigen_dist.hpp"

namespace p2vbme {

enum class TreeInit { Sample, Random, Given };

// The settings of a run; defaults in config.hpp.
struct Settings {
    TreeInit tree_init = TreeInit::Sample;
    int32_t sample_size = DEFAULT_SAMPLE_SIZE;        // sample-and-place start: 0 = min(100000, N / 10)
    uint64_t seed = DEFAULT_SEED;                     // seeds the sample-and-place or random start
    int32_t max_rounds = DEFAULT_MAX_ROUNDS;          // rounds of the search; 0 = no search (score the start)
    double early_stopping = DEFAULT_EARLY_STOPPING;   // stop once a round gains <= this x the total gain;
                                                      // 0 = only when a round changes nothing (always on)
    double time_limit = DEFAULT_TIME_LIMIT;           // seconds; no new round after this (0 = none)
    int32_t profile_cap = DEFAULT_PROFILE_CAP;        // leaves kept per clade profile (1..256)
    double exact_check_eta = DEFAULT_EXACT_CHECK_ETA; // exact re-check of windows whose capped gain is > -eta
    bool branch_lengths = DEFAULT_BRANCH_LENGTHS;     // estimate BME branch lengths
    int32_t bl_profile_cap = DEFAULT_BL_PROFILE_CAP;  // leaves kept per subtree profile for branch lengths
    SubstitutionModel substitution_model;             // the model of the distances (JC69 by default);
                                                      // GTR+GC with given parameters (the fit comes later)
    EigenSettings eigen;                              // how GTR+GC distances are estimated (not on the command line)
    Accelerator accelerator = Accelerator::Auto;      // auto: the GPU if one is visible, else the CPU
    std::vector<int> devices;                         // GPU numbers; empty = all visible
    int32_t threads = 0;                              // CPU: workers (threads); 0 = all (omp_get_max_threads)
    Precision precision = Precision::Single;          // CPU only: single (the GPU's) or double;
                                                      // double with accelerator auto selects the CPU
};

struct Result {
    Tree tree;
    double bme_length = 0.0;  // sum of the raw BME branch lengths
    double delta_bme = 0.0;   // BME change made by the search, from its start tree
    double time_start = 0.0;  // seconds: start tree
    double time_search = 0.0; // seconds: minibatch search
    double time_branch_lengths = 0.0;
    int32_t sample_size_used = 0;
    int32_t rounds_planned = 0;
    int32_t rounds_run = 0;
    std::string stop_reason;               // max_rounds, idle, early_stopping, time_limit, or none (no search)
    std::vector<double> round_gains;       // BME decrease in each round of the main search
                                           // (positive = shorter tree; the last is the cleanup pass when N > 1,050)
    TreeInit tree_init = TreeInit::Sample; // the start used (Random when Sample falls back)
    std::vector<std::string> warnings;     // for the caller to show (the library prints nothing)
};

// Receives the run's progress lines (the library itself prints nothing).
using LogCallback = std::function<void(const std::string &line)>;

// Infers a BME tree on the devices (select_devices; the alignment is loaded onto them).
// First checks the settings and the input, the only place they are checked.
// With fewer than START_SAMPLE_MIN_N sequences, the sample-and-place start falls back to the random start (a
// warning in Result).
// `start_tree` is used when settings.tree_init == TreeInit::Given.
// Errors (bad settings or input, CUDA, GPU memory) are thrown as std::exception.
Result infer(const Alignment &aln, const Settings &settings, Devices &devices, const Tree *start_tree = nullptr,
             const LogCallback &log = nullptr);

} // namespace p2vbme

#endif // P2VBME_INFERENCE_HPP
