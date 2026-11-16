/*
 * inference.hpp
 *
 * Public interface of the Phylo2Vec BME tree search, with io.hpp: the alignment and tree types, Options,
 * Result and infer() (balanced minimum evolution, distances under a substitution model, GPU minibatch search).
 */
#pragma once

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "config.hpp"
#include "eigen_dist.hpp"

namespace p2vbme {

/* N aligned sequences packed as bitplanes: packed[(i * 5 + plane) * n_words + w], bit k of word w is
 * site 64 w + k; planes A, C, G, T/U, valid (= any of the four). Other characters (gaps, N, IUPAC
 * ambiguity codes) set no bit. Same layout as the .bin bundles. */
struct Alignment {
    int32_t n_taxa = 0;
    int32_t n_sites = 0;
    int32_t n_words = 0;
    std::vector<std::string> names;
    std::vector<uint64_t> packed;
};

/* Rooted binary tree on N leaves: leaves 0..N-1 (alignment order), internal nodes N..2N-2, root 2N-2, children
 * c1[u], c2[u]. In start trees children come before parents (c1[u], c2[u] < u); the search keeps the root but
 * reuses the other internal ids, so in its result they are in no particular order: traverse from the root.
 * branch_lengths[u] is the length of the edge above u (empty when not computed): float values, or double values
 * when double_precision_lengths (the CPU device's --precision double; the output keeps their digits). */
struct Tree {
    std::vector<int32_t> c1, c2;
    std::vector<double> branch_lengths;
    bool double_precision_lengths = false;
};

enum class Start { Sample, Random, Given };
enum class Accelerator { Auto, Gpu, Cpu };
enum class Precision { Single, Double };

/* The settings of a run; defaults in config.hpp. */
struct Options {
    Start start = Start::Sample;
    int32_t sample_size = DEFAULT_SAMPLE_SIZE;        /* sample-and-place start: 0 = min(100000, N / 10) */
    uint64_t seed = DEFAULT_SEED;                     /* seeds the sample-and-place or random start */
    int32_t max_rounds = DEFAULT_MAX_ROUNDS;          /* rounds of the search; 0 = no search (score the start) */
    double early_stopping = DEFAULT_EARLY_STOPPING;   /* stop once a round gains <= this x the total gain; 0 =
                                                         only when a round changes nothing (always on) */
    double time_limit = DEFAULT_TIME_LIMIT;           /* seconds; no new round after this (0 = none) */
    int32_t profile_cap = DEFAULT_PROFILE_CAP;        /* leaves kept per clade profile (1..256) */
    double exact_check_eta = DEFAULT_EXACT_CHECK_ETA; /* exact re-check of windows whose capped gain is > -eta */
    bool branch_lengths = DEFAULT_BRANCH_LENGTHS;     /* estimate BME branch lengths */
    int32_t bl_profile_cap = DEFAULT_BL_PROFILE_CAP;  /* leaves kept per subtree profile for branch lengths */
    DistanceModel distance;                           /* the model of the distances (JC69 by default); GTR+GC
                                                         with given parameters (the fit comes later) */
    Accelerator accelerator = Accelerator::Auto;      /* auto: the GPU if one is visible, else the CPU */
    std::vector<int> devices;                         /* GPU numbers; empty = all visible (at most MAX_GPUS) */
    int32_t threads = 0;                              /* CPU: workers (threads); 0 = all (omp_get_max_threads) */
    Precision precision = Precision::Single;          /* CPU only: single (the GPU's) or double;
                                                         double with accelerator auto selects the CPU */
};

struct Result {
    Tree tree;
    double bme_length = 0.0;          /* sum of the raw BME branch lengths */
    double delta_bme = 0.0;           /* BME change made by the search, from its start tree */
    double time_start = 0.0;          /* seconds: start tree */
    double time_start_sample = 0.0;   /* sample-and-place: the sample's own tree (subsample + its search) */
    double time_start_nearest = 0.0;  /*   the nearest sampled sequence of every other sequence */
    double time_start_assembly = 0.0; /*   assembling the full tree */
    double time_search = 0.0;         /* seconds: minibatch search */
    double time_branch_lengths = 0.0;
    std::vector<std::string> devices;         /* the devices used, e.g. "GPU 0: NVIDIA RTX A6000" */
    bool on_gpu = false;                      /* the run used the GPU (else the CPU) */
    DistanceModel distance;                   /* the model as used: with the base frequencies counted */
    double gpu_memory_in_use_at_end_mb = 0.0; /* MiB in use when the run ends, largest over the GPUs; GPU-wide (other
                                                 processes included), not a peak */
    int32_t sample_size_used = 0;
    int32_t rounds_planned = 0;
    int32_t rounds_run = 0;
    std::string stop_reason;         /* max_rounds, idle, early_stopping, time_limit, or none (no search) */
    std::vector<double> round_gains; /* BME decrease in each round of the main search (positive = shorter tree; the
                                        last is the cleanup pass when N > 1,050) */
};

/* Whether a GPU is visible (false in a build without CUDA). */
bool gpu_available();

/* Throws std::runtime_error if a setting is out of range (infer() checks them too; the command line checks them
 * before reading the alignment). */
void check_options(const Options &options);

/* Receives the run's progress lines (the library itself prints nothing). */
using LogCallback = std::function<void(const std::string &line)>;

/* Infers a BME tree. `start_tree` is used when options.start == Start::Given. Errors (input, CUDA,
 * GPU memory) are thrown as std::exception. */
Result infer(const Alignment &aln, const Options &options, const Tree *start_tree = nullptr,
             const LogCallback &log = nullptr);

} // namespace p2vbme
