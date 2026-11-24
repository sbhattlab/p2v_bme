/*
 * report.hpp
 *
 * The run report, <prefix>.json. The tree is in its own file.
 *
 * {
 *   "version", "command_line", "date", "substitution_model" (name and parameters; GTR, GTR+GC: the fit),
 *   "bme_length", "rounds_run", "stop_reason", "bootstrap" (with -b only), "time", "warnings",
 *   "details": {                                   with --report-full only
 *     "input", "settings" (as used), "devices",
 *     "time" (per stage), "memory", "round_gains"
 *   }
 * }
 * With -b A-B (replicates alone) the fields of the search sum over the replicates (rounds_run, the start tree and
 * search times; stop_reason is the last replicate's), and bme_length is null.
 */
#ifndef P2VBME_REPORT_HPP
#define P2VBME_REPORT_HPP

#include <stdint.h>

#include <string>
#include <vector>

#include "gtr_fit.hpp"
#include "inference.hpp"

namespace p2vbme {

// The run around infer(), for <prefix>.json (which adds Settings and Result): what the command line gave
// (parse_command_line) and what the run recorded.
struct Report {
    // From the command line.
    std::vector<std::string> command_line;
    std::string input_path;  // -s
    std::string prefix;      // outputs <prefix>.nwk and <prefix>.json (default: the input path)
    std::string tree_init;   // -t: sample, random, or a Newick file (random when sample falls back)
    bool dedup = true;       // identical sequences set aside (--no-dedup turns it off)
    bool neg_as_zero = true; // negative branch lengths written as 0 (--no-neg-as-zero turns it off)
    bool full = false;       // --report-full: add "details"
    int32_t gtr_fit_pairs = DEFAULT_GTR_FIT_PAIRS; // GTR, GTR+GC: sequence pairs drawn for the fit (--gtr-fit-pairs)
    std::string bootstrap = "0";                   // -b as given: B, A-B, or 0 (none)
    int32_t bootstrap_first = 0;                   // -b: the replicates run, first..last (-b B: 1..B; 0 = none)
    int32_t bootstrap_last = 0;
    bool bootstrap_only = false; // -b A-B: the replicates alone, without a tree of their own
    std::string supports;        // --supports: the file of replicate trees (empty = none)

    // Recorded during the run.
    std::vector<std::string> warnings;       // also printed on stderr
    int32_t n_taxa = 0, n_sites = 0;         // as read, before identical sequences are set aside
    int32_t n_identical_removed = 0;         // sequences set aside and put back at the end
    GtrFit gtr_fit;                          // GTR, GTR+GC: the fit of the rates (and alpha) (n_pairs = 0: none)
    double mean_tbe = 0.0, mean_exact = 0.0; // bootstrap: mean supports of the internal branches, in %
    double time_bootstrap = 0.0;             // seconds
    double time_read = 0.0;                  // seconds
    double time_total = 0.0;                 // seconds, from start to the report

    // The devices of the run (select_devices) and their memory when it ends.
    std::vector<std::string> devices;          // the devices used, e.g. "GPU 0: NVIDIA RTX A6000"
    bool on_gpu = false;                       // the run used the GPU (else the CPU)
    double gpu_memory_in_use_at_end_mib = 0.0; // MiB in use when the run ends, largest over the GPUs; GPU-wide
                                               // (other processes included), not a peak
};

// The report as JSON text; run.full adds "details" (--report-full).
std::string report_json(const Report &run, const Settings &settings, const Result &result);

// Writes report_json() to path.
void write_report(const std::string &path, const Report &run, const Settings &settings, const Result &result);

// The supports of a tree's branches over saved replicate trees (--supports), for <prefix>.json.
struct SupportsReport {
    int32_t n_replicates = 0;
    int32_t n_leaves = 0;                    // of the tree
    int32_t n_leaves_supported = 0;          // the replicates' leaves (the tree is pruned to them)
    double mean_tbe = 0.0, mean_exact = 0.0; // in %, over the internal branches of the pruned tree
};

// Writes the report of --supports to path: {"version", "command_line", "date", "supports": {"tree", "replicates",
// "n_replicates", "leaves", "leaves_supported", "mean_tbe", "mean_exact"}, "time"}.
void write_supports_report(const std::string &path, const Report &run, const SupportsReport &supports);

} // namespace p2vbme

#endif // P2VBME_REPORT_HPP
