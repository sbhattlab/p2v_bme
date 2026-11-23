/*
 * report.hpp
 *
 * The run report, <prefix>.json. The tree is in its own file.
 *
 * {
 *   "version", "command_line", "date", "substitution_model" (name and parameters; GTR+GC: its fit),
 *   "bme_length", "rounds_run", "stop_reason", "time", "warnings",
 *   "details": {                                   with --report-full only
 *     "input", "settings" (as used), "devices",
 *     "time" (per stage), "memory", "round_gains"
 *   }
 * }
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
    int32_t gtr_fit_pairs = DEFAULT_GTR_FIT_PAIRS; // GTR+GC: sequence pairs drawn for the fit (--gtr-fit-pairs)

    // Recorded during the run.
    std::vector<std::string> warnings; // also printed on stderr
    int32_t n_taxa = 0, n_sites = 0;   // as read, before identical sequences are set aside
    int32_t n_identical_removed = 0;   // sequences set aside and put back at the end
    GtrFit gtr_fit;                    // GTR+GC: the fit of the rates and alpha (n_pairs = 0: none)
    double time_read = 0.0;            // seconds
    double time_total = 0.0;           // seconds, from start to the report

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

} // namespace p2vbme

#endif // P2VBME_REPORT_HPP
