/*
 * report.hpp
 *
 * The run report, <prefix>.json. The tree is in its own file.
 *
 * {
 *   "version", "command_line", "date",
 *   "bme_length", "rounds_run", "stop_reason", "time", "warnings",
 *   "details": {                                   with --report-full only
 *     "host", "input", "settings" (as used), "devices",
 *     "time" (per stage), "memory", "model" (its parameters), "round_gains"
 *   }
 * }
 */
#ifndef P2VBME_REPORT_HPP
#define P2VBME_REPORT_HPP

#include <stdint.h>

#include <string>
#include <vector>

#include "inference.hpp"

namespace p2vbme {

// What the command line knows about a run, besides Settings and Result.
struct RunInfo {
    std::vector<std::string> command_line;
    std::vector<std::string> warnings; // also printed on stderr
    std::string input_path;
    int32_t n_taxa = 0, n_sites = 0; // as read, before identical sequences are set aside
    std::string start_tree;          // sample, random, or the start tree's file
    std::string prefix;              // outputs <prefix>.nwk and <prefix>.json
    bool neg_as_zero = true;         // negative branch lengths written as 0
    bool dedup = true;               // identical sequences set aside (--no-dedup turns it off)
    int32_t n_identical_removed = 0; // sequences set aside and put back at the end
    double time_read = 0.0;          // seconds
    double time_total = 0.0;         // seconds, from start to the report

    // The devices of the run (select_devices) and their memory when it ends.
    std::vector<std::string> devices;         // the devices used, e.g. "GPU 0: NVIDIA RTX A6000"
    bool on_gpu = false;                      // the run used the GPU (else the CPU)
    std::string gpu_build_info;               // on the GPU: the CUDA versions and architectures
    double gpu_memory_in_use_at_end_mb = 0.0; // MiB in use when the run ends, largest over the GPUs; GPU-wide
                                              // (other processes included), not a peak
};

// The report as JSON text; full adds "details" (--report-full).
std::string report_json(const RunInfo &run, const Settings &settings, const Result &result, bool full);

// Writes report_json() to path.
void write_report(const std::string &path, const RunInfo &run, const Settings &settings, const Result &result,
                  bool full);

} // namespace p2vbme

#endif // P2VBME_REPORT_HPP
