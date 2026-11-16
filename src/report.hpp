/*
 * report.hpp
 *
 * The run report, <prefix>.json: provenance (version, build, command line, every setting
 * and the algorithm's constants, input, date, host, devices), results (BME length, rounds, stop reason,
 * timings per stage, memory) and the model with the search's diagnostics. The tree is in its own file.
 */
#ifndef P2VBME_REPORT_HPP
#define P2VBME_REPORT_HPP

#include <stdint.h>

#include <string>
#include <vector>

#include "inference.hpp"

namespace p2vbme {

/* What the command line knows about a run, besides Options and Result. */
struct RunInfo {
    std::vector<std::string> command_line;
    std::string input_path;
    uint64_t input_bytes = 0;
    int32_t n_taxa = 0, n_sites = 0;
    std::string model;               /* as normalised by the command line, e.g. JC69 */
    std::string start_tree;          /* sample, random, or the start tree's file */
    std::string prefix;              /* outputs <prefix>.nwk and <prefix>.json */
    bool neg_as_zero = true;         /* negative branch lengths written as 0 */
    bool dedup = true;               /* identical sequences set aside (--no-dedup turns it off) */
    int32_t n_identical_removed = 0; /* sequences set aside and put back at the end */
    double time_read = 0.0;          /* seconds */
    double time_write = 0.0;         /* seconds: the tree file */
    double time_total = 0.0;         /* seconds, from start to the report */
};

/* The report as JSON text. */
std::string report_json(const RunInfo &run, const Options &options, const Result &result);

/* Writes report_json() to path. Throws std::runtime_error. */
void write_report(const std::string &path, const RunInfo &run, const Options &options, const Result &result);

} // namespace p2vbme

#endif // P2VBME_REPORT_HPP
