/*
 * report.cpp
 *
 * The run report as JSON (json_writer.hpp).
 */
#include "report.hpp"

#include <omp.h>
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <stdexcept>
#include <string>

#include "config.hpp"
#include "device.hpp"
#include "json_writer.hpp"
#include "version.hpp"

namespace p2vbme {

namespace {

std::string utc_now() {
    char buf[32];
    const time_t t = time(NULL);
    struct tm tm_utc;
    gmtime_r(&t, &tm_utc);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

std::string host_name() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) != 0)
        return "unknown";
    return buf;
}

/* Peak resident memory of the process, MiB. */
double peak_host_ram_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;
}

const char *start_name(Start s) { return s == Start::Sample ? "sample" : (s == Start::Random ? "random" : "given"); }

const char *accelerator_name(Accelerator a) {
    return a == Accelerator::Auto ? "auto" : (a == Accelerator::Gpu ? "gpu" : "cpu");
}

} // namespace

std::string report_json(const RunInfo &run, const Options &opt, const Result &res) {
    std::string out;
    JsonWriter w(out);
    w.begin_object();

    w.key("provenance");
    w.begin_object();
    w.field("program", "p2v-bme");
    w.field("version", package_version());
    w.field("git", package_git_revision());
    w.key("build");
    w.begin_object();
    w.field("compiler", __VERSION__);
    w.field("gpu", gpu_build_info());
    w.end_object();
    w.array_field("command_line", run.command_line);
    w.field("date", utc_now());
    w.field("host", host_name());
    w.key("input");
    w.begin_object();
    w.field("path", run.input_path);
    w.field("bytes", run.input_bytes);
    w.field("sequences", run.n_taxa);
    w.field("sites", run.n_sites);
    w.field("identical_set_aside", run.n_identical_removed);
    w.end_object();
    w.key("settings");
    w.begin_object();
    w.field("model", run.model);
    w.field("start_tree", run.start_tree);
    w.field("start", start_name(opt.start));
    w.field("sample_size", opt.sample_size);
    w.field("seed", opt.seed);
    w.field("max_rounds", opt.max_rounds);
    w.field("early_stopping", opt.early_stopping);
    w.field("time_limit", opt.time_limit);
    w.field("profile_cap", opt.profile_cap);
    w.field("exact_check_eta", opt.exact_check_eta);
    w.field("branch_lengths", opt.branch_lengths);
    w.field("bl_profile_cap", opt.bl_profile_cap);
    w.field("neg_as_zero", run.neg_as_zero);
    w.field("dedup", run.dedup);
    w.field("accelerator", accelerator_name(opt.accelerator));
    w.array_field("devices", opt.devices);
    w.field("threads", opt.threads);
    w.field("precision", opt.precision == Precision::Double ? "double" : "single");
    w.field("prefix", run.prefix);
    w.end_object();
    w.key("constants");
    w.begin_object();
    w.field("max_gpus", MAX_GPUS);
    w.field("profile_cap_max", PROFILE_CAP_MAX);
    w.field("workspace_max_k", WORKSPACE_MAX_K);
    w.field("window_max_sweeps", WINDOW_MAX_SWEEPS);
    w.field("window_max_internals", WINDOW_MAX_INTERNALS);
    w.field("wave_window_max_internals", WAVE_WINDOW_MAX_INTERNALS);
    w.field("wave_component_cap", WAVE_COMPONENT_CAP);
    w.field("move_batch_cap", MOVE_BATCH_CAP);
    w.field("exact_profile_cap_min_n", EXACT_PROFILE_CAP_MIN_N);
    w.field("exact_profile_cap", EXACT_PROFILE_CAP);
    w.field("backbone_min_n", BACKBONE_MIN_N);
    w.field("vcycle_min_n", VCYCLE_MIN_N);
    w.field("vcycle_base_scale", VCYCLE_BASE_SCALE);
    w.field("vcycle_scale_factor", VCYCLE_SCALE_FACTOR);
    w.field("vcycle_finest_scale", VCYCLE_FINEST_SCALE);
    w.field("start_sample_min_n", START_SAMPLE_MIN_N);
    w.field("start_sample_max", START_SAMPLE_MAX);
    w.field("start_sample_divisor", START_SAMPLE_DIVISOR);
    w.field("start_sample_min", START_SAMPLE_MIN);
    w.field("start_random_tree_max", START_RANDOM_TREE_MAX);
    w.field("nn_p_no_overlap", NN_P_NO_OVERLAP);
    w.end_object();
    w.field("accelerator", res.on_gpu ? "gpu" : "cpu");
    w.array_field("devices", res.devices);
    w.field("threads", (int32_t)omp_get_max_threads());
    w.end_object();

    w.key("results");
    w.begin_object();
    w.key("bme_length"); /* null without branch lengths */
    if (opt.branch_lengths)
        w.value(res.bme_length);
    else
        w.null();
    w.field("delta_bme", res.delta_bme);
    w.field("rounds_planned", res.rounds_planned);
    w.field("rounds_run", res.rounds_run);
    w.field("stop_reason", res.stop_reason);
    w.field("sample_size_used", res.sample_size_used);
    w.key("time");
    w.begin_object();
    w.field("read", run.time_read);
    w.field("start_tree", res.time_start);
    w.field("start_tree_sample", res.time_start_sample);
    w.field("start_tree_nearest", res.time_start_nearest);
    w.field("start_tree_assembly", res.time_start_assembly);
    w.field("search", res.time_search);
    w.field("branch_lengths", res.time_branch_lengths);
    w.field("write", run.time_write);
    w.field("total", run.time_total);
    w.end_object();
    w.field("host_ram_mb", peak_host_ram_mb());
    w.field("gpu_memory_in_use_at_end_mb", res.gpu_memory_in_use_at_end_mb);
    w.end_object();

    w.key("model");
    w.begin_object();
    w.field("name", run.model);
    const Model m = res.distance.model;
    if (m == Model::F81 || m == Model::F84 || m == Model::TN93) {
        std::vector<double> freqs(res.distance.base_freqs, res.distance.base_freqs + 4);
        w.array_field("base_frequencies", freqs); /* A, C, G, T: the input alignment's */
    }
    w.end_object();

    w.key("diagnostics");
    w.begin_object();
    w.array_field("round_gains", res.round_gains);
    w.end_object();

    w.end_object();
    w.finish();
    return out;
}

void write_report(const std::string &path, const RunInfo &run, const Options &options, const Result &result) {
    const std::string text = report_json(run, options, result);
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    if (fclose(f) != 0 || !ok)
        throw std::runtime_error("cannot write " + path);
}

} // namespace p2vbme
