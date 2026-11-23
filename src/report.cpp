/*
 * report.cpp
 *
 * The run report as JSON (json_writer.hpp): the summary, then the details with --report-full, one detail_ function
 * per section.
 */
#include "report.hpp"

#include <sys/resource.h>
#include <time.h>

#include <fstream>
#include <stdexcept>
#include <string>

#include "accelerator.hpp"
#include "distances.hpp"
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

// Peak resident memory of the process, MiB (ru_maxrss is in KiB on Linux).
double peak_host_ram_mib() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;
}

// The input as read, before identical sequences are set aside.
void detail_input(JsonWriter &w, const Report &run) {
    w.key("input");
    w.begin_object();
    w.field("path", run.input_path);
    w.field("sequences", run.n_taxa);
    w.field("sites", run.n_sites);
    w.field("identical_set_aside", run.n_identical_removed);
    w.end_object();
}

// Every setting, defaults included, with the auto values resolved: accelerator, sample size.
void detail_settings(JsonWriter &w, const Report &run, const Settings &opt, const Result &res) {
    w.key("settings");
    w.begin_object();
    w.field("start_tree", run.tree_init);
    w.field("sample_size", res.sample_size_used); // 0 without sample-and-place
    w.field("seed", opt.seed);
    w.field("max_rounds", opt.max_rounds);
    w.field("tol", opt.tol);
    w.field("time_limit", opt.time_limit);
    w.field("representative_cap", opt.representative_cap);
    w.field("branch_lengths", opt.branch_lengths);
    w.field("bl_representative_cap", opt.bl_representative_cap);
    w.field("neg_as_zero", run.neg_as_zero);
    w.field("dedup", run.dedup);
    w.field("gtr_fit_pairs", run.gtr_fit_pairs);
    w.field("bootstrap", run.bootstrap);
    w.field("accelerator", accelerator_name(run.on_gpu ? Accelerator::Gpu : Accelerator::Cpu));
    w.field("precision", precision_name(opt.precision));
    w.end_object();
}

// The GPUs used, or one line for the CPU workers.
void detail_devices(JsonWriter &w, const Report &run) {
    std::vector<std::string> devices;
    if (run.on_gpu) {
        devices = run.devices;
    } else {
        devices.push_back("CPU: " + std::to_string(run.devices.size()) + " threads");
    }
    w.array_field("devices", devices);
}

// Seconds per stage; the total is in the summary.
void detail_time(JsonWriter &w, const Report &run, const Result &res) {
    w.key("time");
    w.begin_object();
    w.field("read", run.time_read);
    w.field("gtr_fit", run.gtr_fit.time);
    w.field("start_tree", res.time_start);
    w.field("search", res.time_search);
    w.field("branch_lengths", res.time_branch_lengths);
    w.field("bootstrap", run.time_bootstrap);
    w.end_object();
}

void detail_memory(JsonWriter &w, const Report &run) {
    w.key("memory");
    w.begin_object();
    w.field("host_ram_mib", peak_host_ram_mib()); // peak
    // GPU-wide (other processes included) and not a peak
    w.field("gpu_memory_in_use_at_end_mib", run.gpu_memory_in_use_at_end_mib);
    w.end_object();
}

// The substitution model of the distances: its name and parameters.
// F81, F84 and TN93: the input alignment's base frequencies (A, C, G, T).
// GTR+GC: the exchangeabilities (AC, AG, AT, CG, CT, GT) and alpha, and how their fit went (the base frequencies
// pooled over its pairs are the fit's, not the distance's, which uses each pair's own).
void substitution_model(JsonWriter &w, const Report &run, const Settings &opt) {
    const SubstitutionModel &m = opt.substitution_model;
    w.key("substitution_model");
    w.begin_object();
    w.field("name", model_name(m.model));
    if (m.model == Model::F81 || m.model == Model::F84 || m.model == Model::TN93) {
        std::vector<double> freqs(m.base_freqs, m.base_freqs + N_BASES);
        w.array_field("base_frequencies", freqs);
    }
    if (m.model == Model::GTR_GC) {
        const GtrFit &fit = run.gtr_fit;
        w.array_field("rates", std::vector<double>(m.rates, m.rates + N_GTR_RATES));
        w.field("alpha", m.alpha);
        w.key("fit");
        w.begin_object();
        w.field("pairs_used", fit.n_pairs);
        w.field("iterations", fit.n_iterations);
        w.field("converged", fit.converged);
        w.field("log_likelihood", fit.log_lik);
        w.array_field("base_frequencies", std::vector<double>(fit.base_freqs, fit.base_freqs + N_BASES));
        w.end_object();
    }
    w.end_object();
}

} // namespace

std::string report_json(const Report &run, const Settings &opt, const Result &res) {
    std::string out;
    JsonWriter w(out);
    w.begin_object();
    w.field("version", version_string());
    w.array_field("command_line", run.command_line);
    w.field("date", utc_now());
    substitution_model(w, run, opt);
    w.key("bme_length"); // null without branch lengths
    if (opt.branch_lengths)
        w.value(res.bme_length);
    else
        w.null();
    w.field("rounds_run", res.rounds_run);
    w.field("stop_reason", stop_reason_name(res.stop_reason));
    if (run.bootstrap_last > 0) { // the replicates' trees; -b B: the mean supports of the internal branches (each
                                  // branch's are in <prefix>.support.nwk)
        w.key("bootstrap");
        w.begin_object();
        w.field("replicates", run.bootstrap_last - run.bootstrap_first + 1);
        w.field("first", run.bootstrap_first);
        w.field("last", run.bootstrap_last);
        w.field("trees", run.prefix + ".boottrees");
        if (!run.bootstrap_only) {
            w.field("mean_tbe", run.mean_tbe);
            w.field("mean_exact", run.mean_exact);
        }
        w.end_object();
    }
    w.field("time", run.time_total);
    w.array_field("warnings", run.warnings);
    if (run.full) {
        w.key("details");
        w.begin_object();
        detail_input(w, run);
        detail_settings(w, run, opt, res);
        detail_devices(w, run);
        detail_time(w, run, res);
        detail_memory(w, run);
        w.array_field("round_gains", res.round_gains);
        w.end_object();
    }
    w.end_object();
    w.finish();
    return out;
}

void write_report(const std::string &path, const Report &run, const Settings &settings, const Result &result) {
    const std::string text = report_json(run, settings, result);
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    out << text;
    // close() flushes; a failed write or flush leaves the stream failed.
    out.close();
    if (!out)
        throw std::runtime_error("cannot write " + path);
}

void write_supports_report(const std::string &path, const Report &run, const SupportsReport &supports) {
    std::string text;
    JsonWriter w(text);
    w.begin_object();
    w.field("version", version_string());
    w.array_field("command_line", run.command_line);
    w.field("date", utc_now());
    w.key("supports");
    w.begin_object();
    w.field("tree", run.tree_init);
    w.field("replicates", run.supports);
    w.field("n_replicates", supports.n_replicates);
    w.field("leaves", supports.n_leaves);
    w.field("leaves_supported", supports.n_leaves_supported);
    w.field("mean_tbe", supports.mean_tbe);
    w.field("mean_exact", supports.mean_exact);
    w.end_object();
    w.field("time", run.time_total);
    w.end_object();
    w.finish();
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    out << text;
    out.close();
    if (!out)
        throw std::runtime_error("cannot write " + path);
}

} // namespace p2vbme
