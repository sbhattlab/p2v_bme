/*
 * inference.cpp
 *
 * p2vbme::infer: one run of the method on an in-memory alignment, on the given devices: start tree, search,
 * branch lengths, with the settings taken from Settings.
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

#include "branch_lengths.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "search.hpp"
#include "start_tree.hpp"
#include "timing.hpp"

namespace p2vbme {

namespace {

// Throws std::runtime_error if a setting is out of range.
// The device settings (devices, threads) are checked by select_devices, which uses them.
void check_settings(const Settings &opt) {
    if (opt.profile_cap < 1 || opt.profile_cap > PROFILE_CAP_MAX)
        throw std::runtime_error("profile cap must be in 1.." + std::to_string(PROFILE_CAP_MAX));
    if (opt.bl_profile_cap < 1 || opt.bl_profile_cap > PROFILE_CAP_MAX)
        throw std::runtime_error("branch-length profile cap must be in 1.." + std::to_string(PROFILE_CAP_MAX));
    if (opt.max_rounds < 0)
        throw std::runtime_error("max rounds must be >= 0");
    if (opt.sample_size < 0)
        throw std::runtime_error("sample size must be >= 0 (0 = the default rule)");
    if (!(opt.tol >= 0.0))
        throw std::runtime_error("tol must be >= 0");
    if (!(opt.time_limit >= 0.0))
        throw std::runtime_error("time limit must be >= 0 (0 = none)");
    if (!(opt.exact_check_eta >= 0.0))
        throw std::runtime_error("exact-check eta must be >= 0");
    for (double f : opt.substitution_model.base_freqs)
        if (!(std::isfinite(f) && f >= 0.0))
            throw std::runtime_error("base frequencies must be finite and >= 0");
    if ((int32_t)opt.substitution_model.model < (int32_t)Model::JC69 ||
        (int32_t)opt.substitution_model.model > (int32_t)Model::GTR_GC)
        throw std::runtime_error("unknown distance model");
    if (opt.substitution_model.model == Model::GTR_GC) {
        // What the distance needs.
        // Single precision divides by alpha, so alpha must be a normal float.
        // The exchangeabilities are scaled to mean 1 (distance_params), so their sum must be finite.
        const SubstitutionModel &m = opt.substitution_model;
        if (!(m.gtr_alpha >= std::numeric_limits<float>::min() && m.gtr_alpha <= std::numeric_limits<float>::max()))
            throw std::runtime_error("GTR+GC: alpha must be > 0 and finite in single precision");
        double rate_sum = 0.0;
        for (double r : m.gtr_rates) {
            if (!(std::isfinite(r) && r > 0.0))
                throw std::runtime_error("GTR+GC: exchangeabilities must be finite and > 0");
            rate_sum += r;
        }
        if (!std::isfinite(rate_sum))
            throw std::runtime_error("GTR+GC: the sum of the exchangeabilities must be finite");
        if (opt.eigen.reweight < 0)
            throw std::runtime_error("GTR+GC: reweighting passes must be >= 0");
    }
}

// Warnings for GTR+GC parameters outside the range where single precision was checked against double (config.hpp).
void warn_gtr_precision(const Settings &opt, std::vector<std::string> &warnings) {
    const SubstitutionModel &m = opt.substitution_model;
    if (m.model != Model::GTR_GC || opt.precision != Precision::Single)
        return;
    const char *consequence = "; single-precision distances may be less accurate (--precision double avoids this)";
    if (m.gtr_alpha < GTR_FLOAT_ALPHA_MIN || m.gtr_alpha > GTR_FLOAT_ALPHA_MAX) {
        std::ostringstream w;
        w << "GTR+GC: alpha = " << m.gtr_alpha << " is outside [" << GTR_FLOAT_ALPHA_MIN << ", " << GTR_FLOAT_ALPHA_MAX
          << "]" << consequence;
        warnings.push_back(w.str());
    }
    const auto lo_hi = std::minmax_element(m.gtr_rates, m.gtr_rates + 6);
    const double ratio = *lo_hi.second / *lo_hi.first;
    if (ratio > GTR_FLOAT_RATE_RATIO_MAX) {
        std::ostringstream w;
        w << "GTR+GC: the largest exchangeability is " << ratio << " times the smallest, above "
          << GTR_FLOAT_RATE_RATIO_MAX << consequence;
        warnings.push_back(w.str());
    }
}

} // namespace

Result infer(const Alignment &aln, const Settings &opt, Devices &devices, const Tree *start_tree,
             const LogCallback &log) {
    check_settings(opt);
    const int32_t N = aln.n_taxa;
    const int32_t W = aln.n_words;
    const int32_t n_nodes = 2 * N - 1;
    if (N < 4)
        throw std::runtime_error("need at least 4 sequences, got " + std::to_string(N));
    if ((size_t)N * 5 * (size_t)W != aln.packed.size())
        throw std::runtime_error("alignment: packed size mismatch");
    if (opt.tree_init == TreeInit::Given) {
        if (!start_tree || (int32_t)start_tree->c1.size() != n_nodes || (int32_t)start_tree->c2.size() != n_nodes)
            throw std::runtime_error("start tree missing or not on the alignment's taxa");
    }

    // F81, F84, TN93: the caller gives the base frequencies (prepare_alignment counts those of the whole input)
    const SubstitutionModel &substitution = opt.substitution_model;
    const double freq_sum = substitution.base_freqs[0] + substitution.base_freqs[1] + substitution.base_freqs[2] +
                            substitution.base_freqs[3];
    if ((substitution.model == Model::F81 || substitution.model == Model::F84 || substitution.model == Model::TN93) &&
        !(freq_sum > 0.0))
        throw std::runtime_error("F81, F84 and TN93 need the base frequencies (count_base_frequencies)");
    if (devices.empty())
        throw std::runtime_error("no devices given (select_devices)");
    const bool double_precision = opt.precision == Precision::Double;

    // Load the alignment and set the distance model on all devices.
    for (auto &dev : devices) {
        dev->load(aln.packed.data(), N, W);
        dev->set_distance(substitution, opt.eigen);
    }
    // The devices are used in parallel, so they must all finish their work before the next step.
    for (auto &dev : devices)
        dev->synchronize();

    SearchSettings settings;
    settings.rounds = opt.max_rounds;
    settings.stop_tol = opt.tol;
    settings.time_cap = opt.time_limit;
    settings.profile_cap = opt.profile_cap;
    settings.exact_check_eta = opt.exact_check_eta;

    Result res;
    warn_gtr_precision(opt, res.warnings);
    res.tree_init = opt.tree_init;
    if (res.tree_init == TreeInit::Sample && N < START_SAMPLE_MIN_N) {
        res.tree_init = TreeInit::Random;
        res.warnings.push_back("fewer than " + std::to_string(START_SAMPLE_MIN_N) +
                               " sequences, using the random start instead of sample-and-place");
    }
    // Progress lines for the caller's log (the library prints nothing): one per search start, stage and early stop
    // (rf: -1 = no reference tree, -2 = not computed at this stage).
    double t_start = 0.0, t_last = 0.0, dL_last = 0.0, round_end_dL = 0.0;
    settings.on_event = [&](const SearchEvent &e) {
        if (e.level == 0 && e.kind == SearchEvent::Stage && e.c1) { // a round end of the main search
            res.round_gains.push_back(round_end_dL - e.total_dL);   // positive = the tree got shorter
            round_end_dL = e.total_dL;
        }
        if (!log)
            return;
        std::ostringstream line;
        if (e.kind == SearchEvent::Start) {
            t_start = t_last = wall_sec();
            dL_last = 0.0;
            line << "[DIAG] N=" << e.n_leaves << " n_gpus=" << e.n_devices << " n_outer_rounds=" << e.rounds_planned
                 << " max_prof_leaves=" << settings.profile_cap << " stop_tol=" << settings.stop_tol
                 << " time_cap=" << settings.time_cap;
        } else if (e.kind == SearchEvent::Stop) {
            line << std::fixed << std::setprecision(6) << "[DIAG] stopping after round " << e.round << " (gain "
                 << e.round_gain << ", round-0 gain " << e.round0_gain << "); running final cleanup";
        } else {
            const double t = wall_sec();
            line << std::fixed << "[DIAG] round=" << e.round << " stage=" << std::left << std::setw(10) << e.stage
                 << " sc=" << std::setw(9) << e.detail << std::right << " dL=" << std::showpos << std::setprecision(6)
                 << std::setw(14) << e.total_dL - dL_last << std::noshowpos << std::setprecision(2)
                 << " dt=" << std::setw(8) << t - t_last << "s t=" << std::setw(9) << t - t_start
                 << "s rf=" << (e.c1 ? -1 : -2);
            t_last = t;
            dL_last = e.total_dL;
        }
        log(line.str());
    };

    const double t0 = wall_sec();
    if (settings.time_cap > 0.0)
        settings.deadline = t0 + settings.time_cap; // covers start tree + search
    // Set up the start tree (sample-and-place, random, or given) in c1/c2, and time it.
    std::vector<int32_t> c1(n_nodes), c2(n_nodes);
    if (res.tree_init == TreeInit::Sample) {
        int32_t n0 =
            opt.sample_size > 0 ? opt.sample_size : std::min<int32_t>(START_SAMPLE_MAX, N / START_SAMPLE_DIVISOR);
        // the start tree's own searches use the run's settings;
        // with max_rounds = 0 (score the start tree) they keep the default rounds,
        // so the start tree is the one a normal run starts from
        SearchSettings start_settings = settings;
        if (start_settings.rounds == 0)
            start_settings.rounds = DEFAULT_MAX_ROUNDS;
        build_sample_and_place_tree(N, n0, opt.seed, devices, c1, c2, start_settings);
        res.sample_size_used = n0; // after the clamp
    } else if (res.tree_init == TreeInit::Random) {
        build_random_tree(N, opt.seed, c1, c2);
    } else {
        c1 = start_tree->c1;
        c2 = start_tree->c2;
    }
    const double t1 = wall_sec();
    res.time_start = t1 - t0;

    SearchStats search_stats;
    // Core of the method: the minibatch BME search, which edits c1/c2 in place.
    res.delta_bme = run_minibatch_bme(N, devices, c1, c2, settings, 0, &search_stats);
    for (auto &dev : devices)
        dev->synchronize();
    res.time_search = wall_sec() - t1;
    res.rounds_planned = search_stats.rounds_planned;
    res.rounds_run = search_stats.rounds_run;
    res.stop_reason = search_stats.stop_reason;

    // Compute the BME branch lengths of the final tree
    if (opt.branch_lengths) {
        double prof_s = 0.0, kern_s = 0.0;
        res.bme_length = compute_bme_branch_lengths(N, c1.data(), c2.data(), devices, res.tree.branch_lengths,
                                                    opt.bl_profile_cap, &prof_s, &kern_s, double_precision);
        res.tree.double_precision_lengths = double_precision;
        res.time_branch_lengths = prof_s + kern_s;
    }
    res.tree.c1 = c1;
    res.tree.c2 = c2;
    return res;
}

} // namespace p2vbme
