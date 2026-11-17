/*
 * inference.cpp
 *
 * p2vbme::infer: one run of the method on an in-memory alignment, on the given devices: start tree, search,
 * branch lengths, with the settings taken from Settings.
 */
#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "branch_lengths.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "search.hpp"
#include "start_tree.hpp"

namespace p2vbme {

void check_settings(const Settings &opt) {
    if (opt.profile_cap < 1 || opt.profile_cap > PROFILE_CAP_MAX)
        throw std::runtime_error("profile cap must be in 1.." + std::to_string(PROFILE_CAP_MAX));
    if (opt.bl_profile_cap < 1 || opt.bl_profile_cap > PROFILE_CAP_MAX)
        throw std::runtime_error("branch-length profile cap must be in 1.." + std::to_string(PROFILE_CAP_MAX));
    if (opt.max_rounds < 0)
        throw std::runtime_error("max rounds must be >= 0");
    if (opt.sample_size < 0)
        throw std::runtime_error("sample size must be >= 0 (0 = the default rule)");
    if (!(opt.early_stopping >= 0.0))
        throw std::runtime_error("early stopping must be >= 0");
    if (!(opt.time_limit >= 0.0))
        throw std::runtime_error("time limit must be >= 0 (0 = none)");
    if (!(opt.exact_check_eta >= 0.0))
        throw std::runtime_error("exact-check eta must be >= 0");
    for (size_t i = 0; i < opt.devices.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (opt.devices[i] == opt.devices[j])
                throw std::runtime_error("GPU " + std::to_string(opt.devices[i]) + " is listed twice");
    for (double f : opt.substitution.base_freqs)
        if (!(f >= 0.0 && f <= 1e30))
            throw std::runtime_error("base frequencies must be finite and >= 0");
    if (opt.threads < 0)
        throw std::runtime_error("CPU threads must be >= 0 (0 = all)");
    if ((int32_t)opt.substitution.model < (int32_t)Model::JC69 ||
        (int32_t)opt.substitution.model > (int32_t)Model::GTR_GC)
        throw std::runtime_error("unknown distance model");
    if (opt.substitution.model == Model::GTR_GC) {
        // ranges the single-precision GPU evaluates accurately (the fit's own range is [0.05, 148])
        const SubstitutionModel &m = opt.substitution;
        if (!(m.gtr_alpha >= 1e-3 && m.gtr_alpha <= 1e4))
            throw std::runtime_error("GTR+GC: alpha must be in [0.001, 10000]");
        for (double r : m.gtr_rates)
            if (!(r > 0.0 && r <= 1e6))
                throw std::runtime_error("GTR+GC: exchangeabilities must be in (0, 1e6]");
        if (opt.eigen.reweight < 0 || opt.eigen.reweight > 100)
            throw std::runtime_error("GTR+GC: reweighting passes must be in 0..100");
    }
    // last, so that the other errors come first in a build without a GPU
    if (opt.threads > omp_get_num_procs())
        throw std::runtime_error("CPU threads must be at most the " + std::to_string(omp_get_num_procs()) +
                                 " processors available");
}

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
    if (opt.tree_init == TreeInit::Sample && N < START_SAMPLE_MIN_N)
        throw std::runtime_error("the sample-and-place start needs at least " + std::to_string(START_SAMPLE_MIN_N) +
                                 " sequences; use the random start");
    if (opt.tree_init == TreeInit::Given) {
        if (!start_tree || (int32_t)start_tree->c1.size() != n_nodes || (int32_t)start_tree->c2.size() != n_nodes)
            throw std::runtime_error("start tree missing or not on the alignment's taxa");
    }

    // F81, F84, TN93: the caller gives the base frequencies (prepare_alignment counts those of the whole input)
    const SubstitutionModel &substitution = opt.substitution;
    const double freq_sum = substitution.base_freqs[0] + substitution.base_freqs[1] + substitution.base_freqs[2] +
                            substitution.base_freqs[3];
    if ((substitution.model == Model::F81 || substitution.model == Model::F84 || substitution.model == Model::TN93) &&
        !(freq_sum > 0.0))
        throw std::runtime_error("F81, F84 and TN93 need the base frequencies (count_base_frequencies)");
    if (devices.empty())
        throw std::runtime_error("no devices given (select_devices)");
    const bool double_precision = opt.precision == Precision::Double;

    for (auto &dev : devices) {
        dev->load(aln.packed.data(), N, W);
        dev->set_distance(substitution, opt.eigen);
    }
    for (auto &dev : devices)
        dev->synchronize();

    SearchSettings settings;
    settings.rounds = opt.max_rounds;
    settings.stop_zero = true; // the lossless idle stop is always on
    settings.stop_tol = opt.early_stopping;
    settings.time_cap = opt.time_limit;
    settings.profile_cap = opt.profile_cap;
    settings.exact_check_eta = opt.exact_check_eta;

    Result res;
    res.substitution = substitution;
    // Progress lines for the caller's log (the library prints nothing): one per search start, stage and early stop,
    // and per sample-and-place level (rf: -1 = no reference tree, -2 = not computed at this stage).
    double t_start = 0.0, t_last = 0.0, dL_last = 0.0, round_end_dL = 0.0;
    settings.on_event = [&](const SearchEvent &e) {
        if (e.level == 0 && e.kind == SearchEvent::Stage && e.c1) { // a round end of the main search
            res.round_gains.push_back(round_end_dL - e.total_dL);   // positive = the tree got shorter
            round_end_dL = e.total_dL;
        }
        if (!log)
            return;
        {
            char line[512];
            if (e.kind == SearchEvent::Start) {
                t_start = t_last = wall_sec();
                dL_last = 0.0;
                snprintf(line, sizeof(line),
                         "[DIAG] N=%d n_gpus=%d n_outer_rounds=%d max_prof_leaves=%d stop_zero=%d stop_tol=%g "
                         "time_cap=%g",
                         e.n_leaves, e.n_devices, e.rounds_planned, e.settings->profile_cap, (int)e.settings->stop_zero,
                         e.settings->stop_tol, e.settings->time_cap);
            } else if (e.kind == SearchEvent::Stop) {
                snprintf(line, sizeof(line),
                         "[DIAG] stopping after round %d (gain %.6f, round-0 gain %.6f); running final cleanup",
                         e.round, e.round_gain, e.round0_gain);
            } else if (e.kind == SearchEvent::StartTree) {
                const SampleInitStats &st = *e.init_stats;
                snprintf(line, sizeof(line),
                         "[INIT] sample-and-place: n0=%d sub=%.2fs (dL %.4f) nn=%.2fs build=%.2fs total=%.2fs "
                         "max_bucket=%d",
                         st.n0, st.dt_sub, st.dL_sub, st.dt_nn, st.dt_build, st.dt_total, st.max_bucket);
            } else {
                const double t = wall_sec();
                snprintf(line, sizeof(line),
                         "[DIAG] round=%d stage=%-10s sc=%-9lld dL=%+14.6f dt=%8.2fs t=%9.2fs rf=%lld", e.round,
                         e.stage, (long long)e.detail, e.total_dL - dL_last, t - t_last, t - t_start,
                         e.c1 ? -1LL : -2LL);
                t_last = t;
                dL_last = e.total_dL;
            }
            log(line);
        }
    };

    const double t0 = wall_sec();
    if (settings.time_cap > 0.0)
        settings.deadline = t0 + settings.time_cap; // covers start tree + search
    std::vector<int32_t> c1(n_nodes), c2(n_nodes);
    if (opt.tree_init == TreeInit::Sample) {
        const int32_t n0 =
            opt.sample_size > 0 ? opt.sample_size : std::min<int32_t>(START_SAMPLE_MAX, N / START_SAMPLE_DIVISOR);
        SampleInitStats stats;
        // the start tree's own searches use the run's settings;
        // with max_rounds = 0 (score the start tree) they keep the default rounds,
        // so the start tree is the one a normal run starts from
        SearchSettings start_settings = settings;
        if (start_settings.rounds == 0)
            start_settings.rounds = DEFAULT_MAX_ROUNDS;
        build_sample_init_tree(N, n0, opt.seed, devices, c1, c2, stats, start_settings);
        res.sample_size_used = stats.n0;
        res.time_start_sample = stats.dt_sub;
        res.time_start_nearest = stats.dt_nn;
        res.time_start_assembly = stats.dt_build;
    } else if (opt.tree_init == TreeInit::Random) {
        make_uniform_random_tree_host(N, opt.seed, c1, c2);
    } else {
        c1 = start_tree->c1;
        c2 = start_tree->c2;
    }
    const double t1 = wall_sec();
    res.time_start = t1 - t0;

    SearchStats search_stats;
    res.delta_bme = run_multi_gpu_minibatch_bme(N, devices, c1, c2, settings, 0, &search_stats);
    for (auto &dev : devices)
        dev->synchronize();
    res.time_search = wall_sec() - t1;
    res.rounds_planned = search_stats.rounds_planned;
    res.rounds_run = search_stats.rounds_run;
    res.stop_reason = search_stats.stop_reason;

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
