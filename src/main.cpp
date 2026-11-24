/*
 * main.cpp
 *
 * p2v-bme: a balanced minimum evolution tree from an aligned DNA alignment (Phylo2Vec minibatch search). Writes
 * the unrooted tree to <prefix>.nwk and the run report to <prefix>.json. The run, step by step: the arguments
 * (cli), the devices (device), the alignment (io), before inference (processing), inference, after inference,
 * the tree (newick) and the report.
 * Progress on stdout (none with -q); warnings and errors on stderr.
 * <prefix>.log gets all of it, the progress also with -q.
 * A run stops before it starts if a previous run's outputs exist, unless --overwrite.
 * With -b the bootstrap replicates' trees go to <prefix>.boottrees; -b A-B runs those replicates without the tree.
 * With --supports it only reads trees: the supports of a tree's branches over saved replicate trees.
 */
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "bootstrap.hpp"
#include "cli.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "io.hpp"
#include "newick.hpp"
#include "processing.hpp"
#include "report.hpp"
#include "support.hpp"

namespace {

// Writes what a stream gets to the log as well, or only to the log (screen null: stdout with -q).
// It keeps no buffer, so the log gets the lines of stdout and stderr in the order they are written.
class TeeBuffer : public std::streambuf {
  public:
    TeeBuffer(std::streambuf *screen, std::streambuf *log) : screen_(screen), log_(log) {}

  protected:
    int overflow(int c) override {
        if (traits_type::eq_int_type(c, traits_type::eof()))
            return traits_type::not_eof(c);
        if (screen_)
            screen_->sputc(traits_type::to_char_type(c));
        return log_->sputc(traits_type::to_char_type(c));
    }
    std::streamsize xsputn(const char *s, std::streamsize n) override {
        if (screen_)
            screen_->sputn(s, n);
        return log_->sputn(s, n);
    }
    int sync() override {
        const int screen = screen_ ? screen_->pubsync() : 0;
        return log_->pubsync() == 0 && screen == 0 ? 0 : -1;
    }

  private:
    std::streambuf *screen_, *log_;
};

// <prefix>.log: copies stdout and stderr to the log while it lives (stdout only to the log with -q).
// Opened before the run reads anything, as IQ-TREE and RAxML-NG open theirs: an unwritable prefix fails at once, and a
// failed run leaves its log with the error, no empty outputs.
class RunLog {
  public:
    RunLog(const std::string &path, bool quiet)
        : file_(path), out_(quiet ? nullptr : std::cout.rdbuf(), file_.rdbuf()),
          err_(std::cerr.rdbuf(), file_.rdbuf()) {
        if (!file_)
            throw std::runtime_error("cannot write " + path);
        cout_ = std::cout.rdbuf(&out_);
        cerr_ = std::cerr.rdbuf(&err_);
    }
    ~RunLog() {
        std::cout.flush();
        std::cout.rdbuf(cout_);
        std::cerr.rdbuf(cerr_);
    }
    RunLog(const RunLog &) = delete;
    RunLog &operator=(const RunLog &) = delete;

  private:
    std::ofstream file_;
    TeeBuffer out_, err_;
    std::streambuf *cout_ = nullptr, *cerr_ = nullptr;
};

// Stops the run if one of its outputs exists, a previous run's (-b and --supports write others), as IQ-TREE and
// RAxML-NG stop without -redo / --redo; --overwrite replaces them.
// The log does not count: a failed run leaves only its log.
void check_no_previous_run(const p2vbme::Report &run) {
    std::vector<std::string> outputs;
    if (!run.supports.empty()) {
        outputs = {".support.nwk", ".json"};
    } else {
        if (!run.bootstrap_only)
            outputs.push_back(".nwk");
        outputs.push_back(".json");
        if (run.bootstrap_last > 0) {
            if (!run.bootstrap_only)
                outputs.push_back(".support.nwk");
            outputs.push_back(".boottrees");
        }
    }
    for (const std::string &ext : outputs) {
        std::error_code ec;
        if (std::filesystem::exists(run.prefix + ext, ec))
            throw std::runtime_error(run.prefix + ext +
                                     " exists: --overwrite replaces a previous run's outputs, or choose another -p");
    }
}

// --supports: the supports of the branches of the tree -t over the replicate trees of a file, written as labels of the
// tree to <prefix>.support.nwk, with a report <prefix>.json.
// The tree is pruned to the replicates' leaves (the trees of -b have only the sequences searched), the
// supports are counted on that tree, and its branches carry them in the whole tree; the other branches get none.
void supports_of_saved_trees(p2vbme::Report &run) {
    const double t0 = omp_get_wtime();
    std::vector<std::string> names, replicate_names;
    const p2vbme::Tree tree = p2vbme::read_newick_with_names(run.tree_init, names);
    std::unique_ptr<p2vbme::SupportCounter> counter;
    std::vector<int32_t> node_map; // node of the pruned tree -> node of the tree
    const int32_t n_replicates =
        p2vbme::read_newick_trees(run.supports, replicate_names, [&](const p2vbme::Tree &replicate) {
            if (!counter) { // the first replicate gives the leaves: leaf k of the pruned tree is replicate_names[k]
                std::unordered_map<std::string, int32_t> leaf_of;
                for (int32_t x = 0; x < (int32_t)names.size(); ++x)
                    leaf_of.emplace(names[x], x);
                p2vbme::Deduplication keep;
                keep.copy_of.assign(names.size(), 0);
                for (const std::string &name : replicate_names) {
                    const auto it = leaf_of.find(name);
                    if (it == leaf_of.end())
                        throw std::runtime_error(run.supports + ": leaf '" + name + "' is not in " + run.tree_init);
                    keep.kept.push_back(it->second);
                }
                counter = std::make_unique<p2vbme::SupportCounter>(p2vbme::prune(tree, keep, &node_map));
            }
            counter->add(replicate);
        });
    if (n_replicates == 0)
        throw std::runtime_error(run.supports + ": no tree");
    const std::vector<double> tbe = counter->tbe(), exact = counter->exact();
    std::vector<std::string> labels(tree.c1.size());
    for (size_t u = 0; u < node_map.size(); ++u)
        if (tbe[u] != p2vbme::NO_SUPPORT)
            labels[node_map[u]] = p2vbme::format_support(tbe[u]) + "/" + p2vbme::format_support(exact[u]);
    // the root's two children are one branch of the unrooted tree: if one has no label, it takes the other's
    const int32_t N = tree.n_leaves(), root = 2 * N - 2, a = tree.c1[root], b = tree.c2[root];
    if (labels[a].empty() && a >= N)
        labels[a] = labels[b];
    if (labels[b].empty() && b >= N)
        labels[b] = labels[a];
    p2vbme::write_newick_unrooted(run.prefix + ".support.nwk", names, tree, run.neg_as_zero, labels);
    p2vbme::SupportsReport supports;
    supports.n_replicates = n_replicates;
    supports.n_leaves = N;
    supports.n_leaves_supported = (int32_t)replicate_names.size();
    supports.mean_tbe = counter->mean_tbe();
    supports.mean_exact = counter->mean_exact();
    run.time_total = omp_get_wtime() - t0;
    p2vbme::write_supports_report(run.prefix + ".json", run, supports);
    std::cout << "p2v-bme: supports of " << N << " leaves' branches over " << n_replicates << " trees (on their "
              << supports.n_leaves_supported << " leaves): mean TBE " << std::fixed << std::setprecision(2)
              << supports.mean_tbe << ", mean exact " << supports.mean_exact << "; wrote " << run.prefix
              << ".support.nwk, " << run.prefix << ".json" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    std::unique_ptr<RunLog> run_log; // lives until the end of main, so that the log also gets the error
    try {
        // Time: start of main
        const double t0 = omp_get_wtime();
        // Input: parse CLI args
        CommandLine args = parse_command_line(argc, argv);
        if (args.done)
            return args.exit_code;
        p2vbme::Settings &opt = args.settings;
        p2vbme::Report &run = args.report;
        // Output: stop if a previous run's outputs exist (before the log, which would replace that run's log)
        if (!args.overwrite)
            check_no_previous_run(run);
        // Output: open <prefix>.log first (an unwritable prefix fails here); the other outputs come at the end
        run_log = std::make_unique<RunLog>(run.prefix + ".log", args.quiet);
        if (!run.supports.empty()) {
            supports_of_saved_trees(run);
            return 0;
        }
        // Device: leave one core free (the CPU workers with --devices auto, the host threads of GPU runs), unless
        // OMP_NUM_THREADS sets the number of threads
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, omp_get_num_procs() - 1));

        const p2vbme::Accelerator accelerator = p2vbme::resolve_accelerator(opt.accelerator, opt.precision);
        const bool on_gpu = accelerator == p2vbme::Accelerator::Gpu;
        // Device: if auto and the CPU runs, say why (double precision, or why no GPU is visible).
        if (opt.accelerator == p2vbme::Accelerator::Auto && !on_gpu) {
            const std::string note = (opt.precision == p2vbme::Precision::Double ? std::string("double precision")
                                                                                 : p2vbme::no_gpu_reason()) +
                                     ": running on the CPU";
            run.warnings.push_back(note);
            std::cerr << "p2v-bme: warning: " << note << '\n';
        }
        parse_devices_option(args.devices, on_gpu, opt);

        const bool with_tree = !run.bootstrap_only, with_supports = run.bootstrap_last > 0 && with_tree;
        const std::string boottrees = run.prefix + ".boottrees";

        // Device: select the devices (this checks the device settings), before the read so that errors come early
        p2vbme::Devices devices = p2vbme::select_devices(accelerator, opt.precision, opt.devices, opt.threads);
        run.on_gpu = on_gpu;
        for (const auto &dev : devices)
            run.devices.push_back(dev->name());

        // Time: start of reading the alignment
        const double t_read = omp_get_wtime();
        p2vbme::Alignment aln = p2vbme::read_alignment(run.input_path);
        // Time: end of reading the alignment
        run.time_read = omp_get_wtime() - t_read;
        run.n_taxa = aln.n_taxa;
        run.n_sites = aln.n_sites;
        std::cout << "p2v-bme: read " << aln.n_taxa << " sequences x " << aln.n_sites << " sites in " << std::fixed
                  << std::setprecision(2) << run.time_read << " s" << std::endl;

        // Data: if identical sequences, two of each group stay in the search, the rest are put back after it.
        // F81, F84, TN93: base frequencies are counted on the whole input, before the copies are set aside
        //   so dedup does not change any distance (FastME and ape count them on the whole input too).
        // GTR+GC: the rates and alpha are fitted on the whole input too, for the same reason.
        const std::vector<std::string> all_names = aln.names;
        p2vbme::Deduplication dd;
        // Data: count base frequencies, fit GTR+GC and deduplicate
        // (the fit uses the run's CPU threads: -d N on the CPU, else all but one or OMP_NUM_THREADS)
        run.gtr_fit = p2vbme::prepare_alignment(aln, opt.substitution_model, run.gtr_fit_pairs, opt.seed, opt.threads,
                                                run.dedup, dd);
        if (opt.substitution_model.model == p2vbme::Model::GTR_GC) {
            const p2vbme::GtrFit &fit = run.gtr_fit;
            std::string note;
            if (fit.n_pairs == 0)
                note = "GTR+GC: no sequence pair shares " + std::to_string((int)p2vbme::GTR_FIT_MIN_SITES) +
                       " valid sites; the rates and alpha are not fitted (equal rates, alpha = 1)";
            else if (!fit.converged)
                note = "GTR+GC: the fit of the rates and alpha did not converge (" + std::to_string(fit.n_iterations) +
                       " iterations)";
            if (!note.empty()) {
                run.warnings.push_back(note);
                std::cerr << "p2v-bme: warning: " << note << '\n';
            }
            std::cout << "p2v-bme: GTR+GC fit on " << fit.n_pairs << " pairs in " << std::setprecision(2) << fit.time
                      << " s: alpha " << std::setprecision(4) << opt.substitution_model.alpha
                      << ", rates (AC, AG, AT, CG, CT, GT)" << std::setprecision(3);
            for (double r : opt.substitution_model.rates)
                std::cout << ' ' << r;
            std::cout << std::endl;
        }
        run.n_identical_removed = dd.n_removed();
        if (dd.n_removed() > 0)
            std::cout << "p2v-bme: " << dd.n_removed()
                      << " identical sequences set aside (two of each group kept): " << aln.n_taxa << " left"
                      << std::endl;

        p2vbme::Tree start_tree;
        if (run.tree_init == "sample") {
            // Tree: Sample-and-place start tree
            // 1. Pick n0 sequences at random
            // 2. Start t0 on the n0 sequences from a random (Yule-Harding) tree, or, if n0 > START_RANDOM_TREE_MAX
            //    (1000), from a sample-and-place tree of its own (on a sample of max(1000, n0 / 10))
            // 3. Optimise t0 with the minibatch search
            // 4. Assign every other sequence to its nearest neighbour in t0 (p-distance)
            // 5. Replace each sampled leaf by a small balanced subtree made of itself + its assigned sequences
            // (with fewer than START_SAMPLE_MIN_N sequences, infer() uses the random start instead and warns)
            opt.tree_init = p2vbme::TreeInit::Sample;
        } else if (run.tree_init == "random") {
            opt.tree_init = p2vbme::TreeInit::Random;
        } else {
            // Tree: predefined tree as a Newick
            start_tree = p2vbme::read_newick(run.tree_init, all_names);
            p2vbme::prepare_start_tree(start_tree, dd);
            opt.tree_init = p2vbme::TreeInit::Given;
        }

        const p2vbme::LogCallback log = [](const std::string &line) { std::cout << line << std::endl; };
        const p2vbme::LogCallback boot_log = [](const std::string &line) {
            std::cout << "p2v-bme: " << line << std::endl;
        };
        // Bootstrap: each replicate's tree, on the sequences searched, as one line of <prefix>.boottrees.
        // The file is emptied when the first replicate is saved, so a run that fails before leaves no empty file.
        bool boottrees_started = false;
        const p2vbme::ReplicateCallback save_tree = [&](int32_t, const p2vbme::Result &rep) {
            if (!boottrees_started && !std::ofstream(boottrees, std::ios::trunc))
                throw std::runtime_error("cannot write " + boottrees);
            boottrees_started = true;
            p2vbme::append_newick_unrooted(boottrees, aln.names, rep.tree);
        };
        p2vbme::Result res;
        if (run.bootstrap_only) {
            // Bootstrap: replicates A..B alone, as -b runs them (sample-and-place unless random, no branch lengths);
            // the report sums their rounds and times
            if (opt.tree_init == p2vbme::TreeInit::Given)
                run.tree_init = "sample";
            opt.branch_lengths = false;
            const double t_boot = omp_get_wtime();
            p2vbme::bootstrap_replicates(
                aln, opt, devices, run.bootstrap_first, run.bootstrap_last,
                [&](int32_t r, const p2vbme::Result &rep) {
                    save_tree(r, rep);
                    res.time_start += rep.time_start;
                    res.time_search += rep.time_search;
                    res.rounds_run += rep.rounds_run;
                    res.stop_reason = rep.stop_reason;
                    res.tree_init = rep.tree_init;
                    res.sample_size_used = rep.sample_size_used;
                    for (const std::string &w : rep.warnings)
                        if (std::find(res.warnings.begin(), res.warnings.end(), w) == res.warnings.end())
                            res.warnings.push_back(w);
                },
                boot_log);
            run.time_bootstrap = omp_get_wtime() - t_boot;
        } else {
            // Main: inference search on the devices (the alignment is loaded onto them); infer() checks the settings
            res = p2vbme::infer(aln, opt, devices, &start_tree, log);
        }
        if (res.tree_init == p2vbme::TreeInit::Random)
            run.tree_init = "random";
        for (const std::string &w : res.warnings) {
            run.warnings.push_back(w);
            std::cerr << "p2v-bme: warning: " << w << '\n';
        }
        // Bootstrap: replicates of the searched alignment's columns, each searched as the main tree with the model's
        // parameters as fitted, and the supports of the main tree's branches over them
        p2vbme::BootstrapResult boot;
        if (with_supports) {
            boot = p2vbme::bootstrap(aln, opt, devices, res.tree, run.bootstrap_last, boot_log, save_tree);
            run.mean_tbe = boot.mean_tbe;
            run.mean_exact = boot.mean_exact;
            run.time_bootstrap = boot.time;
        }
        // Device: record the largest memory in use on the devices (GPU-wide, other processes included)
        for (const auto &dev : devices)
            run.gpu_memory_in_use_at_end_mib = std::max(run.gpu_memory_in_use_at_end_mib, dev->memory_in_use_mb());
        // Data: put back the identical sequences set aside, with zero-length branches (the replicates' trees stay on
        // the sequences searched, as -b counts the supports)
        std::vector<int32_t> node_map;
        if (with_tree) {
            p2vbme::restore_copies(res.tree, dd, &node_map);
            // Output: write the unrooted tree, and with -b the same tree with the supports
            p2vbme::write_newick_unrooted(run.prefix + ".nwk", all_names, res.tree, run.neg_as_zero);
        }
        if (with_supports)
            p2vbme::write_newick_unrooted(run.prefix + ".support.nwk", all_names, res.tree, run.neg_as_zero,
                                          p2vbme::support_labels(boot, node_map, res.tree));
        // Time: end of main
        run.time_total = omp_get_wtime() - t0;
        // Output: write the report
        p2vbme::write_report(run.prefix + ".json", run, opt, res);
        if (with_tree)
            std::cout << "p2v-bme: " << run.tree_init << " start, " << res.rounds_run << " rounds, BME length "
                      << std::fixed << std::setprecision(6) << res.bme_length << "; " << std::setprecision(2)
                      << run.time_total << " s in total; wrote " << run.prefix << ".nwk, "
                      << (with_supports ? run.prefix + ".support.nwk, " + boottrees + ", " : "") << run.prefix
                      << ".json" << std::endl;
        else
            std::cout << "p2v-bme: bootstrap replicates " << run.bootstrap << ", " << res.rounds_run << " rounds; "
                      << std::fixed << std::setprecision(2) << run.time_total << " s in total; wrote " << boottrees
                      << ", " << run.prefix << ".json" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "p2v-bme: error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
