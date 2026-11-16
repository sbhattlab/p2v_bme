/*
 * main.cpp
 *
 * p2v-bme: a balanced minimum evolution tree from an aligned DNA alignment (Phylo2Vec minibatch search). Writes
 * the unrooted tree to <prefix>.nwk and the run report to <prefix>.json. The run, step by step: the arguments
 * (cli), the devices (device), the alignment (io), before inference (processing), inference, after inference,
 * the tree (newick) and the report.
 * Quiet by default: progress with -v; warnings and errors always on stderr.
 */
#include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "io.hpp"
#include "newick.hpp"
#include "processing.hpp"
#include "report.hpp"

int main(int argc, char **argv) {
    try {
        const double t0 = omp_get_wtime();
        CommandLine args = parse_command_line(argc, argv);
        if (args.done)
            return args.exit_code;
        p2vbme::Settings &opt = args.settings;
        const RunOptions &options = args.options;
        const bool verbose = options.verbose;

        p2vbme::RunInfo run;
        for (int a = 0; a < argc; ++a)
            run.command_line.push_back(argv[a]);
        run.input_path = options.input_path;
        run.prefix = options.prefix;
        run.neg_as_zero = options.neg_as_zero;

        /* auto: the GPU if one is found (and single precision), else the CPU, with a message */
        const p2vbme::DeviceChoice choice = p2vbme::choose_devices(opt.accelerator, opt.precision);
        if (!choice.note.empty()) {
            run.warnings.push_back(choice.note);
            fprintf(stderr, "p2v-bme: %s\n", run.warnings.back().c_str());
        }
        parse_devices_option(options.devices, choice.on_gpu, opt);
        p2vbme::check_settings(opt);
        p2vbme::check_devices(choice, opt.precision);
        /* fail now, not after the search, if the outputs cannot be written */
        for (const char *ext : {".nwk", ".json"}) {
            FILE *f = fopen((run.prefix + ext).c_str(), "a");
            if (!f)
                throw std::runtime_error("cannot write " + run.prefix + ext);
            fclose(f);
        }

        const double t_read = omp_get_wtime();
        p2vbme::Alignment aln = p2vbme::read_alignment(run.input_path);
        run.time_read = omp_get_wtime() - t_read;
        run.n_taxa = aln.n_taxa;
        run.n_sites = aln.n_sites;
        if (verbose)
            fprintf(stderr, "p2v-bme: read %d sequences x %d sites in %.2f s\n", aln.n_taxa, aln.n_sites,
                    run.time_read);

        /* identical sequences: two of each group stay in the search, the rest are put back; F81, F84, TN93: the
         * base frequencies of the whole input, as FastME and ape count them */
        const std::vector<std::string> all_names = aln.names;
        p2vbme::Deduplication dd;
        run.dedup = options.dedup;
        p2vbme::prepare_alignment(aln, opt.substitution, run.dedup, dd);
        run.n_identical_removed = dd.n_removed();
        if (verbose && dd.n_removed() > 0)
            fprintf(stderr, "p2v-bme: %d identical sequences set aside (two of each group kept): %d left\n",
                    dd.n_removed(), aln.n_taxa);
        if (aln.n_taxa < 4)
            throw std::runtime_error("need at least 4 sequences" +
                                     std::string(run.dedup ? " after setting identical copies aside" : "") + ", have " +
                                     std::to_string(aln.n_taxa) + (run.dedup ? " (--no-dedup keeps them)" : ""));

        run.start_tree = options.start_tree;
        p2vbme::Tree start_tree;
        if (run.start_tree == "sample") {
            opt.start = p2vbme::Start::Sample;
            if (aln.n_taxa < p2vbme::START_SAMPLE_MIN_N) {
                run.warnings.push_back("fewer than " + std::to_string(p2vbme::START_SAMPLE_MIN_N) +
                                       " sequences, using the random start instead of sample-and-place");
                fprintf(stderr, "p2v-bme: warning: %s\n", run.warnings.back().c_str());
                opt.start = p2vbme::Start::Random;
                run.start_tree = "random";
            }
        } else if (run.start_tree == "random") {
            opt.start = p2vbme::Start::Random;
        } else {
            start_tree = p2vbme::read_newick(run.start_tree, all_names);
            p2vbme::prepare_start_tree(start_tree, dd);
            opt.start = p2vbme::Start::Given;
        }

        p2vbme::LogCallback log;
        if (verbose)
            log = [](const std::string &line) {
                fprintf(stderr, "%s\n", line.c_str());
                fflush(stderr);
            };
        p2vbme::Devices devices = p2vbme::select_devices(choice, opt.precision, opt.devices, opt.threads);
        run.on_gpu = choice.on_gpu;
        for (const auto &dev : devices)
            run.devices.push_back(dev->name());
        if (run.on_gpu)
            run.gpu_build_info = p2vbme::gpu_build_info();
        p2vbme::Result res = p2vbme::infer(aln, opt, devices, &start_tree, log);
        for (const auto &dev : devices)
            run.gpu_memory_in_use_at_end_mb = std::max(run.gpu_memory_in_use_at_end_mb, dev->memory_in_use_mb());
        p2vbme::restore_copies(res.tree, dd);

        p2vbme::write_newick_unrooted(run.prefix + ".nwk", all_names, res.tree, run.neg_as_zero);
        run.time_total = omp_get_wtime() - t0;
        p2vbme::write_report(run.prefix + ".json", run, opt, res, options.report_full);
        if (verbose)
            fprintf(stderr, "p2v-bme: %s start, %d rounds, BME length %.6f; %.2f s in total; wrote %s.nwk, %s.json\n",
                    run.start_tree.c_str(), res.rounds_run, res.bme_length, run.time_total, run.prefix.c_str(),
                    run.prefix.c_str());
    } catch (const std::exception &e) {
        fprintf(stderr, "p2v-bme: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
