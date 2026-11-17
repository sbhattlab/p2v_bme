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

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
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
        // Time: start of main
        const double t0 = omp_get_wtime();
        // Input: parse CLI args
        CommandLine args = parse_command_line(argc, argv);
        if (args.done)
            return args.exit_code;
        p2vbme::Settings &opt = args.settings;
        p2vbme::Report &run = args.report;
        const bool verbose = args.verbose;
        // Device: leave one core free (the CPU workers with --devices auto, the host threads of GPU runs), unless
        // OMP_NUM_THREADS sets the number of threads
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, omp_get_num_procs() - 1));

        const p2vbme::DeviceChoice choice = p2vbme::choose_devices(opt.accelerator, opt.precision);
        // Device: if auto and no GPU found, indicate what precision is used for CPU.
        if (!choice.note.empty()) {
            run.warnings.push_back(choice.note);
            std::cerr << "p2v-bme: warning: " << choice.note << '\n';
        }
        parse_devices_option(args.devices, choice.on_gpu, opt);
        p2vbme::check_settings(opt);
        p2vbme::check_devices(choice, opt.precision);

        // Output: fail now if the outputs cannot be written
        for (const char *ext : {".nwk", ".json"}) {
            std::ofstream f(run.prefix + ext, std::ios::app);
            if (!f)
                throw std::runtime_error("cannot write " + run.prefix + ext);
        }

        // Time: start of reading the alignment
        const double t_read = omp_get_wtime();
        p2vbme::Alignment aln = p2vbme::read_alignment(run.input_path);
        // Time: end of reading the alignment
        run.time_read = omp_get_wtime() - t_read;
        run.n_taxa = aln.n_taxa;
        run.n_sites = aln.n_sites;
        if (verbose)
            std::cerr << "p2v-bme: read " << aln.n_taxa << " sequences x " << aln.n_sites << " sites in " << std::fixed
                      << std::setprecision(2) << run.time_read << " s\n";

        // Data: if identical sequences, two of each group stay in the search, the rest are put back after it.
        // F81, F84, TN93: base frequencies are counted on the whole input, before the copies are set aside
        //   so dedup does not change any distance (FastME and ape count them on the whole input too).
        const std::vector<std::string> all_names = aln.names;
        p2vbme::Deduplication dd;
        // Data: count base frequencies and deduplicate
        p2vbme::prepare_alignment(aln, opt.substitution, run.dedup, dd);
        run.n_identical_removed = dd.n_removed();
        if (verbose && dd.n_removed() > 0)
            std::cerr << "p2v-bme: " << dd.n_removed()
                      << " identical sequences set aside (two of each group kept): " << aln.n_taxa << " left\n";
        if (aln.n_taxa < 4)
            throw std::runtime_error("need at least 4 sequences" +
                                     std::string(run.dedup ? " after setting identical copies aside" : "") + ", have " +
                                     std::to_string(aln.n_taxa) + (run.dedup ? " (--no-dedup keeps them)" : ""));

        p2vbme::Tree start_tree;
        if (run.tree_init == "sample") {
            // Tree: Sample-and-place start tree
            // 1. Pick n0 sequences at random
            // 2. Start t0 on the n0 sequences from a random (Yule-Harding) tree, or, if n0 > START_RANDOM_TREE_MAX
            //    (1000), from a sample-and-place tree of its own (on a sample of max(1000, n0 / 10))
            // 3. Optimise t0 with the minibatch search
            // 4. Assign every other sequence to its nearest neighbour in t0 (p-distance)
            // 5. Replace each sampled leaf by a small balanced subtree made of itself + its assigned sequences
            opt.tree_init = p2vbme::TreeInit::Sample;
            // Tree: if too few sequences, use a random start
            if (aln.n_taxa < p2vbme::START_SAMPLE_MIN_N) {
                run.warnings.push_back("fewer than " + std::to_string(p2vbme::START_SAMPLE_MIN_N) +
                                       " sequences, using the random start instead of sample-and-place");
                std::cerr << "p2v-bme: warning: " << run.warnings.back() << '\n';
                opt.tree_init = p2vbme::TreeInit::Random;
                run.tree_init = "random";
            }
        } else if (run.tree_init == "random") {
            opt.tree_init = p2vbme::TreeInit::Random;
        } else {
            // Tree: predefined tree as a Newick
            start_tree = p2vbme::read_newick(run.tree_init, all_names);
            p2vbme::prepare_start_tree(start_tree, dd);
            opt.tree_init = p2vbme::TreeInit::Given;
        }

        p2vbme::LogCallback log;
        if (verbose)
            log = [](const std::string &line) { std::cerr << line << std::endl; };
        // Device: select the devices
        p2vbme::Devices devices = p2vbme::select_devices(choice, opt.precision, opt.devices, opt.threads);
        run.on_gpu = choice.on_gpu;
        for (const auto &dev : devices)
            run.devices.push_back(dev->name());
        if (run.on_gpu)
            run.gpu_build_info = p2vbme::gpu_build_info();
        // Main: inference search on the devices (the alignment is loaded onto them)
        p2vbme::Result res = p2vbme::infer(aln, opt, devices, &start_tree, log);
        // Device: record the largest memory in use on the devices (GPU-wide, other processes included)
        for (const auto &dev : devices)
            run.gpu_memory_in_use_at_end_mb = std::max(run.gpu_memory_in_use_at_end_mb, dev->memory_in_use_mb());
        // Data: put back the identical sequences set aside, with zero-length branches
        p2vbme::restore_copies(res.tree, dd);

        // Output: write the unrooted tree
        p2vbme::write_newick_unrooted(run.prefix + ".nwk", all_names, res.tree, run.neg_as_zero);
        // Time: end of main
        run.time_total = omp_get_wtime() - t0;
        // Output: write the report
        p2vbme::write_report(run.prefix + ".json", run, opt, res);
        if (verbose)
            std::cerr << "p2v-bme: " << run.tree_init << " start, " << res.rounds_run << " rounds, BME length "
                      << std::fixed << std::setprecision(6) << res.bme_length << "; " << std::setprecision(2)
                      << run.time_total << " s in total; wrote " << run.prefix << ".nwk, " << run.prefix << ".json\n";
    } catch (const std::exception &e) {
        std::cerr << "p2v-bme: error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
