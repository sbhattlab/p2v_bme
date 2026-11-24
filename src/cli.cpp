/*
 * cli.cpp
 *
 * The command line of p2v-bme, via cxxopts.
 *
 * Note: most options follow IQ-TREE names where the tools overlap
 *       Device and precision options follow lightning names.
 */
#include <cstdint> // cxxopts.hpp 3.2.1 uses uint8_t without including <cstdint>, which GCC 15 needs
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "accelerator.hpp"
#include "cli.hpp"
#include "version.hpp"

namespace {
// --devices: "auto" (the first GPU), N (the first N), or a comma list of GPU numbers ("0,2"; "1," for GPU 1 alone).
std::vector<int> parse_gpu_devices(const std::string &d) {
    std::vector<int> ids;
    if (d == "auto")
        return {0};
    const bool list = d.find(',') != std::string::npos;
    size_t pos = 0;
    while (pos < d.size()) {
        size_t end = d.find(',', pos);
        if (end == std::string::npos)
            end = d.size();
        const std::string item = d.substr(pos, end - pos);
        if (!item.empty()) {
            size_t used = 0;
            int v = -1;
            try {
                v = std::stoi(item, &used);
            } catch (const std::exception &) {
                used = 0;
            }
            if (used != item.size() || v < 0)
                throw std::runtime_error("unknown value for devices: '" + d +
                                         "' (auto, a number, or a comma list of GPU numbers)");
            ids.push_back(v);
        }
        pos = end + 1;
    }
    if (ids.empty())
        throw std::runtime_error("unknown value for devices: '" + d + "' (no GPU number given)");
    if (!list) { // a single number N: the first N GPUs
        const int n = ids[0];
        if (n < 1)
            throw std::runtime_error("unknown value for devices: '" + d + "' (at least one GPU)");
        ids.clear();
        for (int g = 0; g < n; ++g)
            ids.push_back(g);
    }
    return ids;
}

// --devices on the CPU: auto (0: the OpenMP default, all cores but one; see main) or a number of threads.
int32_t parse_cpu_threads(const std::string &d) {
    if (d == "auto")
        return 0;
    bool digits = !d.empty() && d.size() <= 9;
    for (char c : d)
        digits = digits && c >= '0' && c <= '9';
    const int v = digits ? std::stoi(d) : 0;
    if (v < 1)
        throw std::runtime_error("unknown value for devices on the CPU: '" + d + "' (auto or a number of threads)");
    return v;
}

// Reads -b into the report: B (replicates 1..B after the tree; 0 = none) or A-B (replicates A..B alone).
// Throws std::runtime_error for anything else, or A-B without 1 <= A <= B.
void parse_bootstrap_option(p2vbme::Report &run) {
    const std::string &b = run.bootstrap;
    auto number = [&](const std::string &s) -> int32_t {
        if (s.empty() || s.size() > 9 || s.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("-b takes B or A-B (replicates A to B), e.g. -b 100 or -b 26-50, not '" + b + "'");
        return std::stoi(s);
    };
    const size_t dash = b.find('-');
    if (dash == std::string::npos) {
        run.bootstrap_last = number(b);
        run.bootstrap_first = run.bootstrap_last > 0 ? 1 : 0;
        return;
    }
    run.bootstrap_first = number(b.substr(0, dash));
    run.bootstrap_last = number(b.substr(dash + 1));
    run.bootstrap_only = true;
    if (run.bootstrap_first < 1 || run.bootstrap_last < run.bootstrap_first)
        throw std::runtime_error("-b A-B needs 1 <= A <= B, not '" + b + "'");
}

} // namespace

CommandLine parse_command_line(int argc, char **argv) {
    CommandLine result;
    p2vbme::Settings &opt = result.settings; // the numeric options are parsed into it; not given: its defaults
    cxxopts::Options cli("p2v-bme", "Balanced minimum evolution tree from an aligned DNA alignment (FASTA), by the "
                                    "Phylo2Vec minibatch search.\nWrites <prefix>.nwk (unrooted tree), "
                                    "<prefix>.json (run report) and <prefix>.log (what it prints).\n");
    cli.custom_help("-s ALIGNMENT -m MODEL [options]");
    // clang-format off
    cli.add_options()
        ("s,alignment", "aligned DNA sequences: FASTA, or .maple (by the file extension)", cxxopts::value<std::string>(),
         "FILE")
        ("m,model", "substitution model of the distances: JC69, K80, F81, F84, TN93, LogDet or GTR+GC (its rates "
         "and alpha fitted to the alignment)",
         cxxopts::value<std::string>(), "MODEL")
        ("p,prefix", "output prefix: <prefix>.nwk, <prefix>.json, <prefix>.log (default: the alignment's file name)",
         cxxopts::value<std::string>(), "STR")
        ("a,accelerator", "cpu, gpu or auto (auto: the GPU if one is found, else the CPU)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("d,devices", "GPU: auto (the first GPU), N (the first N) or a list such as 0,2; CPU: auto (all "
         "cores but one, or OMP_NUM_THREADS if set) or N threads", cxxopts::value<std::string>()->default_value("auto"))
        ("b,bootstrap", "the standard bootstrap: B replicates after the tree, whose branches get their supports "
         "(TBE/exact, in %) in <prefix>.support.nwk; or A-B: replicates A to B alone, as -b B with the same --seed "
         "runs them (to spread B over jobs). The replicates' trees go to <prefix>.boottrees (0 = none)",
         cxxopts::value(result.report.bootstrap), "B|A-B")
        ("seed", "seed of the start tree, of the GTR+GC fit's pairs and of the bootstrap", cxxopts::value(opt.seed),
         "N")
        ("q,quiet", "print no progress (it still goes to <prefix>.log; warnings and errors still go to stderr)")
        ("v,version", "print the version")
        ("h,help", "print the core options")
        ("help-all", "print all options");
    cli.add_options("Substitution model")
        ("gtr-fit-pairs", "GTR+GC: sequence pairs drawn to fit the rates and alpha (all pairs, once each, if there "
         "are fewer)", cxxopts::value(result.report.gtr_fit_pairs), "N");
    cli.add_options("Search")
        ("max-rounds", "rounds of the search; 0 = no search (score the start tree)",
         cxxopts::value(opt.max_rounds), "N")
        ("tol", "stop once a round gains at most TOL x the total gain; 0 = only when a round changes nothing",
         cxxopts::value(opt.tol), "TOL")
        ("time-limit", "seconds after which no new round starts (0 = none)",
         cxxopts::value(opt.time_limit), "S")
        ("representative-cap", "representatives kept per clade in a window (1-256)",
         cxxopts::value(opt.representative_cap), "N")
        ("precision", "single or double; double runs on the CPU only (the GPU computes in single precision)",
         cxxopts::value<std::string>()->default_value("single"));
    cli.add_options("Start tree")
        ("t,tree", "start tree: sample (sample-and-place), random, or a Newick file",
         cxxopts::value<std::string>()->default_value("sample"), "TREE")
        ("sample-size", "sequences sampled for sample-and-place (0 = min(100000, N/10))",
         cxxopts::value(opt.sample_size), "N");
    cli.add_options("Branch lengths and output")
        ("no-branch-lengths", "do not estimate BME branch lengths")
        ("no-neg-as-zero", "write negative branch lengths as they are (default: as 0)")
        ("report-full", "add the details to <prefix>.json: input, settings as used, devices, time per stage, "
         "memory, gain of each round")
        ("overwrite", "replace a previous run's outputs (without it, a run whose outputs exist stops before it "
         "starts)");
    cli.add_options("Bootstrap")
        ("supports", "with -t TREE and no alignment: the supports (TBE/exact, in %) of TREE's branches over the "
         "trees in FILE (one Newick per line, e.g. the .boottrees of -b; TREE is pruned to their leaves) go to "
         "<prefix>.support.nwk (default prefix: TREE)", cxxopts::value<std::string>(), "FILE");
    cli.add_options("Input")
        ("no-dedup", "keep all identical sequences (default: two of each group are kept for the search and the "
         "others put back next to them, with zero-length branches)");
    // clang-format on

    auto done = [&](int exit_code) {
        result.done = true;
        result.exit_code = exit_code;
        return result;
    };
    const auto args = cli.parse(argc, argv);
    if (args.count("help")) {
        std::cout << cli.help({""}) << "\nAll options: --help-all\n";
        return done(0);
    }
    if (args.count("help-all")) {
        std::cout << cli.help({"", "Substitution model", "Search", "Start tree", "Branch lengths and output",
                               "Bootstrap", "Input"})
                  << '\n';
        return done(0);
    }
    if (args.count("version")) {
        std::cout << "p2v-bme " << p2vbme::version_string() << '\n';
        return done(0);
    }
    if (!args.unmatched().empty())
        throw std::runtime_error("unexpected argument '" + args.unmatched()[0] + "' (see --help)");
    if (args.count("supports")) { // trees only: no alignment, no search
        if (!args.count("tree") || args.count("alignment") || args.count("model") || args.count("bootstrap")) {
            std::cerr << "p2v-bme: --supports needs -t TREE, and no -s, -m or -b (see --help-all)\n";
            return done(2);
        }
        p2vbme::Report &run = result.report;
        for (int a = 0; a < argc; ++a)
            run.command_line.push_back(argv[a]);
        result.quiet = args.count("quiet") > 0;
        result.overwrite = args.count("overwrite") > 0;
        run.supports = args["supports"].as<std::string>();
        run.tree_init = args["tree"].as<std::string>();
        run.prefix = args.count("prefix") ? args["prefix"].as<std::string>() : run.tree_init;
        run.neg_as_zero = !args.count("no-neg-as-zero");
        return result;
    }
    if (!args.count("alignment") || !args.count("model")) {
        std::cerr << "p2v-bme: -s ALIGNMENT and -m MODEL are required (see --help)\n";
        return done(2);
    }
    result.quiet = args.count("quiet") > 0;
    result.overwrite = args.count("overwrite") > 0;
    p2vbme::Report &run = result.report;
    for (int a = 0; a < argc; ++a)
        run.command_line.push_back(argv[a]);
    run.input_path = args["alignment"].as<std::string>();
    const std::string model_name = args["model"].as<std::string>();
    const p2vbme::Model model = p2vbme::parse_model(model_name);
    run.prefix = args.count("prefix") ? args["prefix"].as<std::string>() : run.input_path;
    run.neg_as_zero = !args.count("no-neg-as-zero");

    opt.substitution_model.model = model;
    opt.branch_lengths = !args.count("no-branch-lengths");
    opt.accelerator = p2vbme::parse_accelerator(args["accelerator"].as<std::string>());
    opt.precision = p2vbme::parse_precision(args["precision"].as<std::string>());
    result.devices = args["devices"].as<std::string>();
    run.tree_init = args["tree"].as<std::string>();
    run.dedup = !args.count("no-dedup");
    parse_bootstrap_option(run);
    run.full = args.count("report-full") > 0;
    return result;
}

void parse_devices_option(const std::string &devices, bool on_gpu, p2vbme::Settings &opt) {
    if (on_gpu)
        opt.devices = parse_gpu_devices(devices);
    else
        opt.threads = parse_cpu_threads(devices);
}
