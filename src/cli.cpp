/*
 * cli.cpp
 *
 * The command line of p2v-bme, via cxxopts.
 *
 * Note: most options follow IQ-TREE names where the tools overlap
 *       Device and precision options follow lightning names.
 */
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

} // namespace

CommandLine parse_command_line(int argc, char **argv) {
    CommandLine result;
    p2vbme::Settings &opt = result.settings; // the numeric options are parsed into it; not given: its defaults
    cxxopts::Options cli("p2v-bme", "Balanced minimum evolution tree from an aligned DNA alignment (FASTA), by the "
                                    "Phylo2Vec minibatch search.\nWrites <prefix>.nwk (unrooted tree) and "
                                    "<prefix>.json (run report).\n");
    cli.custom_help("-s ALIGNMENT -m MODEL [options]");
    // clang-format off
    cli.add_options()
        ("s,alignment", "aligned DNA sequences: FASTA, or .maple (by the file extension)", cxxopts::value<std::string>(),
         "FILE")
        ("m,model", "substitution model of the distances: JC69, K80, F81, F84, TN93 or LogDet",
         cxxopts::value<std::string>(), "MODEL")
        ("p,prefix", "output prefix: <prefix>.nwk, <prefix>.json (default: the alignment's file name)",
         cxxopts::value<std::string>(), "STR")
        ("a,accelerator", "cpu, gpu or auto (auto: the GPU if one is found, else the CPU)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("d,devices", "GPU: auto (the first GPU), N (the first N) or a list such as 0,2; CPU: auto (all "
         "cores but one, or OMP_NUM_THREADS if set) or N threads", cxxopts::value<std::string>()->default_value("auto"))
        ("seed", "seed of the start tree", cxxopts::value(opt.seed), "N")
        ("q,quiet", "print no progress (warnings and errors still go to stderr)")
        ("v,version", "print the version")
        ("h,help", "print the core options")
        ("help-all", "print all options");
    cli.add_options("Search")
        ("max-rounds", "rounds of the search; 0 = no search (score the start tree)",
         cxxopts::value(opt.max_rounds), "N")
        ("early-stopping", "stop once a round gains at most TOL x the total gain; 0 = only when a round changes "
         "nothing", cxxopts::value(opt.early_stopping), "TOL")
        ("time-limit", "seconds after which no new round starts (0 = none)",
         cxxopts::value(opt.time_limit), "S")
        ("profile-cap", "leaves kept per clade profile in a window (1-256)",
         cxxopts::value(opt.profile_cap), "N")
        ("exact-check-eta", "re-check windows whose capped gain is above -ETA on complete profiles (0 = never)",
         cxxopts::value(opt.exact_check_eta), "ETA")
        ("precision", "single or double; double runs on the CPU only (the GPU computes in single precision)",
         cxxopts::value<std::string>()->default_value("single"));
    cli.add_options("Start tree")
        ("t,tree", "start tree: sample (sample-and-place), random, or a Newick file",
         cxxopts::value<std::string>()->default_value("sample"), "TREE")
        ("sample-size", "sequences sampled for sample-and-place (0 = min(100000, N/10))",
         cxxopts::value(opt.sample_size), "N");
    cli.add_options("Branch lengths and output")
        ("no-branch-lengths", "do not estimate BME branch lengths")
        ("bl-profile-cap", "leaves kept per subtree profile for branch lengths (1-256)",
         cxxopts::value(opt.bl_profile_cap), "N")
        ("no-neg-as-zero", "write negative branch lengths as they are (default: as 0)")
        ("report-full", "add the details to <prefix>.json: host, input, settings as used, devices, time per stage, "
         "memory, model parameters, gain of each round");
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
        std::cout << cli.help({"", "Search", "Start tree", "Branch lengths and output", "Input"}) << '\n';
        return done(0);
    }
    if (args.count("version")) {
        std::cout << "p2v-bme " << p2vbme::version_string() << '\n';
        return done(0);
    }
    if (!args.unmatched().empty())
        throw std::runtime_error("unexpected argument '" + args.unmatched()[0] + "' (see --help)");
    if (!args.count("alignment") || !args.count("model")) {
        std::cerr << "p2v-bme: -s ALIGNMENT and -m MODEL are required (see --help)\n";
        return done(2);
    }
    result.quiet = args.count("quiet") > 0;
    p2vbme::Report &run = result.report;
    for (int a = 0; a < argc; ++a)
        run.command_line.push_back(argv[a]);
    run.input_path = args["alignment"].as<std::string>();
    const std::string model_name = args["model"].as<std::string>();
    const p2vbme::Model model = p2vbme::parse_model(model_name);
    if (model == p2vbme::Model::GTR_GC) // it needs its parameters, and their fit comes later
        throw std::runtime_error("model '" + model_name +
                                 "' is not available in this version (models: JC69, K80, F81, F84, TN93, LogDet)");
    run.prefix = args.count("prefix") ? args["prefix"].as<std::string>() : run.input_path;
    run.neg_as_zero = !args.count("no-neg-as-zero");

    opt.substitution_model.model = model;
    opt.branch_lengths = !args.count("no-branch-lengths");
    opt.accelerator = p2vbme::parse_accelerator(args["accelerator"].as<std::string>());
    opt.precision = p2vbme::parse_precision(args["precision"].as<std::string>());
    result.devices = args["devices"].as<std::string>();
    run.tree_init = args["tree"].as<std::string>();
    run.dedup = !args.count("no-dedup");
    run.full = args.count("report-full") > 0;
    return result;
}

void parse_devices_option(const std::string &devices, bool on_gpu, p2vbme::Settings &opt) {
    if (on_gpu)
        opt.devices = parse_gpu_devices(devices);
    else
        opt.threads = parse_cpu_threads(devices);
}
