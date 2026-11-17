/*
 * cli.cpp
 *
 * The command line of p2v-bme (cxxopts): the options and their help, and the arguments as Settings and run
 * options. Options follow IQ-TREE where the tools overlap (-s, -m, --prefix, --seed, -v, -t); --accelerator and
 * --devices follow PyTorch Lightning.
 */
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "cli.hpp"
#include "version.hpp"

namespace {

// Default value of an option, as text for cxxopts (from config.hpp).
template <class T> std::string text(T value) {
    std::ostringstream s;
    s << value;
    return s.str();
}

// -m (IQ-TREE style names, case-insensitive). GTR+GC needs the parameter fit, which comes later.
p2vbme::Model parse_model(const std::string &name) {
    const p2vbme::Model m = p2vbme::parse_model(name);
    if (m == p2vbme::Model::GTR_GC)
        throw std::runtime_error("model '" + name +
                                 "' is not available in this version (models: JC69, K80, F81, F84, TN93, LogDet)");
    return m;
}

p2vbme::Accelerator parse_accelerator(const std::string &a) {
    if (a == "auto")
        return p2vbme::Accelerator::Auto;
    if (a == "gpu")
        return p2vbme::Accelerator::Gpu;
    if (a == "cpu")
        return p2vbme::Accelerator::Cpu;
    throw std::runtime_error("--accelerator must be cpu, gpu or auto, not '" + a + "'");
}

// --devices: "auto" (all), N (the first N), or a comma list of GPU numbers ("0,2"; "1," for GPU 1 alone).
std::vector<int> parse_devices(const std::string &d) {
    std::vector<int> ids;
    if (d == "auto")
        return ids;
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
                throw std::runtime_error("--devices must be auto, a number or a comma list of GPU numbers, not '" + d +
                                         "'");
            ids.push_back(v);
        }
        pos = end + 1;
    }
    if (ids.empty())
        throw std::runtime_error("--devices: no device given in '" + d + "'");
    if (!list) { // a single number N: the first N GPUs
        const int n = ids[0];
        if (n < 1)
            throw std::runtime_error("--devices: need at least one device");
        ids.clear();
        for (int g = 0; g < n; ++g)
            ids.push_back(g);
    }
    return ids;
}

// --devices on the CPU: auto (0: all cores) or a number of threads.
int32_t parse_threads(const std::string &d) {
    if (d == "auto")
        return 0;
    bool digits = !d.empty() && d.size() <= 9;
    for (char c : d)
        digits = digits && c >= '0' && c <= '9';
    const int v = digits ? std::stoi(d) : 0;
    if (v < 1)
        throw std::runtime_error("--devices on the CPU must be auto or a number of threads, not '" + d + "'");
    return v;
}

p2vbme::Precision parse_precision(const std::string &p) {
    if (p == "single")
        return p2vbme::Precision::Single;
    if (p == "double")
        return p2vbme::Precision::Double;
    throw std::runtime_error("--precision must be single or double, not '" + p + "'");
}

} // namespace

CommandLine parse_command_line(int argc, char **argv) {
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
        ("prefix", "output prefix: <prefix>.nwk, <prefix>.json (default: the alignment's file name)",
         cxxopts::value<std::string>(), "STR")
        ("accelerator", "cpu, gpu or auto (auto: the GPU if one is found, else the CPU)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("devices", "GPU: auto (all visible, at most 8), N (the first N) or a list such as 0,2; CPU: auto (all "
         "cores) or N threads", cxxopts::value<std::string>()->default_value("auto"))
        ("seed", "seed of the start tree", cxxopts::value<uint64_t>()->default_value(text(p2vbme::DEFAULT_SEED)), "N")
        ("v,verbose", "print progress on stderr")
        ("version", "print the version")
        ("h,help", "print the core options")
        ("help-all", "print all options");
    cli.add_options("Search")
        ("max-rounds", "rounds of the search; 0 = no search (score the start tree)",
         cxxopts::value<int>()->default_value(text(p2vbme::DEFAULT_MAX_ROUNDS)), "N")
        ("early-stopping", "stop once a round gains at most TOL x the total gain; 0 = only when a round changes "
         "nothing", cxxopts::value<double>()->default_value(text(p2vbme::DEFAULT_EARLY_STOPPING)), "TOL")
        ("time-limit", "seconds after which no new round starts (0 = none)",
         cxxopts::value<double>()->default_value(text(p2vbme::DEFAULT_TIME_LIMIT)), "S")
        ("profile-cap", "leaves kept per clade profile in a window (1-256)",
         cxxopts::value<int>()->default_value(text(p2vbme::DEFAULT_PROFILE_CAP)), "N")
        ("exact-check-eta", "re-check windows whose capped gain is above -ETA on complete profiles (0 = never)",
         cxxopts::value<double>()->default_value(text(p2vbme::DEFAULT_EXACT_CHECK_ETA)), "ETA")
        ("precision", "single or double; double runs on the CPU only (the GPU computes in single precision)",
         cxxopts::value<std::string>()->default_value("single"));
    cli.add_options("Start tree")
        ("t,tree", "start tree: sample (sample-and-place), random, or a Newick file",
         cxxopts::value<std::string>()->default_value("sample"), "TREE")
        ("sample-size", "sequences sampled for sample-and-place (0 = min(100000, N/10))",
         cxxopts::value<int>()->default_value(text(p2vbme::DEFAULT_SAMPLE_SIZE)), "N");
    cli.add_options("Branch lengths and output")
        ("no-branch-lengths", "do not estimate BME branch lengths")
        ("bl-profile-cap", "leaves kept per subtree profile for branch lengths (1-256)",
         cxxopts::value<int>()->default_value(text(p2vbme::DEFAULT_BL_PROFILE_CAP)), "N")
        ("no-neg-as-zero", "write negative branch lengths as they are (default: as 0)")
        ("report-full", "add the details to <prefix>.json: host, input, settings as used, devices, time per stage, "
         "memory, model parameters, gain of each round");
    cli.add_options("Input")
        ("no-dedup", "keep all identical sequences (default: two of each group are kept for the search and the "
         "others put back next to them, with zero-length branches)");
    // clang-format on

    CommandLine result;
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
    RunOptions &options = result.options;
    options.verbose = args.count("verbose") > 0;
    options.input_path = args["alignment"].as<std::string>();
    const p2vbme::Model model = parse_model(args["model"].as<std::string>());
    options.prefix = args.count("prefix") ? args["prefix"].as<std::string>() : options.input_path;
    options.neg_as_zero = !args.count("no-neg-as-zero");

    p2vbme::Settings &opt = result.settings;
    opt.substitution.model = model;
    opt.sample_size = args["sample-size"].as<int>();
    opt.seed = args["seed"].as<uint64_t>();
    opt.max_rounds = args["max-rounds"].as<int>();
    opt.early_stopping = args["early-stopping"].as<double>();
    opt.time_limit = args["time-limit"].as<double>();
    opt.profile_cap = args["profile-cap"].as<int>();
    opt.exact_check_eta = args["exact-check-eta"].as<double>();
    opt.branch_lengths = !args.count("no-branch-lengths");
    opt.bl_profile_cap = args["bl-profile-cap"].as<int>();
    opt.accelerator = parse_accelerator(args["accelerator"].as<std::string>());
    opt.precision = parse_precision(args["precision"].as<std::string>());
    options.devices = args["devices"].as<std::string>();
    options.tree_init = args["tree"].as<std::string>();
    options.dedup = !args.count("no-dedup");
    options.report_full = args.count("report-full") > 0;
    return result;
}

void parse_devices_option(const std::string &devices, bool on_gpu, p2vbme::Settings &opt) {
    if (on_gpu)
        opt.devices = parse_devices(devices);
    else
        opt.threads = parse_threads(devices);
}
