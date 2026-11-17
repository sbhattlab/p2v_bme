/*
 * cli.hpp
 *
 * The command line of p2v-bme: the arguments as Settings and run options; --help, --help-all and --version.
 */
#ifndef P2VBME_CLI_HPP
#define P2VBME_CLI_HPP

#include <string>

#include "inference.hpp"

// Run options
struct RunOptions {
    std::string input_path;   // -s
    std::string prefix;       // outputs <prefix>.nwk and <prefix>.json (default: the input path)
    std::string start_tree;   // -t: sample, random, or a Newick file
    std::string devices;      // --devices, read by parse_devices_option once the devices are chosen
    bool verbose = false;     // -v
    bool dedup = true;        // identical sequences set aside (--no-dedup turns it off)
    bool neg_as_zero = true;  // negative branch lengths written as 0 (--no-neg-as-zero turns it off)
    bool report_full = false; // --report-full
};

// The arguments of a run. When `done` is set, the run ends here with exit_code: --help, --help-all and --version
// (printed on stdout; 0), or -s or -m missing (the message on stderr; 2).
struct CommandLine {
    p2vbme::Settings settings; // all but the devices (parse_devices_option)
    RunOptions options;
    bool done = false;
    int exit_code = 0;
};

// Parses the arguments. Throws std::exception for an invalid one.
CommandLine parse_command_line(int argc, char **argv);

// --devices once the devices are chosen (choose_devices): on the GPU, GPU numbers into settings.devices ("auto", N:
// the first N, or a list such as 0,2); on the CPU, a number of threads into settings.threads ("auto" or N). Throws
// std::runtime_error.
void parse_devices_option(const std::string &devices, bool on_gpu, p2vbme::Settings &settings);

#endif // P2VBME_CLI_HPP
