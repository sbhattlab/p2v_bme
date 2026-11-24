/*
 * cli.hpp
 *
 * The command line of p2v-bme: the arguments as Settings and the Report's command-line part; --help, --help-all and
 * --version.
 */
#ifndef P2VBME_CLI_HPP
#define P2VBME_CLI_HPP

#include <string>

#include "inference.hpp"
#include "report.hpp"

// Command line arguments
struct CommandLine {
    p2vbme::Settings settings; // Inference settings
    p2vbme::Report report;     // Report of the run
    std::string devices;       // Devices
    bool quiet = false;
    bool overwrite = false; // replace a previous run's outputs
    bool done = false;
    int exit_code = 0;
};

// Parse CLI args
CommandLine parse_command_line(int argc, char **argv);

// Parse device options via `--devices`
// on the GPU, GPU numbers into settings.devices ("auto": the first GPU, N: the first N, or a list such as 0,2)
// on the CPU: a number of threads into settings.threads ("auto": 0, all cores but one, or N).
void parse_devices_option(const std::string &devices, bool on_gpu, p2vbme::Settings &settings);

#endif // P2VBME_CLI_HPP
