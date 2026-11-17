// Helpers shared by the tests that run infer() (test_cpu_device, test_device_distances).
#ifndef P2VBME_TESTS_INFER_TEST_HELPERS_HPP
#define P2VBME_TESTS_INFER_TEST_HELPERS_HPP

#include "core.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "processing.hpp"

namespace {

// The devices the settings choose, as the command line chooses them.
p2vbme::Devices devices_for(const p2vbme::Settings &opt) {
    return p2vbme::select_devices(p2vbme::choose_devices(opt.accelerator, opt.precision), opt.precision, opt.devices,
                                  opt.threads);
}

// The settings with the alignment's base frequencies (F81, F84, TN93), as the command line counts them.
p2vbme::Settings with_base_frequencies(const p2vbme::Alignment &aln, p2vbme::Settings opt) {
    p2vbme::count_base_frequencies(aln, opt.substitution_model);
    return opt;
}

// infer() on those devices, with the alignment's base frequencies.
p2vbme::Result run_infer(const p2vbme::Alignment &aln, const p2vbme::Settings &settings,
                         const p2vbme::Tree *start_tree = nullptr) {
    const p2vbme::Settings opt = with_base_frequencies(aln, settings);
    p2vbme::Devices devices = devices_for(opt);
    return p2vbme::infer(aln, opt, devices, start_tree);
}

} // namespace

#endif // P2VBME_TESTS_INFER_TEST_HELPERS_HPP
