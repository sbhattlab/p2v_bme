// Helpers shared by the tests that run infer() (test_inference, test_branch_lengths).
#ifndef VECME_TESTS_INFER_TEST_HELPERS_HPP
#define VECME_TESTS_INFER_TEST_HELPERS_HPP

#include "core.hpp"
#include "device.hpp"
#include "inference.hpp"
#include "processing.hpp"

namespace {

// The devices the settings choose, as the command line chooses them.
vecme::Devices devices_for(const vecme::Settings &opt) {
    return vecme::select_devices(vecme::resolve_accelerator(opt.accelerator, opt.precision), opt.precision, opt.devices,
                                 opt.threads);
}

// The settings with the alignment's base frequencies (F81, F84, TN93), as the command line counts them.
vecme::Settings with_base_frequencies(const vecme::Alignment &aln, vecme::Settings opt) {
    vecme::count_base_frequencies(aln, opt.substitution_model);
    return opt;
}

// infer() on those devices, with the alignment's base frequencies.
vecme::Result run_infer(const vecme::Alignment &aln, const vecme::Settings &settings,
                        const vecme::Tree *start_tree = nullptr) {
    const vecme::Settings opt = with_base_frequencies(aln, settings);
    vecme::Devices devices = devices_for(opt);
    return vecme::infer(aln, opt, devices, start_tree);
}

} // namespace

#endif // VECME_TESTS_INFER_TEST_HELPERS_HPP
