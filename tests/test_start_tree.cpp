/*
 * test_start_tree.cpp
 *
 * The start trees (start_tree.cpp): build_sample_and_place_tree rethrows a nearest-sample step that throws on one of
 * the devices (a fake device). The device count's effect on the sample-and-place tree: InferenceGpu in test_inference.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "fake_device_test_helpers.hpp"
#include "search.hpp"
#include "start_tree.hpp"

using namespace vecme;

TEST(StartTree, NearestSampleFailureLeavesTheStartTree) {
    Devices devices = failing_devices(2, "nearest_sampled");
    int32_t n0 = 20;
    const std::string msg = thrown_message([&] { build_sample_and_place_tree(200, n0, 1, devices, SearchSettings()); });
    EXPECT_EQ(msg, "injected failure in nearest_sampled");
}
