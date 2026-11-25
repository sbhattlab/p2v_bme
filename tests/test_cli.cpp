// Unit tests for --devices (parse_devices_option): GPU numbers on the GPU, a number of threads on the CPU.
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"

namespace {

std::vector<int> gpus(const std::string &devices) {
    vecme::Settings settings;
    parse_devices_option(devices, true, settings);
    return settings.devices;
}

int32_t threads(const std::string &devices) {
    vecme::Settings settings;
    parse_devices_option(devices, false, settings);
    return settings.threads;
}

// The message of the error for `devices`, or "" if it is accepted.
std::string error(const std::string &devices, bool on_gpu) {
    vecme::Settings settings;
    try {
        parse_devices_option(devices, on_gpu, settings);
    } catch (const std::runtime_error &e) {
        return e.what();
    }
    return "";
}

TEST(DevicesOption, GpuForms) {
    EXPECT_EQ(gpus("auto"), (std::vector<int>{0})); // the first GPU
    EXPECT_EQ(gpus("1"), (std::vector<int>{0}));    // N: the first N
    EXPECT_EQ(gpus("3"), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(gpus("0,2"), (std::vector<int>{0, 2})); // a list of GPU numbers
    EXPECT_EQ(gpus("1,"), (std::vector<int>{1}));     // GPU 1 alone: a list of one
}

TEST(DevicesOption, GpuRefused) {
    for (const std::string d : {"x", "0", "-1", "1.5", ",", "0,x", ""}) {
        const std::string message = error(d, true);
        EXPECT_EQ(message.rfind("unknown value for devices: '" + d + "'", 0), 0u) << "'" << d << "': " << message;
    }
}

TEST(DevicesOption, CpuForms) {
    EXPECT_EQ(threads("auto"), 0); // the OpenMP default, which main sets to all cores but one
    EXPECT_EQ(threads("1"), 1);
    EXPECT_EQ(threads("16"), 16);
}

TEST(DevicesOption, CpuRefused) {
    for (const std::string d : {"0", "0,1", "x", "-2", "+4", " 4", "", "1234567890"}) {
        const std::string message = error(d, false);
        EXPECT_EQ(message.rfind("unknown value for devices on the CPU: '" + d + "'", 0), 0u)
            << "'" << d << "': " << message;
    }
}

} // namespace
