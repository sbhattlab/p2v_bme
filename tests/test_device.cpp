// The run's devices (device.hpp, device.cpp): the settings select_devices refuses (runs in CPU-only builds).
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "accelerator.hpp"
#include "device.hpp"

using namespace vecme;

// A negative thread count, and the GPU in double precision (refused before any GPU is touched).
TEST(Device, SelectDevicesRefusesBadSettings) {
    EXPECT_THROW(select_devices(Accelerator::Cpu, Precision::Single, {}, -1), std::runtime_error);
    EXPECT_THROW(select_devices(Accelerator::Gpu, Precision::Double, {}, 0), std::runtime_error);
}
