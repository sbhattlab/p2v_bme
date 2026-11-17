/*
 * device.cpp
 *
 * The run's devices (device.hpp): the GPU or the CPU, the errors when the choice cannot run, and the devices.
 */
#include <omp.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "device.hpp"

namespace p2vbme {

// auto: the GPU if one is found (and single precision), else the CPU, with a message
DeviceChoice choose_devices(Accelerator accelerator, Precision precision) {
    DeviceChoice choice;
    choice.on_gpu = accelerator == Accelerator::Gpu;
    if (accelerator == Accelerator::Auto) {
        choice.on_gpu = precision == Precision::Single && visible_gpus() > 0;
        if (!choice.on_gpu)
            choice.note = std::string(precision == Precision::Double ? "double precision" : "no GPU found") +
                          ": running on the CPU";
    }
    return choice;
}

namespace {

// Throws std::runtime_error if the choice cannot run: the GPU in double precision, or no GPU visible.
void check_devices(const DeviceChoice &choice, Precision precision) {
    if (choice.on_gpu && precision == Precision::Double)
        throw std::runtime_error("the GPU computes in single precision; double precision runs on the CPU "
                                 "(--accelerator cpu)");
    if (choice.on_gpu && visible_gpus() < 1)
        throw std::runtime_error("no GPU found (" + gpu_build_info() + "); run on the CPU (--accelerator cpu)");
}

} // namespace

Devices select_devices(const DeviceChoice &choice, Precision precision, const std::vector<int> &gpu_ids,
                       int32_t threads) {
    for (size_t i = 0; i < gpu_ids.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (gpu_ids[i] == gpu_ids[j])
                throw std::runtime_error("GPU " + std::to_string(gpu_ids[i]) + " is listed twice");
    if (threads < 0)
        throw std::runtime_error("CPU threads must be >= 0 (0 = all)");
    if (threads > omp_get_num_procs())
        throw std::runtime_error("CPU threads must be at most the " + std::to_string(omp_get_num_procs()) +
                                 " processors available");
    check_devices(choice, precision);
    if (choice.on_gpu)
        return make_devices(gpu_ids);
    return make_cpu_devices(threads > 0 ? threads : omp_get_max_threads(), precision == Precision::Double);
}

} // namespace p2vbme
