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

// auto: the GPU if one is found (and single precision), else the CPU
Accelerator resolve_accelerator(Accelerator accelerator, Precision precision) {
    if (accelerator != Accelerator::Auto)
        return accelerator;
    return (precision == Precision::Single && visible_gpus() > 0) ? Accelerator::Gpu : Accelerator::Cpu;
}

namespace {

// Throws std::runtime_error if the accelerator cannot run: the GPU in double precision, or no GPU visible.
void check_devices(Accelerator accelerator, Precision precision) {
    if (accelerator == Accelerator::Gpu && precision == Precision::Double)
        throw std::runtime_error("the GPU computes in single precision; double precision runs on the CPU "
                                 "(--accelerator cpu)");
    if (accelerator == Accelerator::Gpu && visible_gpus() < 1)
        throw std::runtime_error(no_gpu_reason() + "; run on the CPU (--accelerator cpu)");
}

} // namespace

Devices select_devices(Accelerator accelerator, Precision precision, const std::vector<int> &gpu_ids, int32_t threads) {
    if (accelerator == Accelerator::Auto)
        throw std::logic_error("select_devices: resolve the accelerator first (resolve_accelerator)");
    for (size_t i = 0; i < gpu_ids.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (gpu_ids[i] == gpu_ids[j])
                throw std::runtime_error("GPU " + std::to_string(gpu_ids[i]) + " is listed twice");
    if (threads < 0)
        throw std::runtime_error("CPU threads must be >= 0 (0 = all)");
    if (threads > omp_get_num_procs())
        throw std::runtime_error("CPU threads must be at most the " + std::to_string(omp_get_num_procs()) +
                                 " processors available");
    check_devices(accelerator, precision);
    if (accelerator == Accelerator::Gpu)
        return make_devices(gpu_ids);
    return make_cpu_devices(threads > 0 ? threads : omp_get_max_threads(), precision == Precision::Double);
}

} // namespace p2vbme
