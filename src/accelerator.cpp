/*
 * accelerator.cpp
 *
 * The text forms of Accelerator and Precision (accelerator.hpp).
 */
#include "accelerator.hpp"

#include <stdexcept>
#include <string>

namespace p2vbme {

Accelerator parse_accelerator(const std::string &name) {
    if (name == "auto")
        return Accelerator::Auto;
    if (name == "gpu")
        return Accelerator::Gpu;
    if (name == "cpu")
        return Accelerator::Cpu;
    throw std::runtime_error("unknown value for accelerator: '" + name + "' (cpu, gpu or auto)");
}

const char *accelerator_name(Accelerator accelerator) {
    switch (accelerator) {
    case Accelerator::Gpu:
        return "gpu";
    case Accelerator::Cpu:
        return "cpu";
    default:
        return "auto";
    }
}

Precision parse_precision(const std::string &name) {
    if (name == "single")
        return Precision::Single;
    if (name == "double")
        return Precision::Double;
    throw std::runtime_error("unknown value for precision: '" + name + "' (single or double)");
}

const char *precision_name(Precision precision) { return precision == Precision::Double ? "double" : "single"; }

} // namespace p2vbme
