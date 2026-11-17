/*
 * accelerator.hpp
 *
 * What a run asks to compute on (the GPU, the CPU, or auto) and in which precision, with their text forms (the
 * command line, the report).
 */
#ifndef P2VBME_ACCELERATOR_HPP
#define P2VBME_ACCELERATOR_HPP

#include <string>

namespace p2vbme {

enum class Accelerator { Auto, Gpu, Cpu };
enum class Precision { Single, Double };

// Text forms: auto, gpu, cpu; single, double.
// parse_ throws std::runtime_error.
Accelerator parse_accelerator(const std::string &name);
const char *accelerator_name(Accelerator accelerator);
Precision parse_precision(const std::string &name);
const char *precision_name(Precision precision);

} // namespace p2vbme

#endif // P2VBME_ACCELERATOR_HPP
