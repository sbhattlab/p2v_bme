/*
 * accelerator.hpp
 *
 * What a run asks to compute on (the GPU, the CPU, or auto) and in which precision, with their text forms (the
 * command line, the report).
 */
#ifndef VECME_ACCELERATOR_HPP
#define VECME_ACCELERATOR_HPP

#include <string>

namespace vecme {

enum class Accelerator { Auto, Gpu, Cpu };
enum class Precision { Single, Double };

// Text forms: auto, gpu, cpu; single, double.
// parse_ throws std::runtime_error.
Accelerator parse_accelerator(const std::string &name);
const char *accelerator_name(Accelerator accelerator);
Precision parse_precision(const std::string &name);
const char *precision_name(Precision precision);

} // namespace vecme

#endif // VECME_ACCELERATOR_HPP
