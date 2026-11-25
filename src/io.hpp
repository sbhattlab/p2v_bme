/*
 * io.hpp
 *
 * Alignment input by file type (FASTA or .maple).
 */
#ifndef VECME_IO_HPP
#define VECME_IO_HPP

#include <string>

#include "core.hpp"

namespace vecme {

// read_maple for a path ending in .maple (any case), read_fasta otherwise.
Alignment read_alignment(const std::string &path);

} // namespace vecme

#endif // VECME_IO_HPP
