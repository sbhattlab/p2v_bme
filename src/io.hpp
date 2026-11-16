/*
 * io.hpp
 *
 * Alignment input by file type (FASTA or .maple).
 */
#ifndef P2VBME_IO_HPP
#define P2VBME_IO_HPP

#include <string>

#include "core.hpp"

namespace p2vbme {

/* read_maple for a path ending in .maple (any case), read_fasta otherwise. */
Alignment read_alignment(const std::string &path);

} // namespace p2vbme

#endif // P2VBME_IO_HPP
