/*
 * fasta.hpp
 *
 * Aligned DNA FASTA input, packed as 5 bitplanes per sequence (A, C, G, T/U, valid).
 */
#ifndef VECME_FASTA_HPP
#define VECME_FASTA_HPP

#include <string>

#include "core.hpp"

namespace vecme {

// Reads an aligned DNA FASTA file.
// All sequences must have the same length.
Alignment read_fasta(const std::string &path);

} // namespace vecme

#endif // VECME_FASTA_HPP
