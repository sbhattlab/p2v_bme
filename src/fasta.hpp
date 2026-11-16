/*
 * fasta.hpp
 *
 * Aligned DNA FASTA input, packed as in the .bin bundles: 5 bitplanes per sequence (A, C, G, T/U, observed).
 */
#ifndef P2VBME_FASTA_HPP
#define P2VBME_FASTA_HPP

#include <string>

#include "core.hpp"

namespace p2vbme {

/* Reads an aligned DNA FASTA file; all sequences must have the same length. Throws std::runtime_error. */
Alignment read_fasta(const std::string &path);

} // namespace p2vbme

#endif // P2VBME_FASTA_HPP
