/*
 * fasta.hpp
 *
 * Aligned DNA FASTA input, packed as in the .bin bundles: 5 bitplanes per sequence (A, C, G, T/U, observed).
 */
#ifndef P2VBME_FASTA_HPP
#define P2VBME_FASTA_HPP

#include <stdint.h>
#include <string>

#include "inference.hpp"

namespace p2vbme {

/* Reads an aligned DNA FASTA file; all sequences must have the same length. Throws std::runtime_error. */
Alignment read_fasta(const std::string &path);

// Packs one aligned sequence into its 5 bitplanes of n_words words each (A, C, G, T or U, observed), from out on;
// out must start zeroed. The .maple reader packs its reference with it.
void pack_sequence(const std::string &seq, int32_t n_words, uint64_t *out);

} // namespace p2vbme

#endif // P2VBME_FASTA_HPP
