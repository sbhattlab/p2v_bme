/*
 * core.hpp
 *
 * The types the whole package shares: a DNA alignment packed as bitplanes (Alignment) and a rooted binary tree
 * (Tree), and the packing of one aligned sequence into its bitplanes (pack_sequence).
 */
#ifndef P2VBME_CORE_HPP
#define P2VBME_CORE_HPP

#include <stdint.h>

#include <string>
#include <vector>

namespace p2vbme {

// The DNA alphabet: the bases A, C, G, T/U, and the GTR exchangeabilities of their pairs (AC, AG, AT, CG, CT, GT).
constexpr int32_t N_BASES = 4;
constexpr int32_t N_GTR_RATES = 6;

// The bitplanes of a packed sequence: one per base (planes 0..3), then valid (= any of the four).
constexpr int32_t VALID_PLANE = N_BASES;
constexpr int32_t N_PLANES = N_BASES + 1;
// Bits per word of a bitplane.
constexpr int32_t WORD_BITS = 64;

// N aligned sequences packed as bitplanes: packed[(i * N_PLANES + plane) * n_words + w], bit k of word w is site
// w * WORD_BITS + k.
// Other characters (gaps, N, IUPAC ambiguity codes) set no bit.
struct Alignment {
    int32_t n_taxa = 0;
    int32_t n_sites = 0;
    int32_t n_words = 0;
    std::vector<std::string> names;
    std::vector<uint64_t> packed;
};

// Rooted binary tree on N leaves: leaves 0..N-1 (alignment order), internal nodes N..2N-2, root 2N-2, children
// c1[u], c2[u].
// In start trees children come before parents (c1[u], c2[u] < u); the search keeps the root but reuses the other
// internal ids, so in its result they are in no particular order: traverse from the root.
// branch_lengths[u] is the length of the edge above u (empty when not computed): float values, or double values when
// double_precision_lengths (the CPU device's --precision double; the output keeps their digits).
struct Tree {
    std::vector<int32_t> c1, c2;
    std::vector<double> branch_lengths;
    bool double_precision_lengths = false;

    // N, from the 2N - 1 nodes.
    int32_t n_leaves() const { return ((int32_t)c1.size() + 1) / 2; }
};

// Packs one aligned sequence into its N_PLANES bitplanes (A, C, G, T or U, valid), n_words words each, starting at out.
// out must start zeroed, and seq must have at most n_words * WORD_BITS sites.
// The .maple reader packs its reference with it.
void pack_sequence(const std::string &seq, int32_t n_words, uint64_t *out);

} // namespace p2vbme

#endif // P2VBME_CORE_HPP
