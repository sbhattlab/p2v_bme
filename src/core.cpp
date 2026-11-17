/*
 * core.cpp
 *
 * The packing of one aligned sequence into its bitplanes (FASTA and .maple input).
 */
#include <array>
#include <stdint.h>
#include <string>

#include "core.hpp"

namespace p2vbme {

namespace {

// The plane of each byte: A 0, C 1, G 2, T and U 3 (either case), -1 for anything else (a missing site).
// One table read per site instead of toupper and a chain of comparisons: about 3 times faster.
constexpr std::array<int8_t, 256> kPlaneOf = [] {
    std::array<int8_t, 256> t{};
    for (int c = 0; c < 256; ++c)
        t[c] = -1;
    t['A'] = t['a'] = 0;
    t['C'] = t['c'] = 1;
    t['G'] = t['g'] = 2;
    t['T'] = t['t'] = t['U'] = t['u'] = 3;
    return t;
}();

} // namespace

// Packs one aligned sequence; mirrors bme/bundles.py pack_sequences_bits.
void pack_sequence(const std::string &seq, int32_t n_words, uint64_t *out) {
    const char *s = seq.data();
    const size_t n = seq.size();
    uint64_t *observed = out + (size_t)4 * n_words;
    for (size_t i = 0; i < n; ++i) {
        const int plane = kPlaneOf[(unsigned char)s[i]];
        if (plane < 0)
            continue;
        const uint64_t bit = 1ULL << (i % 64);
        out[(size_t)plane * n_words + i / 64] |= bit;
        observed[i / 64] |= bit;
    }
}

} // namespace p2vbme
