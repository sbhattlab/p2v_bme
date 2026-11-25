/*
 * core.cpp
 *
 * The packing of one aligned sequence into its bitplanes (FASTA and .maple input).
 */
#include "core.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace vecme {

namespace {

// The plane of each byte: A 0, C 1, G 2, T and U 3 (either case), -1 for anything else (a missing site).
// One table read per site, faster than toupper and a chain of comparisons.
constexpr std::array<int8_t, 256> PLANE_OF = [] {
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

void pack_sequence(const std::string &seq, int32_t n_words, uint64_t *out) {
    const char *s = seq.data();
    const size_t n = seq.size();
    uint64_t *valid = out + (size_t)VALID_PLANE * n_words;
    for (size_t i = 0; i < n; ++i) {
        const int plane = PLANE_OF[(unsigned char)s[i]];
        if (plane < 0)
            continue;
        const uint64_t bit = 1ULL << (i % WORD_BITS);
        out[(size_t)plane * n_words + i / WORD_BITS] |= bit;
        valid[i / WORD_BITS] |= bit;
    }
}

} // namespace vecme
