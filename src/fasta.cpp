/*
 * fasta.cpp
 *
 * Aligned DNA FASTA input, packed as in the .bin bundles.
 */
#include <array>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

#include "fasta.hpp"

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

/* Packs one aligned sequence; mirrors bme/bundles.py pack_sequences_bits. */
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

Alignment read_fasta(const std::string &path) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    Alignment aln;
    std::unordered_map<std::string, int32_t> seen;
    std::string name, seq;
    bool have_record = false;
    auto finish_record = [&]() {
        if (!have_record)
            return;
        if (aln.n_taxa == 0) {
            if (seq.empty())
                throw std::runtime_error(path + ": first sequence '" + name + "' is empty");
            aln.n_sites = (int32_t)seq.size();
            aln.n_words = (aln.n_sites + 63) / 64;
        } else if ((int32_t)seq.size() != aln.n_sites) {
            throw std::runtime_error(path + ": sequence '" + name + "' has " + std::to_string(seq.size()) +
                                     " sites, the first has " + std::to_string(aln.n_sites) +
                                     " (the alignment must be aligned)");
        }
        if (!seen.emplace(name, aln.n_taxa).second)
            throw std::runtime_error(path + ": duplicate name '" + name + "'");
        const size_t row = aln.packed.size();
        aln.packed.resize(row + (size_t)5 * aln.n_words, 0);
        pack_sequence(seq, aln.n_words, aln.packed.data() + row);
        aln.names.push_back(name);
        ++aln.n_taxa;
        seq.clear();
    };

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    long line_no = 0;
    try {
        while ((len = getline(&line, &cap, f)) != -1) {
            ++line_no;
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';
            if (len > 0 && line[0] == '>') {
                finish_record();
                const char *p = line + 1;
                while (*p == ' ' || *p == '\t')
                    ++p;
                const char *q = p;
                while (*q && *q != ' ' && *q != '\t')
                    ++q;
                name.assign(p, q - p);
                if (name.empty())
                    throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty sequence name");
                have_record = true;
                continue;
            }
            for (ssize_t i = 0; i < len; ++i) {
                const char ch = line[i];
                if (ch == ' ' || ch == '\t')
                    continue;
                if (!have_record)
                    throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                             ": sequence data before the first '>' header");
                seq.push_back(ch);
            }
        }
    } catch (...) { /* release the line buffer and the file before passing the error on */
        free(line);
        fclose(f);
        throw;
    }
    free(line);
    fclose(f);
    finish_record();
    if (aln.n_taxa == 0)
        throw std::runtime_error(path + ": no sequences");
    return aln;
}

} // namespace p2vbme
