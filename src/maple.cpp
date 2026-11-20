/*
 * maple.cpp
 *
 * .maple alignment input (MAPLE / CMAPLE), packed as FASTA input is.
 */
#include <ctype.h>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core.hpp"
#include "maple.hpp"

namespace p2vbme {

Alignment read_maple(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    // Every line, without its ending ('\n', or '\r\n') and trailing spaces or tabs.
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        lines.push_back(std::move(line));
    }
    if (in.bad())
        throw std::runtime_error("cannot read " + path);
    auto header_name = [&](size_t i) {
        const std::string &h = lines[i];
        size_t p = 1;
        while (p < h.size() && (h[p] == ' ' || h[p] == '\t'))
            ++p;
        size_t q = p;
        while (q < h.size() && h[q] != ' ' && h[q] != '\t')
            ++q;
        if (q == p)
            throw std::runtime_error(path + ":" + std::to_string(i + 1) + ": empty sequence name");
        return h.substr(p, q - p);
    };
    size_t i = 0;
    while (i < lines.size() && lines[i].empty())
        ++i;
    if (i == lines.size() || lines[i][0] != '>')
        throw std::runtime_error(path + ": a .maple file starts with the reference sequence ('>' header)");
    // the reference: every line up to the next header
    std::string ref;
    for (++i; i < lines.size() && (lines[i].empty() || lines[i][0] != '>'); ++i)
        for (char ch : lines[i])
            if (ch != ' ' && ch != '\t')
                ref.push_back(ch);
    if (ref.empty())
        throw std::runtime_error(path + ": the reference sequence is empty");

    Alignment aln;
    aln.n_sites = (int32_t)ref.size();
    aln.n_words = (aln.n_sites + 63) / 64;
    const int32_t W = aln.n_words, L = aln.n_sites;
    std::vector<uint64_t> ref_row((size_t)N_PLANES * W, 0);
    pack_sequence(ref, W, ref_row.data());
    std::unordered_map<std::string, int32_t> seen;
    while (i < lines.size()) {
        const std::string name = header_name(i);
        if (!seen.emplace(name, aln.n_taxa).second)
            throw std::runtime_error(path + ": duplicate name '" + name + "'");
        const size_t row = aln.packed.size();
        aln.packed.insert(aln.packed.end(), ref_row.begin(), ref_row.end());
        uint64_t *r = aln.packed.data() + row;
        // differences: "<char> <position> [<length>]", positions from 1; A, C, G, T (or U) set the base, anything else
        // (n, -, ambiguity codes) marks the sites missing
        for (++i; i < lines.size() && (lines[i].empty() || lines[i][0] != '>'); ++i) {
            const std::string &ln = lines[i];
            if (ln.empty())
                continue;
            const std::string where = path + ":" + std::to_string(i + 1) + ": ";
            const char *p = ln.c_str() + 1;
            char *endp = NULL;
            const long pos = strtol(p, &endp, 10);
            if (endp == p || (ln.size() > 1 && ln[1] != ' ' && ln[1] != '\t'))
                throw std::runtime_error(where + "expected '<character> <position> [<length>]', got '" + ln + "'");
            long span = 1;
            const char *q = endp;
            while (*q == ' ' || *q == '\t')
                ++q;
            if (*q) {
                span = strtol(q, &endp, 10);
                if (endp == q || *endp)
                    throw std::runtime_error(where + "expected '<character> <position> [<length>]', got '" + ln + "'");
            }
            if (pos < 1 || pos > L || span < 1 || span > L - pos + 1) // (no overflow for huge lengths)
                throw std::runtime_error(where + "position " + std::to_string(pos) + ", length " +
                                         std::to_string(span) + ": outside the reference (sites 1.." +
                                         std::to_string(L) + ")");
            const int c = toupper((unsigned char)ln[0]);
            const int plane = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : (c == 'T' || c == 'U') ? 3 : -1;
            for (long site = pos - 1; site < pos - 1 + span; ++site) {
                const uint64_t bit = 1ULL << (site % 64);
                const size_t w = (size_t)(site / 64);
                for (int b = 0; b < N_PLANES; ++b)
                    r[(size_t)b * W + w] &= ~bit;
                if (plane >= 0) {
                    r[(size_t)plane * W + w] |= bit;
                    r[(size_t)VALID_PLANE * W + w] |= bit;
                }
            }
        }
        aln.names.push_back(name);
        ++aln.n_taxa;
    }
    if (aln.n_taxa == 0)
        throw std::runtime_error(path + ": no sequences after the reference");
    return aln;
}

} // namespace p2vbme
