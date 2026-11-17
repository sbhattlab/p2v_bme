/*
 * substitution_models.cpp
 *
 * Model names and the pair counts of two alignment rows (host).
 */
#include "substitution_models.hpp"

#include <ctype.h>

#include <stdexcept>

namespace p2vbme {

const char *model_name(Model model) {
    switch (model) {
    case Model::JC69:
        return "JC69";
    case Model::K80:
        return "K80";
    case Model::F81:
        return "F81";
    case Model::F84:
        return "F84";
    case Model::TN93:
        return "TN93";
    case Model::LogDet:
        return "LogDet";
    case Model::GTR_GC:
        return "GTR+GC";
    }
    return "unknown";
}

Model parse_model(const std::string &name) {
    std::string m = name;
    for (char &c : m)
        c = (char)toupper((unsigned char)c);
    if (m == "JC69" || m == "JC")
        return Model::JC69;
    if (m == "K80" || m == "K2P")
        return Model::K80;
    if (m == "F81")
        return Model::F81;
    if (m == "F84")
        return Model::F84;
    if (m == "TN93" || m == "TN")
        return Model::TN93;
    if (m == "LOGDET")
        return Model::LogDet;
    if (m == "GTR+GC")
        return Model::GTR_GC;
    if (m.size() >= 2 && m.compare(m.size() - 2, 2, "+G") == 0)
        throw std::runtime_error("model '" + name + "': continuous gamma is written +GC (e.g. GTR+GC); +G is not used");
    if (m == "GTR")
        throw std::runtime_error("model 'GTR' (without gamma) is not available; use GTR+GC");
    throw std::runtime_error("unknown value for model: '" + name + "' (JC69, K80, F81, F84, TN93, LogDet or GTR+GC)");
}

void base_frequencies(const uint64_t *packed, int32_t n_taxa, int32_t n_words, double freqs[4]) {
    uint64_t counts[4] = {0, 0, 0, 0};
    for (int32_t x = 0; x < n_taxa; ++x)
        for (int b = 0; b < 4; ++b) {
            const uint64_t *plane = packed + ((size_t)x * 5 + b) * n_words;
            for (int32_t w = 0; w < n_words; ++w)
                counts[b] += (uint64_t)__builtin_popcountll(plane[w]);
        }
    const uint64_t total = counts[0] + counts[1] + counts[2] + counts[3];
    for (int b = 0; b < 4; ++b)
        freqs[b] = total > 0 ? (double)counts[b] / (double)total : 0.25;
}

PairCounts count_pair(const uint64_t *row_a, const uint64_t *row_b, int32_t n_words) {
    PairCounts c;
    for (int i = 0; i < 4; ++i) {
        const uint64_t *pa = row_a + (size_t)i * n_words;
        for (int j = 0; j < 4; ++j) {
            const uint64_t *pb = row_b + (size_t)j * n_words;
            uint32_t acc = 0;
            for (int32_t w = 0; w < n_words; ++w)
                acc += (uint32_t)__builtin_popcountll(pa[w] & pb[w]);
            c.n[4 * i + j] = acc;
        }
    }
    return c;
}

} // namespace p2vbme
