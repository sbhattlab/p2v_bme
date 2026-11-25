/*
 * reattach_left_out.cpp
 *
 * Puts the sequences left out of a search back into its tree, as the SILVA analysis does: each one is joined to its
 * nearest searched sequence by zero-length branches (that sequence's branch length goes above them).
 * Nearest: the smallest p-distance (mismatches / valid sites shared), which orders pairs as the Jukes-Cantor distance
 * does; sequences that share no valid site with it are not candidates.
 * Ties go to the earlier searched sequence, and a sequence that shares no valid site with any goes next to the first.
 * Identical copies beyond two are not compared (as vecme's dedup): their distances equal their first copy's.
 *
 *   reattach_left_out TREE.nwk SEARCHED.fa LEFT_OUT.fa OUT.nwk
 *
 * TREE: vecme's tree of SEARCHED (the same leaf names), with branch lengths; LEFT_OUT: the sequences left out, on
 * the same columns.
 * Prints one JSON line: {"searched": S, "left_out": L, "no_shared_site": the left-out sequences without a candidate}.
 */
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "io.hpp"
#include "newick.hpp"
#include "processing.hpp"

namespace {

using vecme::Alignment;
using vecme::N_BASES;
using vecme::N_PLANES;
using vecme::VALID_PLANE;

// Finds each query's nearest sequence of the alignment: (aln, queries on the same columns) -> one row of aln per
// query, or -1 when no row shares a valid site with it.
std::vector<int32_t> nearest_sequences(const Alignment &aln, const Alignment &queries) {
    const int32_t W = aln.n_words;
    const size_t row_words = (size_t)N_PLANES * W;
    std::vector<int32_t> nearest(queries.n_taxa, -1);
#pragma omp parallel for schedule(dynamic, 16)
    for (int32_t q = 0; q < queries.n_taxa; ++q) {
        const uint64_t *a = &queries.packed[(size_t)q * row_words];
        double best = 2.0; // above any p-distance
        for (int32_t k = 0; k < aln.n_taxa; ++k) {
            const uint64_t *b = &aln.packed[(size_t)k * row_words];
            int64_t shared = 0, same = 0;
            for (int32_t w = 0; w < W; ++w) {
                shared += __builtin_popcountll(a[VALID_PLANE * W + w] & b[VALID_PLANE * W + w]);
                uint64_t match = 0;
                for (int32_t p = 0; p < N_BASES; ++p)
                    match |= a[p * W + w] & b[p * W + w];
                same += __builtin_popcountll(match);
            }
            if (shared == 0)
                continue;
            const double p = (double)(shared - same) / (double)shared;
            if (p < best) {
                best = p;
                nearest[q] = k;
            }
        }
    }
    return nearest;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("usage: reattach_left_out TREE.nwk SEARCHED.fa LEFT_OUT.fa OUT.nwk");
        std::vector<std::string> names; // the tree's leaves, in its order; then the sequences left out
        const vecme::Tree tree = vecme::read_newick_with_names(argv[1], names);
        Alignment searched = vecme::read_alignment(argv[2]);
        const Alignment left_out = vecme::read_alignment(argv[3]);
        if (left_out.n_sites != searched.n_sites)
            throw std::runtime_error("the sequences left out are not on the searched sequences' columns");
        const int32_t S = (int32_t)names.size();
        std::unordered_map<std::string, int32_t> leaf;
        for (int32_t i = 0; i < S; ++i)
            leaf[names[i]] = i;
        if (searched.n_taxa != S)
            throw std::runtime_error("the tree's leaves are not the searched sequences");
        for (const std::string &x : searched.names)
            if (!leaf.count(x))
                throw std::runtime_error(x + " is not a leaf of the tree");

        vecme::Deduplication copies; // the candidates: identical copies beyond two set aside
        vecme::deduplicate(searched, copies);
        const std::vector<int32_t> nearest = nearest_sequences(searched, left_out);

        // Each sequence left out as a copy of its nearest sequence's leaf
        vecme::Deduplication join;
        join.kept.resize(S);
        std::iota(join.kept.begin(), join.kept.end(), 0);
        join.copy_of.assign(S, -1);
        int32_t no_shared_site = 0;
        for (int32_t i = 0; i < left_out.n_taxa; ++i) {
            no_shared_site += nearest[i] < 0;
            join.copy_of.push_back(leaf.at(searched.names[nearest[i] < 0 ? 0 : nearest[i]]));
            names.push_back(left_out.names[i]);
        }
        vecme::write_newick_unrooted(argv[4], names, vecme::reattach(tree, join));
        std::printf("{\"searched\": %d, \"left_out\": %d, \"no_shared_site\": %d}\n", S, left_out.n_taxa,
                    no_shared_site);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "reattach_left_out: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
