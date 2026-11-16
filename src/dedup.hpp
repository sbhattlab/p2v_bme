/*
 * dedup.hpp
 *
 * Identical sequences: at most two copies of each go into the search, the others are joined back to the tree after.
 */
#ifndef P2VBME_DEDUP_HPP
#define P2VBME_DEDUP_HPP

#include <stdint.h>
#include <vector>

#include "inference.hpp"

namespace p2vbme {

/* Identical sequences: copies with the same bases and the same missing sites. */
struct Deduplication {
    std::vector<int32_t> kept;    /* the sequences kept, as positions in the input, in input order */
    std::vector<int32_t> copy_of; /* per input sequence: -1 if kept, else the position in `kept` of its first copy */
    int32_t n_removed() const { return (int32_t)(copy_of.size() - kept.size()); }
};

/* Removes the third and later copies of each group of identical sequences from the alignment, in place (the first
 * two stay, as IQ-TREE keeps two, so the group still gets a branch and a support); dd records what was removed. */
void deduplicate(Alignment &aln, Deduplication &dd);

/* The tree on all input sequences from a tree on the kept ones: each removed copy is joined to the first copy of
 * its group by zero-length branches (the first copy's branch length goes above them). */
Tree reattach(const Tree &tree, const Deduplication &dd);

/* A tree on all input sequences restricted to the kept ones (a given start tree with dedup on). */
Tree prune(const Tree &tree, const Deduplication &dd);

} // namespace p2vbme

#endif // P2VBME_DEDUP_HPP
