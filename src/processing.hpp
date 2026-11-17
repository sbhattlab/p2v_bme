/*
 * processing.hpp
 *
 * The alignment before inference and the trees around it.
 * The alignment: the base frequencies of the whole input (F81, F84, TN93), then identical sequences set aside (at most
 * two copies of each go into the search).
 * The trees: a given start tree restricted to the sequences kept and, after inference, the copies joined back.
 */
#ifndef P2VBME_PROCESSING_HPP
#define P2VBME_PROCESSING_HPP

#include <stdint.h>
#include <vector>

#include "core.hpp"
#include "substitution_models.hpp"

namespace p2vbme {

// ---- The alignment: base frequencies, identical sequences set aside ----

// F81, F84 and TN93 without given base frequencies (all 0): those of the alignment (each base's share of all valid
// sites, as FastME and ape count them).
// Other models are left as they are.
void count_base_frequencies(const Alignment &aln, SubstitutionModel &model);

// Identical sequences: copies with the same bases and the same missing sites.
struct Deduplication {
    std::vector<int32_t> kept;    // the sequences kept, as positions in the input, in input order
    std::vector<int32_t> copy_of; // per input sequence: -1 if kept, else the position in `kept` of its first copy
    int32_t n_removed() const { return (int32_t)(copy_of.size() - kept.size()); }
};

// Removes the third and later copies of each group of identical sequences from the alignment, in place.
// The first two stay, as IQ-TREE keeps two, so the group still gets a branch and a support.
// dd records what was removed.
void deduplicate(Alignment &aln, Deduplication &dd);

// Before inference: count_base_frequencies on the whole input, then, with dedup, deduplicate (in place).
// dd records the copies set aside; it stays empty without dedup.
void prepare_alignment(Alignment &aln, SubstitutionModel &model, bool dedup, Deduplication &dd);

// ---- Trees: a given start tree restricted, the copies put back ----

// A tree on all input sequences restricted to the kept ones (a given start tree with dedup on).
Tree prune(const Tree &tree, const Deduplication &dd);

// A given start tree on all input sequences, restricted in place to the sequences prepare_alignment kept.
void prepare_start_tree(Tree &tree, const Deduplication &dd);

// The tree on all input sequences from a tree on the kept ones.
// Each removed copy is joined to the first copy of its group by zero-length branches (the first copy's branch
// length goes above them).
Tree reattach(const Tree &tree, const Deduplication &dd);

// After inference: the tree on all input sequences, with the copies prepare_alignment set aside joined back in place.
void restore_copies(Tree &tree, const Deduplication &dd);

} // namespace p2vbme

#endif // P2VBME_PROCESSING_HPP
