/*
 * processing.hpp
 *
 * The input before inference and the tree after it: the base frequencies of the whole input (F81, F84, TN93),
 * identical sequences set aside and a given start tree restricted to the sequences kept; after inference, the copies
 * joined back to the tree.
 */
#ifndef P2VBME_PROCESSING_HPP
#define P2VBME_PROCESSING_HPP

#include "core.hpp"
#include "dedup.hpp"
#include "substitution_models.hpp"

namespace p2vbme {

// F81, F84 and TN93 without given base frequencies (all 0): those of the alignment (each base's share of all valid
// sites, as FastME and ape count them). Other models are left as they are.
void count_base_frequencies(const Alignment &aln, SubstitutionModel &model);

// Before inference: count_base_frequencies on the whole input, then, with dedup, the third and later copies of
// identical sequences set aside, in place (dd records them; it stays empty without dedup).
void prepare_alignment(Alignment &aln, SubstitutionModel &model, bool dedup, Deduplication &dd);

// A given start tree on all input sequences, restricted in place to the sequences prepare_alignment kept.
void prepare_start_tree(Tree &tree, const Deduplication &dd);

// After inference: the tree on all input sequences, with the copies prepare_alignment set aside joined back in place.
void restore_copies(Tree &tree, const Deduplication &dd);

} // namespace p2vbme

#endif // P2VBME_PROCESSING_HPP
