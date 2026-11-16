/*
 * processing.cpp
 *
 * The input before inference (base frequencies, identical sequences, the start tree) and the tree after it.
 */
#include "processing.hpp"

namespace p2vbme {

void count_base_frequencies(const Alignment &aln, SubstitutionModel &model) {
    const double sum = model.base_freqs[0] + model.base_freqs[1] + model.base_freqs[2] + model.base_freqs[3];
    if ((model.model == Model::F81 || model.model == Model::F84 || model.model == Model::TN93) && !(sum > 0.0))
        base_frequencies(aln.packed.data(), aln.n_taxa, aln.n_words, model.base_freqs);
}

void prepare_alignment(Alignment &aln, SubstitutionModel &model, bool dedup, Deduplication &dd) {
    count_base_frequencies(aln, model); // before identical sequences are set aside (as FastME counts them)
    if (dedup)
        deduplicate(aln, dd);
}

void prepare_start_tree(Tree &tree, const Deduplication &dd) {
    if (dd.n_removed() > 0)
        tree = prune(tree, dd);
}

void restore_copies(Tree &tree, const Deduplication &dd) {
    if (dd.n_removed() > 0)
        tree = reattach(tree, dd);
}

} // namespace p2vbme
