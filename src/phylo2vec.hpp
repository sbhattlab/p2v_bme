/*
 * phylo2vec.hpp
 *
 * Phylo2Vec encoding of rooted trees: the ordered vector of a rooted tree (v[i] <= i, phylo2vec's ordered trees).
 */
#ifndef P2VBME_PHYLO2VEC_HPP
#define P2VBME_PHYLO2VEC_HPP

#include <stdint.h>

namespace p2vbme {

// ---- Phylo2Vec encoding of rooted trees (leaves 0..n-1, internal nodes n..2n-2, root 2n-2) ----

// Ordered Phylo2Vec vector of a rooted tree and the leaf permutation it implies (flip_mode 1: the other labelling
// of each cherry).
// Like phylo2vec's queue_shuffle, except that the child holding the smaller leaf keeps the label.
void queue_shuffle_rooted(const int32_t *c1_r, const int32_t *c2_r, int32_t n_leaves, int32_t flip_mode, int64_t *v_out,
                          int32_t *perm_out);

} // namespace p2vbme

#endif // P2VBME_PHYLO2VEC_HPP
