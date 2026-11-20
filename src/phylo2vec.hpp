/*
 * phylo2vec.hpp
 *
 * Phylo2Vec encoding and decoding of rooted trees with ordered vectors (v[i] <= i, phylo2vec's ordered trees).
 */
#ifndef P2VBME_PHYLO2VEC_HPP
#define P2VBME_PHYLO2VEC_HPP

#include <stdint.h>

namespace p2vbme {

// ---- Phylo2Vec encoding and decoding of rooted trees (leaves 0..n-1, internal nodes n..2n-2, root 2n-2) ----

// Encodes a rooted tree as an ordered vector: (c1_r, c2_r) -> (v_out, perm_out).
// Like phylo2vec's queue_shuffle, except that the child holding the smaller leaf keeps the label.
// perm_out is phylo2vec's label mapping: perm_out[new label] = old leaf.
// flip_mode 1: the other labelling of each cherry.
void queue_shuffle_rooted(const int32_t *c1_r, const int32_t *c2_r, int32_t n_leaves, int32_t flip_mode, int64_t *v_out,
                          int32_t *perm_out);

// Decodes an ordered vector into its rooted tree: v -> (c1_out, c2_out).
// v has n_leaves - 1 entries, 0 <= v[i] <= i; c1_out and c2_out (2n - 1 entries) get each internal node's children,
// 0 for the leaves.
// Rows are read from the last: leaf i + 1 joins the subtree named v[i] under the next node, n + (n - 2 - i), and the
// merged subtree keeps the name v[i]; children always get smaller ids than their parent.
// The merges and node ids of phylo2vec's to_ancestry, with the two children the other way round (c1: leaf i + 1's
// side).
void to_skeleton(const int64_t *v, int32_t n_leaves, int32_t *c1_out, int32_t *c2_out);

} // namespace p2vbme

#endif // P2VBME_PHYLO2VEC_HPP
