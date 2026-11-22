/*
 * phylo2vec.cpp
 *
 * Phylo2Vec encoding and decoding of rooted trees (see phylo2vec.hpp).
 */
#include "phylo2vec.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace p2vbme {

void queue_shuffle_rooted(const int32_t *c1_r, const int32_t *c2_r, int32_t n_leaves, bool flip, int64_t *v_out,
                          int32_t *perm_out) {
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
    // Step 1: the internal nodes in breadth-first order from the root, each node's c1 before its c2: phylo2vec's queue
    // of internal nodes.
    std::vector<int32_t> ord(n_leaves - 1);
    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        if (b0 >= n_leaves)
            ord[ord_len++] = b0;
        if (b1 >= n_leaves)
            ord[ord_len++] = b1;
    }
    // Step 2: the smallest leaf of each subtree, from the leaves up.
    // phylo2vec needs no such step: on the tree of an ordered vector, c1's side always holds the larger smallest leaf
    // (to_skeleton), so the order of the children already tells them apart.
    std::vector<int32_t> min_leaf(n_nodes_r);
    for (int32_t x = 0; x < n_leaves; ++x)
        min_leaf[x] = x;
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        min_leaf[u] = (min_leaf[b0] < min_leaf[b1]) ? min_leaf[b0] : min_leaf[b1];
    }
    // Step 3: the labels, from the root down in the order of step 1 (phylo2vec's node codes).
    // The root has label 0; each internal node m passes its label x to one child (c_stay) and gives the other (c_move)
    // the next free label y, which is the row v[y - 1] = x (phylo2vec's pair (node code, next leaf)).
    // So the root's children get 0 and 1, as phylo2vec's first pair (0, 1).
    // Without flip, c_stay is the child holding the smaller leaf; with flip it is the other one, as in phylo2vec, where
    // c1 keeps the node code.
    std::vector<int32_t> rep_leaf(n_nodes_r, 0);
    int32_t next_label = 1;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t m = ord[idx];
        int32_t c1 = c1_r[m], c2 = c2_r[m];
        int32_t c_stay = (min_leaf[c1] < min_leaf[c2]) ? c1 : c2;
        int32_t c_move = (min_leaf[c1] < min_leaf[c2]) ? c2 : c1;
        if (flip)
            std::swap(c_stay, c_move);
        int32_t x = rep_leaf[m];
        int32_t y = next_label++;
        rep_leaf[c_stay] = x;
        rep_leaf[c_move] = y;
        v_out[y - 1] = x;
    }
    // Step 4: the label mapping, perm_out[new label] = old leaf (phylo2vec's label_mapping).
    for (int32_t old_l = 0; old_l < n_leaves; ++old_l) {
        perm_out[rep_leaf[old_l]] = old_l;
    }
}

void to_skeleton(const int64_t *v, int32_t n_leaves, int32_t *c1_out, int32_t *c2_out) {
    std::vector<int32_t> top(n_leaves); // the top node of the subtree each name (a leaf) stands for
    for (int32_t x = 0; x < n_leaves; ++x) {
        top[x] = x;
        c1_out[x] = 0;
        c2_out[x] = 0;
    }
    for (int32_t k = 0; k < n_leaves - 1; ++k) {
        const int32_t i = n_leaves - 2 - k;
        const int32_t m = n_leaves + k;
        const int32_t x = (int32_t)v[i];
        const int32_t y = i + 1;
        c1_out[m] = top[y];
        c2_out[m] = top[x];
        top[x] = m;
    }
}

} // namespace p2vbme
