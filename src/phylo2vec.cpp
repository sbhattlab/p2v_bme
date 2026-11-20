/*
 * phylo2vec.cpp
 *
 * Phylo2Vec encoding of rooted trees (see phylo2vec.hpp).
 */
#include "phylo2vec.hpp"

#include <stdint.h>

#include <vector>

namespace p2vbme {

void queue_shuffle_rooted(const int32_t *c1_r, const int32_t *c2_r, int32_t n_leaves, int32_t flip_mode, int64_t *v_out,
                          int32_t *perm_out) {
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
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
    std::vector<int32_t> min_leaf(n_nodes_r);
    for (int32_t x = 0; x < n_leaves; ++x)
        min_leaf[x] = x;
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        min_leaf[u] = (min_leaf[b0] < min_leaf[b1]) ? min_leaf[b0] : min_leaf[b1];
    }
    std::vector<int32_t> rep_leaf(n_nodes_r, 0);
    int32_t rc1 = c1_r[root_r], rc2 = c2_r[root_r];
    int32_t r_stay = (min_leaf[rc1] < min_leaf[rc2]) ? rc1 : rc2;
    int32_t r_move = (min_leaf[rc1] < min_leaf[rc2]) ? rc2 : rc1;
    if (flip_mode == 1) {
        int32_t tmp = r_stay;
        r_stay = r_move;
        r_move = tmp;
    }
    rep_leaf[r_stay] = 0;
    rep_leaf[r_move] = 1;
    v_out[0] = 0;
    int32_t next_label = 2;
    for (int32_t idx = 1; idx < ord_len; ++idx) {
        int32_t m = ord[idx];
        int32_t c1 = c1_r[m], c2 = c2_r[m];
        int32_t c_stay = (min_leaf[c1] < min_leaf[c2]) ? c1 : c2;
        int32_t c_move = (min_leaf[c1] < min_leaf[c2]) ? c2 : c1;
        if (flip_mode == 1) {
            int32_t tmp = c_stay;
            c_stay = c_move;
            c_move = tmp;
        }
        int32_t x = rep_leaf[m];
        int32_t y = next_label++;
        rep_leaf[c_stay] = x;
        rep_leaf[c_move] = y;
        v_out[y - 1] = x;
    }
    for (int32_t old_l = 0; old_l < n_leaves; ++old_l) {
        perm_out[rep_leaf[old_l]] = old_l;
    }
}

} // namespace p2vbme
