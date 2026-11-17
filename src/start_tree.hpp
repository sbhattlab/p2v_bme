/*
 * start_tree.hpp
 *
 * Start trees of the search: sample-and-place (from the sequences) and random.
 */
#ifndef P2VBME_START_TREE_HPP
#define P2VBME_START_TREE_HPP

#include <stdint.h>

#include <vector>

#include "device.hpp"
#include "search.hpp"

namespace p2vbme {

struct SampleInitStats {
    int32_t n0 = 0;
    double dt_sub = 0.0, dt_nn = 0.0, dt_build = 0.0, dt_total = 0.0;
    double dL_sub = 0.0;
    int32_t max_bucket = 0;
};

/* Random rooted binary tree on n leaves, post-order (c1[u] < u, c2[u] < u): v_i ~ U{0..i} plus a
 * random leaf permutation. This is the Yule-Harding (ranked-uniform) distribution, NOT uniform over
 * topologies. Same construction as hx2_parallel_suite.make_uniform_random_tree (different RNG). */
void make_uniform_random_tree_host(int32_t n, uint64_t seed, std::vector<int32_t> &c1, std::vector<int32_t> &c2);

/* ============================================================================
 * Sample-and-place start tree (branch diag-scaling)
 *   1. optimise a random subsample of n0 leaves, starting from a Yule random tree, with the
 *      same minibatch optimiser (the run's settings)
 *   2. assign every other leaf to its nearest sampled leaf by p-distance over shared valid sites, for
 *      every model (the order of JC69, which is monotone in p; the paper's GPU implementation did the same for GTR+GC);
 *      a pair with no shared valid site ranks like JC69 = 1.5 (p = 0.6485)
 *   3. replace each sampled leaf by a balanced tree of itself + its assigned leaves (random order)
 * Uses only the sequences: no information from the true tree.
 * ============================================================================ */

void build_sample_init_tree(int32_t N, int32_t n0, uint64_t seed, Devices &devices, std::vector<int32_t> &out_c1,
                            std::vector<int32_t> &out_c2, SampleInitStats &stats, const SearchSettings &settings,
                            int32_t level = 0);

} // namespace p2vbme

#endif // P2VBME_START_TREE_HPP
