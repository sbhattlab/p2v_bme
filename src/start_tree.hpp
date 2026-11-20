/*
 * start_tree.hpp
 *
 * Start trees of the search: sample-and-place (from the sequences) and random.
 * Both functions return a Tree (core.hpp) without branch lengths, with children before parents (c1[u], c2[u] < u).
 */
#ifndef P2VBME_START_TREE_HPP
#define P2VBME_START_TREE_HPP

#include <stdint.h>

#include <vector>

#include "device.hpp"
#include "search.hpp"

namespace p2vbme {

// Random rooted binary tree on n_leaves leaves, post-order (c1[u] < u, c2[u] < u): v_i ~ U{0..i} plus a random leaf
// permutation.
// This is the Yule-Harding (ranked-uniform) distribution, NOT uniform over topologies.
Tree build_random_tree(int32_t n_leaves, uint64_t seed);

// Sample-and-place start tree
//   1. pick n0_leaves sequences at random (n0_leaves clamped to [START_SAMPLE_MIN, n_leaves / 2])
//   2. start a tree t0 on them from a random (Yule-Harding) tree, or, if n0_leaves > START_RANDOM_TREE_MAX, from a
//      sample-and-place tree of its own (this function, one level down)
//   3. optimise t0 with the minibatch search (the run's settings)
//   4. assign every other sequence to its nearest sampled sequence by p-distance over shared valid sites, for every
//      model; a pair with no shared valid site ranks like JC69 = 1.5 (p = 0.6485)
//   5. replace each sampled leaf by a balanced tree of itself + its assigned sequences (random order)
// Needs n_leaves >= START_SAMPLE_MIN_N (below it, infer() uses the random start instead).
// n0_leaves: in, the sample size asked for; out, the size used (after the clamp of step 1).
Tree build_sample_and_place_tree(int32_t n_leaves, int32_t &n0_leaves, uint64_t seed, Devices &devices,
                                 const SearchSettings &settings, int32_t level = 0);

} // namespace p2vbme

#endif // P2VBME_START_TREE_HPP
