/*
 * start_tree.hpp
 *
 * Start trees of the search: sample-and-place (from the sequences) and random.
 * Both functions write the tree into their two output vectors (resized and overwritten), in the layout of Tree::c1
 * and Tree::c2 (core.hpp), with children before parents (c1[u], c2[u] < u).
 * They do not return a Tree: the search (run_multi_gpu_minibatch_bme) edits the two vectors in place, so infer() keeps
 * them through the start tree and the search, and puts them into Result::tree at the end.
 */
#ifndef P2VBME_START_TREE_HPP
#define P2VBME_START_TREE_HPP

#include <stdint.h>

#include <vector>

#include "device.hpp"
#include "search.hpp"

namespace p2vbme {

struct SampleInitStats {
    // Sample size, after clamping to [START_SAMPLE_MIN, N/2]
    int32_t n0 = 0;
    // Time for the subsample's own search (subsample + its search)
    double dt_sub = 0.0;
    // Time for the nearest-sample step (the nearest sampled sequence of every other sequence)
    double dt_nn = 0.0;
    // Time for assembling the full tree (the subsample's tree + the nearest-sample assignments)
    double dt_build = 0.0;
    // Total time for the start tree (subsample + nearest-sample + assembly)
    double dt_total = 0.0;
    // Total BME change of the subsample's own search (the subsample's tree minus the initial random tree)
    double dL_sub = 0.0;
    // The largest bucket of assigned leaves to a sampled leaf (for diagnostics)
    int32_t max_bucket = 0;
};

// Random rooted binary tree on n leaves, post-order (c1[u] < u, c2[u] < u): v_i ~ U{0..i} plus a random leaf
// permutation.
// This is the Yule-Harding (ranked-uniform) distribution, NOT uniform over topologies.
// Same construction as hx2_parallel_suite.make_uniform_random_tree (different RNG).
void make_uniform_random_tree_host(int32_t n, uint64_t seed, std::vector<int32_t> &c1, std::vector<int32_t> &c2);

// Sample-and-place start tree (branch diag-scaling)
//   1. pick n0 sequences at random (n0 clamped to [START_SAMPLE_MIN, N / 2])
//   2. start a tree t0 on them from a random (Yule-Harding) tree, or, if n0 > START_RANDOM_TREE_MAX, from a
//      sample-and-place tree of its own (this function, one level down)
//   3. optimise t0 with the minibatch search (the run's settings)
//   4. assign every other sequence to its nearest sampled sequence by p-distance over shared valid sites, for every
//      model; a pair with no shared valid site ranks like JC69 = 1.5 (p = 0.6485)
//   5. replace each sampled leaf by a balanced tree of itself + its assigned sequences (random order)
// Uses only the sequences: no information from the true tree.
// Needs N >= START_SAMPLE_MIN_N (below it, infer() uses the random start instead).
void build_sample_init_tree(int32_t N, int32_t n0, uint64_t seed, Devices &devices, std::vector<int32_t> &out_c1,
                            std::vector<int32_t> &out_c2, SampleInitStats &stats, const SearchSettings &settings,
                            int32_t level = 0);

} // namespace p2vbme

#endif // P2VBME_START_TREE_HPP
