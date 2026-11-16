/*
 * simulations.hpp
 *
 * Sequences along a tree under GTR with continuous gamma rate variation (+GC), JC69 as a special case, and
 * random trees to simulate on. Used by the tests and the pipeline tools; not part of the p2v-bme command.
 */
#ifndef P2VBME_SIMULATIONS_HPP
#define P2VBME_SIMULATIONS_HPP

#include <stdint.h>

#include "inference.hpp"

namespace p2vbme {

/* GTR rates (AC, AG, AT, CG, CT, GT) and base frequencies (A, C, G, T); the rate matrix is normalised to one
 * expected substitution per unit time. alpha > 0: each site's rate is drawn from Gamma(alpha, 1 / alpha) (mean 1,
 * continuous gamma); alpha = 0: all sites have rate 1. */
struct SimulationModel {
    double rates[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double freqs[4] = {0.25, 0.25, 0.25, 0.25};
    double alpha = 0.0;
};

/* Transition probabilities P(t) = exp(Q t) of the model's normalised rate matrix: P[i][j] = Pr(j at time t | i). */
void transition_matrix(const SimulationModel &model, double t, double P[4][4]);

/* Random rooted binary tree on n >= 2 leaves (Tree layout: post-order, root 2n-2): pairs of subtrees are joined
 * uniformly at random; each branch length is drawn uniformly from [min_length, max_length] (the root's is 0). */
Tree random_tree(int32_t n, double min_length, double max_length, uint64_t seed);

/* Simulates n_sites aligned sites along the tree's branch lengths (expected substitutions per site): the root's
 * bases from the frequencies, then down every edge with P(rate x length). Names are t0 ... t{N-1}; no gaps. */
Alignment simulate_alignment(const Tree &tree, int32_t n_sites, const SimulationModel &model, uint64_t seed);

/* Sum of the branch lengths on the path between leaves i and j. */
double path_length(const Tree &tree, int32_t i, int32_t j);

} // namespace p2vbme

#endif // P2VBME_SIMULATIONS_HPP
