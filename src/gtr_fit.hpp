/*
 * gtr_fit.hpp
 *
 * The GTR+GC parameters of an alignment, fitted without a tree: the six exchangeabilities and the gamma shape alpha
 * that eigen_distance (eigen_dist.hpp) needs, by maximum composite likelihood over the base-count tables of random
 * sequence pairs.
 * The paper's estimator (its supplement's penalised composite likelihood), ported from its code with the same
 * arithmetic; the search over alpha differs (below).
 *
 * Model: a reversible GTR process with exchangeabilities S (mean 1) and base frequencies pi (pooled over the tables),
 * scaled to one expected substitution per unit distance, with Gamma(alpha, alpha) rates across sites.
 * At distance d a site shows bases (i, j) with probability
 *   F_ij(d) = pi_i pi_j + sqrt(pi_i pi_j) sum_k U_ik U_jk (1 - lambda_k d / alpha)^(-alpha),
 * where lambda_k < 0 and U_k are the three non-trivial eigenpairs of Pi^(1/2) Q Pi^(-1/2).
 * F is symmetric, so each table is folded to its ten cells i <= j.
 *
 * Estimator: each pair contributes the multinomial log-likelihood of its table at its own best distance (safeguarded
 * Newton on d in [GTR_FIT_D_MIN, GTR_FIT_D_MAX]).
 * The sum over pairs, minus GTR_FIT_RATE_VARIANCE_PENALTY / alpha, is maximised over the five free log exchangeability
 * ratios and log alpha.
 * Alpha is found by Brent's method on the profile (the objective maximised over the exchangeabilities by BFGS at each
 * alpha) to 0.1%, in a bracket from a scan over alpha.
 * The BFGS gradient is by central differences at fixed distances, exact to second order since every distance is at
 * its optimum (envelope theorem).
 * The paper's code searched all six parameters jointly instead: pairs determine alpha far less well than the
 * exchangeabilities, and on its own simulations of little divergence that search stopped up to ~40% short in alpha.
 *
 * The penalty: pairs cannot determine alpha when the three non-trivial eigenvalues coincide (equal exchangeabilities
 * and base frequencies, as under JC69 or F81), since the likelihood is then flat in alpha.
 * The penalty on the rate variance 1 / alpha is negligible beside the likelihood whenever alpha is determined, and
 * otherwise selects equal rates across sites (alpha at its upper bound).
 *
 * Double precision throughout, on the host.
 * In single precision the cells of close pairs lose up to ~40% to cancellation (d = 1e-7), and the objective (~1e8
 * for 100,000 pairs of 1,000 sites) can no longer resolve the search's steps.
 */
#ifndef P2VBME_GTR_FIT_HPP
#define P2VBME_GTR_FIT_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "core.hpp"
#include "distances.hpp"

namespace p2vbme {

// ---- The estimator's constants (the paper's values) ----------------------------------------------------

// Pairs with fewer shared valid sites are left out.
constexpr double GTR_FIT_MIN_SITES = 30.0;
// The range of each pair's distance.
constexpr double GTR_FIT_D_MIN = 1e-7;
constexpr double GTR_FIT_D_MAX = 10.0;
// The range of log alpha: alpha in [0.0498, 148.4]; the upper end means no rate variation across sites.
constexpr double GTR_FIT_LOG_ALPHA_MIN = -3.0;
constexpr double GTR_FIT_LOG_ALPHA_MAX = 5.0;
// The range of each log exchangeability ratio, log(S_k / S_GT).
constexpr double GTR_FIT_LOG_RATE_MAX = 10.0;
// The objective is the composite log-likelihood minus this / alpha.
constexpr double GTR_FIT_RATE_VARIANCE_PENALTY = 1.0;

// ---- The fit ---------------------------------------------------------------------------------------------

// The fitted parameters and how the fit went.
struct GtrFit {
    double rates[N_GTR_RATES] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // exchangeabilities AC, AG, AT, CG, CT, GT, mean 1
    double base_freqs[N_BASES] = {0.25, 0.25, 0.25, 0.25};      // pi, pooled over the pairs used (A, C, G, T)
    double alpha = 1.0;                                         // gamma shape of the rates across sites
    double log_lik = 0.0;                                       // composite log-likelihood at the optimum (no penalty)
    int32_t n_pairs = 0; // pairs used (>= GTR_FIT_MIN_SITES shared valid sites); 0: not fitted, defaults kept
    int32_t n_iterations =
        0;                  // BFGS iterations, summed over the searches of the exchangeabilities (the scan's left out)
    bool converged = false; // Brent's method and every BFGS search stopped on their own criteria
    double time = 0.0;      // seconds: the pair counts and the fit (fit_gtr_gc only)
};

// Fits GTR+GC to pair count tables: tables -> (rates, alpha), with the pooled base frequencies.
// Tables with fewer than GTR_FIT_MIN_SITES sites are left out; if none is left, the defaults (equal rates, alpha = 1)
// stay, with n_pairs = 0.
// The start scan uses the first GTR_FIT_START_PAIRS tables (gtr_fit.cpp), so their order should be random.
// The result does not depend on the number of threads: the pairs' terms are added in index order.
GtrFit fit_gtr_gc_counts(const std::vector<PairCounts> &tables);

// Draws the pairs of the fit: (n_taxa, n_pairs, seed) -> pairs (a, b), a != b.
// More than n_pairs pairs in all: n_pairs uniform draws with replacement, the paper's draw (a uniform among all, then
// b uniform among the others).
// Otherwise every pair once, in random order.
// Fewer than 2 sequences: no pair.
std::vector<std::pair<int32_t, int32_t>> draw_fit_pairs(int32_t n_taxa, int32_t n_pairs, uint64_t seed);

// Fits GTR+GC to an alignment: (aln, n_pairs, seed) -> (rates, alpha), from the counts of the pairs draw_fit_pairs
// draws.
// Throws std::invalid_argument if n_pairs < 1.
GtrFit fit_gtr_gc(const Alignment &aln, int32_t n_pairs, uint64_t seed);

} // namespace p2vbme

#endif // P2VBME_GTR_FIT_HPP
