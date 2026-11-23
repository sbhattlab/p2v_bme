/*
 * test_gtr_fit.cpp
 *
 * The GTR+GC fit (gtr_fit.hpp): it recovers the true parameters on the paper's simulated pairs and finds them exactly
 * in noise-free tables; tables with too few sites are not fitted; the pairs it draws; the fit of an alignment.
 */
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "distance_test_helpers.hpp"
#include "gtr_fit.hpp"
#include "random.hpp"
#include "simulations.hpp"

using namespace p2vbme;

namespace {

// ---- Pairs simulated site by site ----------------------------------------------------------------------------
// The paper's check of its fit: each site of a pair evolves as a continuous-time Markov chain, jump by jump, at a
// gamma-distributed rate, so the simulation shares no formula with the fit.
// Ported with its random stream, so the tables are the ones its fit was run on (TABLES_HASH checks it).

struct JumpSimulator {
    uint64_t state;

    // Uniform in (0, 1).
    double unif() { return ((double)(splitmix64(state) >> 11) + 0.5) / 9007199254740992.0; }

    // Standard normal (Box-Muller), the two uniforms in the order the paper's build drew them.
    double normal() {
        const double u1 = unif();
        const double u2 = unif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    // Gamma with shape a and scale 1 (Marsaglia-Tsang); a < 1 by the boost Gamma(a + 1) U^(1 / a).
    double gamma_shape(double a) {
        if (a < 1.0) {
            const double g = gamma_shape(a + 1.0);
            return g * std::pow(unif(), 1.0 / a);
        }
        const double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            const double x = normal();
            double v = 1.0 + c * x;
            if (v <= 0.0)
                continue;
            v = v * v * v;
            if (std::log(unif()) < 0.5 * x * x + d - d * v + d * std::log(v))
                return d * v;
        }
    }

    // Index k with probability w[k] / sum(w).
    int draw(const double *w, int n) {
        double tot = 0.0, u;
        int k = 0;
        for (int i = 0; i < n; ++i)
            tot += w[i];
        u = unif() * tot;
        for (; k < n - 1; ++k) {
            if (u < w[k])
                break;
            u -= w[k];
        }
        return k;
    }

    // Simulates n_pairs tables of n_sites sites, each pair at a distance log-uniform in [d_lo, d_hi], under GTR with
    // exchangeabilities s and frequencies pi; alpha <= 0 means equal rates across sites.
    std::vector<PairCounts> simulate(const double s[N_GTR_RATES], const double pi[N_BASES], double alpha, double d_lo,
                                     double d_hi, int32_t n_pairs, int32_t n_sites) {
        static const int si[N_GTR_RATES] = {0, 0, 0, 1, 1, 2}, sj[N_GTR_RATES] = {1, 2, 3, 2, 3, 3};
        double q[N_BASES][N_BASES] = {{0}}, out[N_BASES] = {0}, mu = 0.0;
        for (int k = 0; k < N_GTR_RATES; ++k) {
            q[si[k]][sj[k]] = s[k] * pi[sj[k]];
            q[sj[k]][si[k]] = s[k] * pi[si[k]];
        }
        for (int i = 0; i < N_BASES; ++i) {
            for (int j = 0; j < N_BASES; ++j)
                out[i] += q[i][j];
            mu += pi[i] * out[i];
        }
        std::vector<PairCounts> tables(n_pairs, PairCounts{});
        for (int32_t p = 0; p < n_pairs; ++p) {
            const double d = d_lo * std::pow(d_hi / d_lo, unif());
            for (int32_t site = 0; site < n_sites; ++site) {
                const double r = (alpha > 0.0) ? gamma_shape(alpha) / alpha : 1.0;
                double t = d * r / mu;
                const int a = draw(pi, N_BASES);
                int b = a;
                for (;;) {
                    t -= -std::log(unif()) / out[b];
                    if (t < 0.0)
                        break;
                    b = draw(q[b], N_BASES);
                }
                tables[p].n[a * N_BASES + b]++;
            }
        }
        return tables;
    }
};

// FNV-1a hash of the tables' counts, in order.
uint64_t tables_hash(const std::vector<PairCounts> &tables) {
    uint64_t h = 1469598103934665603ULL;
    for (const PairCounts &t : tables)
        for (uint32_t x : t.n) {
            h ^= x;
            h *= 1099511628211ULL;
        }
    return h;
}

// ---- The paper's check of its fit -------------------------------------------------------------------------

const double S_TRUE[N_GTR_RATES] = {0.5, 2.0, 0.5, 0.5, 2.0, 0.5}, PI_TRUE[N_BASES] = {0.30, 0.20, 0.27, 0.23};
const double S_FLAT[N_GTR_RATES] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, PI_FLAT[N_BASES] = {0.25, 0.25, 0.25, 0.25};
constexpr int32_t N_PAIRS = 4000, N_SITES = 2000;

// A case: the simulation (alpha <= 0: equal rates; seed: the stream's), the hash of its tables (those the paper's code
// was checked on, one seed per case), and what recovering the truth means: |ln(alpha_hat / alpha)| below alpha_tol
// (alpha > 0), or alpha_hat above alpha_min (equal rates).
struct Case {
    const char *name;
    const double *s, *pi;
    double alpha, d_hi;
    uint64_t seed, tables_hash;
    double alpha_tol, alpha_min;
};

const Case CASES[] = {
    {"alpha 0.5", S_TRUE, PI_TRUE, 0.5, 1.5, 1, 0x91e6582054af53b1ULL, 0.08, 0.0},
    {"alpha 1.5", S_TRUE, PI_TRUE, 1.5, 1.5, 2, 0x1a62f75ca0578527ULL, 0.08, 0.0},
    {"alpha 8", S_TRUE, PI_TRUE, 8.0, 1.5, 3, 0x370dc62d6f0f9505ULL, 0.25, 0.0},
    // without rate variation, alpha goes to its upper end
    {"no rate variation", S_TRUE, PI_TRUE, -1.0, 1.0, 4, 0x19aacdcc0ff51e19ULL, 0.0, 15.0},
    // equal rates and frequencies: pairs do not determine alpha, and the penalty selects equal rates across sites
    {"equal rates and frequencies", S_FLAT, PI_FLAT, -1.0, 1.0, 5, 0x5b8e9cc7d236a5c5ULL, 0.0, 100.0},
};

} // namespace

/* On the pairs of the paper's check, the fit recovers the simulated parameters as that check asks: alpha within its
 * tolerance (or large without rate variation), every exchangeability within 5%. */
TEST(GtrFit, RecoversTheTruth) {
    for (const Case &c : CASES) {
        SCOPED_TRACE(c.name);
        JumpSimulator sim{c.seed};
        const std::vector<PairCounts> tables = sim.simulate(c.s, c.pi, c.alpha, 0.05, c.d_hi, N_PAIRS, N_SITES);
        ASSERT_EQ(tables_hash(tables), c.tables_hash) << "not the tables of the paper's check";

        const GtrFit fit = fit_gtr_gc_counts(tables);
        EXPECT_EQ(fit.n_pairs, N_PAIRS);
        EXPECT_TRUE(fit.converged);
        double rate_error = 0.0;
        for (int k = 0; k < N_GTR_RATES; ++k)
            rate_error = std::max(rate_error, std::fabs(std::log(fit.rates[k] / c.s[k])));
        EXPECT_LT(rate_error, 0.05);
        if (c.alpha > 0.0)
            EXPECT_LT(std::fabs(std::log(fit.alpha / c.alpha)), c.alpha_tol);
        else
            EXPECT_GT(fit.alpha, c.alpha_min);
    }
}

/* In noise-free tables (the expected counts of each pair, distance_test_helpers.hpp) the fit finds the true alpha and
 * exchangeabilities, also at little divergence, where alpha is weakly determined beside the exchangeabilities, and
 * with more pairs than the start scan uses. */
TEST(GtrFit, FindsTheParametersOfNoiseFreeTables) {
    const SimulationModel m = model_gtr();
    double rate_sum = 0.0;
    for (double r : m.rates)
        rate_sum += r;
    const int32_t n_pairs = 25000;
    const double n_sites = 1e5;
    for (double alpha : {0.5, 2.0})
        for (auto [d_lo, d_hi] : {std::pair<double, double>{0.01, 0.1}, {0.02, 0.3}}) {
            SCOPED_TRACE("alpha " + std::to_string(alpha) + ", d in [" + std::to_string(d_lo) + ", " +
                         std::to_string(d_hi) + "]");
            std::vector<PairCounts> tables(n_pairs);
            for (int32_t p = 0; p < n_pairs; ++p) // distances log-spaced, in increasing order
                tables[p] =
                    expected_counts_gamma(m, d_lo * std::pow(d_hi / d_lo, (double)p / (n_pairs - 1)), alpha, n_sites);
            const GtrFit fit = fit_gtr_gc_counts(tables);
            EXPECT_TRUE(fit.converged);
            EXPECT_LT(std::fabs(std::log(fit.alpha / alpha)), 5e-3) << "alpha_hat " << fit.alpha;
            for (int k = 0; k < N_GTR_RATES; ++k)
                EXPECT_LT(std::fabs(std::log(fit.rates[k] / (m.rates[k] * N_GTR_RATES / rate_sum))), 1e-3)
                    << "rate " << k;
        }
}

/* Tables with fewer than GTR_FIT_MIN_SITES sites are left out; with none left, nothing is fitted. */
TEST(GtrFit, TooFewSitesNotFitted) {
    std::vector<PairCounts> tables(10, PairCounts{});
    tables[0].n[0] = 10; // 15 sites, the others none
    tables[0].n[5] = 5;
    for (const std::vector<PairCounts> &t : {tables, std::vector<PairCounts>{}}) {
        const GtrFit fit = fit_gtr_gc_counts(t);
        EXPECT_EQ(fit.n_pairs, 0);
        EXPECT_FALSE(fit.converged);
        EXPECT_EQ(fit.alpha, 1.0);
        for (double r : fit.rates)
            EXPECT_EQ(r, 1.0);
    }
}

/* More pairs than asked for: the paper's draw, pair for pair from its seed (its first pairs at the seed of the
 * full alignment, 0xABC00000). */
TEST(GtrFit, DrawsThePapersPairs) {
    const std::vector<std::pair<int32_t, int32_t>> n1000 = {{192, 114}, {826, 963}, {102, 520}, {202, 111},
                                                            {661, 151}, {45, 658},  {935, 527}, {276, 524},
                                                            {935, 546}, {230, 853}, {241, 849}, {54, 159}};
    EXPECT_EQ(draw_fit_pairs(1000, 12, 0xABC00000ULL), n1000);
    // 5 sequences have 10 pairs, so 9 draws still follow the paper (pairs repeat)
    const std::vector<std::pair<int32_t, int32_t>> n5 = {{2, 4}, {1, 0}, {2, 0}, {2, 0}, {1, 2},
                                                         {0, 1}, {0, 4}, {1, 2}, {0, 3}};
    EXPECT_EQ(draw_fit_pairs(5, 9, 0xABC00000ULL), n5);
}

/* At most as many pairs as asked for: every pair once, in an order that depends on the seed. */
TEST(GtrFit, EveryPairOnceWhenFewer) {
    for (int32_t n_pairs : {15, 100}) {
        const auto pairs = draw_fit_pairs(6, n_pairs, 7);
        ASSERT_EQ(pairs.size(), 15u);
        std::set<std::pair<int32_t, int32_t>> seen;
        for (auto [a, b] : pairs) {
            ASSERT_TRUE(a >= 0 && a < b && b < 6) << a << " " << b;
            seen.insert({a, b});
        }
        EXPECT_EQ(seen.size(), 15u);
    }
    EXPECT_NE(draw_fit_pairs(6, 15, 7), draw_fit_pairs(6, 15, 8));
    EXPECT_TRUE(draw_fit_pairs(1, 10, 7).empty());
}

/* The fit of an alignment is the fit of the counts of the pairs it draws, the same at any number of threads, and
 * needs at least one pair. */
TEST(GtrFit, FitsTheCountsOfTheDrawnPairs) {
    SimulationModel m;
    const double rates[N_GTR_RATES] = {1.0, 3.0, 0.7, 1.2, 3.5, 1.0}, freqs[N_BASES] = {0.3, 0.2, 0.25, 0.25};
    for (int k = 0; k < N_GTR_RATES; ++k)
        m.rates[k] = rates[k];
    for (int i = 0; i < N_BASES; ++i)
        m.freqs[i] = freqs[i];
    m.alpha = 0.7;
    const Alignment aln = simulate_alignment(random_tree(40, 0.01, 0.1, 3), 600, m, 4);
    const int32_t n_pairs = 500; // fewer than the 780 pairs: the paper's draw
    const uint64_t seed = 11;

    const auto pairs = draw_fit_pairs(aln.n_taxa, n_pairs, seed);
    const size_t row_words = (size_t)N_PLANES * aln.n_words;
    std::vector<PairCounts> tables;
    for (auto [a, b] : pairs)
        tables.push_back(count_pair(&aln.packed[a * row_words], &aln.packed[b * row_words], aln.n_words));
    const GtrFit expected = fit_gtr_gc_counts(tables);
    ASSERT_EQ(expected.n_pairs, n_pairs);

    const int threads = omp_get_max_threads();
    for (int t : {1, 3}) {
        omp_set_num_threads(t);
        const GtrFit fit = fit_gtr_gc(aln, n_pairs, seed);
        EXPECT_EQ(fit.alpha, expected.alpha) << t << " threads";
        for (int k = 0; k < N_GTR_RATES; ++k)
            EXPECT_EQ(fit.rates[k], expected.rates[k]) << t << " threads, rate " << k;
        EXPECT_EQ(fit.log_lik, expected.log_lik) << t << " threads";
        EXPECT_GT(fit.time, 0.0);
    }
    omp_set_num_threads(threads);
    EXPECT_THROW(fit_gtr_gc(aln, 0, seed), std::invalid_argument);
}
