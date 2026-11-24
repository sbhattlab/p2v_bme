/*
 * test_eigen_dist.cpp
 *
 * The eigenvalue GTR+GC distance (eigen_dist.hpp) on the CPU, on the expected pair counts of GTR models with
 * continuous gamma rates at distance t (distance_test_helpers.hpp, without the code under test), and its limit without
 * rate variation (GTR).
 */
#include <gtest/gtest.h>

#include "distance_test_helpers.hpp"
#include "distances.hpp"
#include "simulations.hpp"

using namespace p2vbme;

namespace {

template <class T> T gtr_distance(const PairCounts &c, const SimulationModel &m, double alpha) {
    SubstitutionModel dm;
    dm.model = Model::GTR_GC;
    for (int k = 0; k < N_GTR_RATES; ++k)
        dm.rates[k] = m.rates[k];
    dm.alpha = alpha;
    return pair_distance<T>(distance_params<T>(dm), c.n);
}

// The distance with m's exchangeabilities, with continuous gamma rates of shape alpha or, with GAMMA false, without
// rate variation.
template <class T, bool GAMMA> T eigen(const PairCounts &c, const SimulationModel &m, double alpha = 1.0) {
    SubstitutionModel dm;
    for (int k = 0; k < N_GTR_RATES; ++k)
        dm.rates[k] = m.rates[k];
    dm.alpha = alpha;
    const DistanceParams<T> p = distance_params<T>(dm);
    return eigen_distance<T, GAMMA>(c.n, p.rates, p.alpha, p.inv_alpha);
}

} // namespace

/* With the true parameters, SymWLS returns t from the expected counts under GTR+GC: each mode gives
 * g(sigma_k) / |lambda_k| = t. */
TEST(EigenDist, GtrGcConsistent) {
    for (const SimulationModel &m : {model_jc(), model_tn93(), model_gtr()})
        for (double alpha : {0.3, 1.0, 5.0})
            for (double t : TIMES) {
                const PairCounts c = expected_counts_gamma(m, t, alpha);
                EXPECT_NEAR(gtr_distance<double>(c, m, alpha), t, 1e-5 * t) << "alpha=" << alpha << " t=" << t;
                EXPECT_NEAR(gtr_distance<float>(c, m, alpha), t, 3e-3 * t)
                    << "alpha=" << alpha << " t=" << t << " (float)";
            }
}

/* With the true exchangeabilities, the distance without rate variation returns t from the expected counts under GTR:
 * each mode gives -ln(sigma_k) / |lambda_k| = t. */
TEST(EigenDist, GtrConsistent) {
    for (const SimulationModel &m : {model_jc(), model_k80(), model_f81(), model_f84(), model_tn93(), model_gtr()})
        for (double t : TIMES) {
            const PairCounts c = expected_counts(m, t);
            EXPECT_NEAR((eigen<double, false>(c, m)), t, 1e-5 * t) << "t=" << t;
            EXPECT_NEAR((eigen<float, false>(c, m)), t, 3e-3 * t) << "t=" << t << " (float)";
        }
}

/* GTR is GTR+GC's limit alpha -> infinity, step by step: on counts whose modes disagree (gamma rates of shape 2, so
 * that the weights and the reweighting passes matter), GTR+GC at alpha = 1e8 gives GTR's distance. */
TEST(EigenDist, GtrIsTheLimitOfGtrGc) {
    const SimulationModel m = model_gtr();
    for (double t : TIMES) {
        const PairCounts c = expected_counts_gamma(m, t, 2.0);
        const double gtr = eigen<double, false>(c, m), limit = eigen<double, true>(c, m, 1e8);
        EXPECT_NEAR(gtr, limit, 1e-6 * limit) << "t=" << t;
        EXPECT_LT(gtr, t) << "t=" << t << ": without rate variation, the distance underestimates";
    }
}
