/*
 * test_eigen_dist.cpp
 *
 * The eigenvalue GTR+GC distance (eigen_dist.hpp) on the CPU, on the expected pair counts of GTR models with
 * continuous gamma rates at distance t (distance_test_helpers.hpp, without the code under test).
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

// Only the ratios of the exchangeabilities matter: scaling them all gives the same distance.
// Before they were scaled to mean 1, a scale of 1e-25 skipped the rate matrix's normalisation (distance about 500x).
TEST(EigenDist, GtrGcScaleFree) {
    const SimulationModel m = model_gtr();
    for (double t : TIMES) {
        const PairCounts c = expected_counts_gamma(m, t, 1.0);
        const double d = gtr_distance<double>(c, m, 1.0);
        const float f = gtr_distance<float>(c, m, 1.0);
        for (double scale : {1e-25, 1e-6, 1e6, 1e25}) {
            SimulationModel scaled = m;
            for (int k = 0; k < N_GTR_RATES; ++k)
                scaled.rates[k] *= scale;
            EXPECT_NEAR(gtr_distance<double>(c, scaled, 1.0), d, 1e-12 * d) << "t=" << t << " scale=" << scale;
            EXPECT_NEAR(gtr_distance<float>(c, scaled, 1.0), f, 1e-6f * f) << "t=" << t << " scale=" << scale;
        }
    }
}
