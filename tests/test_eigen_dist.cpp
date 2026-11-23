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
