/*
 * test_distances.cpp
 *
 * The distance models of distances.hpp on the CPU (GTR+GC's own tests: test_eigen_dist.cpp), against references
 * that do not use the code under test:
 *   - consistency: on the expected pair counts of a model at distance t (L = 10^9 sites, from the simulator's
 *     transition probabilities), each model's estimator returns t when the data follow it;
 *   - one asymmetric count table, evaluated by an independent numpy script with the textbook formulas;
 *   - edge cases (no shared site, identical, saturated, undefined LogDet) and the model names.
 */
#include <math.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "distance_test_helpers.hpp"
#include "distances.hpp"
#include "simulations.hpp"

using namespace p2vbme;

namespace {

/* freqs: the alignment's base frequencies (F81, F84, TN93); uniform if not given */
template <class T> T distance(Model model, const PairCounts &c, const double *freqs = nullptr) {
    SubstitutionModel m;
    m.model = model;
    if (freqs)
        for (int i = 0; i < N_BASES; ++i)
            m.base_freqs[i] = freqs[i];
    return pair_distance<T>(distance_params<T>(m), c.n);
}

/* Each closed form is consistent under its model and under the models it contains. */
void expect_recovers(Model model, const SimulationModel &truth, double tol_double, double tol_float) {
    for (double t : TIMES) {
        const PairCounts c = expected_counts(truth, t);
        /* the model's own base frequencies stand for the alignment's */
        EXPECT_NEAR(distance<double>(model, c, truth.freqs), t, tol_double * t) << model_name(model) << " t=" << t;
        EXPECT_NEAR(distance<float>(model, c, truth.freqs), t, tol_float * t)
            << model_name(model) << " t=" << t << " (float)";
    }
}

} // namespace

TEST(Distances, ModelNames) {
    EXPECT_EQ(parse_model("JC69"), Model::JC69);
    EXPECT_EQ(parse_model("jc"), Model::JC69);
    EXPECT_EQ(parse_model("k2p"), Model::K80);
    EXPECT_EQ(parse_model("K80"), Model::K80);
    EXPECT_EQ(parse_model("f81"), Model::F81);
    EXPECT_EQ(parse_model("F84"), Model::F84);
    EXPECT_EQ(parse_model("tn"), Model::TN93);
    EXPECT_EQ(parse_model("TN93"), Model::TN93);
    EXPECT_EQ(parse_model("logdet"), Model::LogDet);
    EXPECT_EQ(parse_model("gtr+gc"), Model::GTR_GC);
    for (Model m : {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC})
        EXPECT_EQ(parse_model(model_name(m)), m);
    EXPECT_THROW(parse_model("GTR+G"), std::runtime_error);
    EXPECT_THROW(parse_model("GTR"), std::runtime_error);
    EXPECT_THROW(parse_model("HKY"), std::runtime_error);
}

// The float JC69 is the paper's search-core expression, value for value, wherever the core did not cap p (p <= 0.74).
// Above, the logarithm's argument is floored at 1e-4, as for the other models.
TEST(Distances, Jc69IsTheCoreFormula) {
    EXPECT_EQ(jc69_distance<float>(0, 0), (float)NO_OVERLAP_DISTANCE);
    for (int32_t nv : {1, 7, 100, 1500, 50000})
        for (int32_t nm = 0; nm <= nv; nm += (nv > 100 ? nv / 97 + 1 : 1)) {
            const float p = (float)(nv - nm) / (float)nv;
            const float d = jc69_distance<float>(nv, nm);
            if (p <= 0.74f) {
                const float core = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                EXPECT_EQ(d, (core > 0.0f) ? core : 0.0f) << nv << " " << nm;
            } else {
                EXPECT_EQ(d, -0.75f * logf(std::max(1.0f - (4.0f / 3.0f) * p, 1e-4f))) << nv << " " << nm;
            }
        }
}

TEST(Distances, Jc69Consistent) {
    for (Model m : {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet})
        expect_recovers(m, model_jc(), 1e-6, 2e-4);
}

TEST(Distances, K80Consistent) {
    for (Model m : {Model::K80, Model::F84, Model::TN93, Model::LogDet})
        expect_recovers(m, model_k80(), 1e-6, 2e-4);
}

TEST(Distances, F81Consistent) {
    for (Model m : {Model::F81, Model::F84, Model::TN93})
        expect_recovers(m, model_f81(), 1e-6, 2e-4);
}

TEST(Distances, F84Consistent) {
    for (Model m : {Model::F84, Model::TN93})
        expect_recovers(m, model_f84(), 1e-6, 2e-4);
}

TEST(Distances, Tn93Consistent) { expect_recovers(Model::TN93, model_tn93(), 1e-6, 2e-4); }

/* LogDet = t * trace(-Q) / 4 under any GTR model (= t under JC69 and K80). */
TEST(Distances, LogDetUnderGtr) {
    for (const SimulationModel &m : {model_f81(), model_tn93(), model_gtr()}) {
        double Q[N_BASES][N_BASES];
        rate_matrix(m, Q);
        const double scale = -(Q[0][0] + Q[1][1] + Q[2][2] + Q[3][3]) / 4.0;
        for (double t : TIMES) {
            const PairCounts c = expected_counts(m, t);
            EXPECT_NEAR(distance<double>(Model::LogDet, c), scale * t, 1e-6 * t);
            EXPECT_NEAR(distance<float>(Model::LogDet, c), scale * t, 3e-4 * t);
        }
    }
}

/* One asymmetric table, against values from an independent numpy evaluation of ape's dist.dna formulas (F81, F84,
 * TN93 with the alignment frequencies 0.28, 0.22, 0.26, 0.24). */
TEST(Distances, AsymmetricTable) {
    const PairCounts c = {{300, 10, 25, 8, 12, 250, 6, 30, 20, 5, 280, 9, 7, 35, 11, 290}};
    EXPECT_NEAR(distance<double>(Model::JC69, c), 0.15144523277580962, 1e-12);
    EXPECT_NEAR(distance<double>(Model::K80, c), 0.15310763367740979, 1e-12);
    const double freqs[N_BASES] = {0.28, 0.22, 0.26, 0.24};
    EXPECT_NEAR(distance<double>(Model::F81, c, freqs), 0.15148903003135755, 1e-12);
    EXPECT_NEAR(distance<double>(Model::F84, c, freqs), 0.15313574509110148, 1e-12);
    EXPECT_NEAR(distance<double>(Model::TN93, c, freqs), 0.15412841263487045, 1e-12);
    EXPECT_NEAR(distance<double>(Model::LogDet, c), 0.1542839532092013, 1e-12);
}

TEST(Distances, EdgeCases) {
    const PairCounts none = {{0}};
    const PairCounts same = {{40, 0, 0, 0, 0, 10, 0, 0, 0, 0, 30, 0, 0, 0, 0, 20}};
    const PairCounts poly_a = {{50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    /* every site differs: saturated */
    const PairCounts all_diff = {{0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0}};
    /* T missing from sequence b: LogDet undefined */
    const PairCounts no_t = {{30, 2, 3, 0, 1, 25, 1, 0, 2, 1, 28, 0, 3, 2, 1, 0}};
    for (Model m : {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC}) {
        EXPECT_EQ(distance<float>(m, none), 2.0f) << model_name(m);
        EXPECT_EQ(distance<double>(m, same), 0.0) << model_name(m);
        EXPECT_EQ(distance<double>(m, poly_a), 0.0) << model_name(m);
        for (const PairCounts &c : {all_diff, no_t}) {
            const double d = distance<double>(m, c);
            const float f = distance<float>(m, c);
            EXPECT_TRUE(d >= 0.0 && d <= 10.0) << model_name(m) << " " << d;
            EXPECT_TRUE(f >= 0.0f && f <= 10.0f) << model_name(m) << " " << f;
        }
    }
    EXPECT_EQ(distance<double>(Model::LogDet, no_t), 10.0);
    EXPECT_EQ(distance<double>(Model::LogDet, all_diff), 10.0); /* det J < 0 */
}

/* The alignment's base frequencies: each base's share of all valid sites ("AACG" and "T-GA": 3 A, 1 C, 2 G, 1 T). */
TEST(Distances, BaseFrequencies) {
    std::vector<uint64_t> packed(2 * N_PLANES, 0); /* 2 sequences, 1 word, planes A C G T valid */
    auto set = [&](int seq, int site, int base) {
        packed[seq * N_PLANES + base] |= 1ULL << site;
        packed[seq * N_PLANES + VALID_PLANE] |= 1ULL << site;
    };
    set(0, 0, 0), set(0, 1, 0), set(0, 2, 1), set(0, 3, 2);
    set(1, 0, 3), set(1, 2, 2), set(1, 3, 0);
    double f[N_BASES];
    base_frequencies(packed.data(), 2, 1, f);
    EXPECT_DOUBLE_EQ(f[0], 3.0 / 7);
    EXPECT_DOUBLE_EQ(f[1], 1.0 / 7);
    EXPECT_DOUBLE_EQ(f[2], 2.0 / 7);
    EXPECT_DOUBLE_EQ(f[3], 1.0 / 7);
    std::vector<uint64_t> empty(N_PLANES, 0);
    base_frequencies(empty.data(), 1, 1, f);
    EXPECT_EQ(f[0], 0.25);
}
