/*
 * test_distances.cpp
 *
 * The distance models on the CPU (substitution_models.hpp, eigen_dist.hpp), against references that do not use
 * the code under test:
 *   - consistency: on the expected pair counts of a model at distance t (L = 10^9 sites, from the simulator's
 *     transition probabilities), each model's estimator returns t when the data follow it;
 *   - one asymmetric count table, evaluated by an independent numpy script with the textbook formulas;
 *   - edge cases (no shared site, identical, saturated, undefined LogDet) and the model names.
 */
#include <math.h>

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "eigen_dist.hpp"
#include "simulations.hpp"
#include "substitution_models.hpp"

using namespace p2vbme;

namespace {

/* Expected counts of L sites at distance t under a model without rate variation: n_ij = round(L pi_i P_ij(t)). */
PairCounts expected_counts(const SimulationModel &m, double t, double L = 1e9) {
    double P[N_BASES][N_BASES];
    transition_matrix(m, t, P);
    double fsum = 0.0;
    for (int i = 0; i < N_BASES; ++i)
        fsum += m.freqs[i];
    PairCounts c;
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j)
            c.n[N_BASES * i + j] = (uint32_t)llround(L * m.freqs[i] / fsum * P[i][j]);
    return c;
}

/* Rate matrix Q of a GTR model, normalised to one substitution per unit time (written here, not taken from the
 * code under test). */
void rate_matrix(const SimulationModel &m, double Q[N_BASES][N_BASES]) {
    const int pairs[N_GTR_RATES][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    double s[N_BASES][N_BASES] = {}, pi[N_BASES], fsum = 0.0;
    for (int i = 0; i < N_BASES; ++i)
        fsum += m.freqs[i];
    for (int i = 0; i < N_BASES; ++i)
        pi[i] = m.freqs[i] / fsum;
    for (int k = 0; k < N_GTR_RATES; ++k)
        s[pairs[k][0]][pairs[k][1]] = s[pairs[k][1]][pairs[k][0]] = m.rates[k];
    double mu = 0.0;
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j)
            if (i != j)
                mu += pi[i] * s[i][j] * pi[j];
    for (int i = 0; i < N_BASES; ++i) {
        double row = 0.0;
        for (int j = 0; j < N_BASES; ++j)
            if (i != j) {
                Q[i][j] = s[i][j] * pi[j] / mu;
                row += Q[i][j];
            }
        Q[i][i] = -row;
    }
}

/* Jacobi eigen-decomposition of a symmetric 4 x 4 matrix: A = V diag(d) V^T. */
void jacobi(double A[4][4], double V[4][4], double d[4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            V[i][j] = (i == j);
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q)
                off += A[p][q] * A[p][q];
        if (off < 1e-30)
            break;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) {
                if (fabs(A[p][q]) < 1e-300)
                    continue;
                const double th = 0.5 * atan2(2 * A[p][q], A[q][q] - A[p][p]);
                const double c = cos(th), s = sin(th);
                for (int k = 0; k < 4; ++k) {
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; ++k) {
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 4; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 4; ++i)
        d[i] = A[i][i];
}

/* Expected counts under GTR with continuous gamma rates (shape alpha): P(t) = E_r[exp(Q r t)] =
 * Pi^(-1/2) V diag((1 - lambda_k t / alpha)^(-alpha)) V^T Pi^(1/2), from the symmetric form of Q. */
PairCounts expected_counts_gamma(const SimulationModel &m, double t, double alpha, double L = 1e9) {
    double Q[N_BASES][N_BASES], pi[N_BASES], fsum = 0.0;
    rate_matrix(m, Q);
    for (int i = 0; i < N_BASES; ++i)
        fsum += m.freqs[i];
    for (int i = 0; i < N_BASES; ++i)
        pi[i] = m.freqs[i] / fsum;
    double S[N_BASES][N_BASES], V[N_BASES][N_BASES], lam[N_BASES];
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j)
            S[i][j] = sqrt(pi[i]) * Q[i][j] / sqrt(pi[j]);
    jacobi(S, V, lam);
    PairCounts c;
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j) {
            double p = 0.0;
            for (int k = 0; k < N_BASES; ++k)
                p += V[i][k] * pow(1.0 - lam[k] * t / alpha, -alpha) * V[j][k];
            p *= sqrt(pi[j] / pi[i]);
            c.n[N_BASES * i + j] = (uint32_t)llround(L * pi[i] * p);
        }
    return c;
}

/* freqs: the alignment's base frequencies (F81, F84, TN93); uniform if not given */
template <class T> T distance(Model model, const PairCounts &c, const double *freqs = nullptr) {
    SubstitutionModel m;
    m.model = model;
    if (freqs)
        for (int i = 0; i < N_BASES; ++i)
            m.base_freqs[i] = freqs[i];
    return pair_distance<T>(distance_params<T>(m, EigenSettings()), c.n);
}

SimulationModel model_jc() { return SimulationModel(); }
SimulationModel model_k80() { /* kappa = 4 */
    SimulationModel m;
    m.rates[1] = m.rates[4] = 4.0;
    return m;
}
SimulationModel model_f81() {
    SimulationModel m;
    const double f[N_BASES] = {0.1, 0.2, 0.3, 0.4};
    for (int i = 0; i < N_BASES; ++i)
        m.freqs[i] = f[i];
    return m;
}
SimulationModel model_f84() { /* F84: s_AG = 1 + K / pi_R, s_CT = 1 + K / pi_Y, K = 2 */
    SimulationModel m = model_f81();
    m.rates[1] = 1.0 + 2.0 / (0.1 + 0.3);
    m.rates[4] = 1.0 + 2.0 / (0.2 + 0.4);
    return m;
}
SimulationModel model_tn93() {
    SimulationModel m = model_f81();
    m.rates[1] = 6.0;
    m.rates[4] = 2.5;
    return m;
}
SimulationModel model_gtr() {
    SimulationModel m;
    const double r[N_GTR_RATES] = {0.5, 2.0, 0.8, 1.2, 3.0, 0.7}, f[N_BASES] = {0.35, 0.15, 0.2, 0.3};
    for (int k = 0; k < N_GTR_RATES; ++k)
        m.rates[k] = r[k];
    for (int i = 0; i < N_BASES; ++i)
        m.freqs[i] = f[i];
    return m;
}

const double TIMES[] = {0.02, 0.1, 0.4, 1.0};

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
    EXPECT_EQ(parse_estimator("svd"), GtrEstimator::Svd);
    EXPECT_EQ(parse_saturation("pair"), Saturation::Pair);
    EXPECT_THROW(parse_estimator("ols"), std::runtime_error);
}

/* The float JC69 is the search core's expression, value for value. */
TEST(Distances, Jc69IsTheCoreFormula) {
    for (int32_t nv : {0, 1, 7, 100, 1500, 50000})
        for (int32_t nm = 0; nm <= nv; nm += (nv > 100 ? nv / 97 + 1 : 1)) {
            float core = 1.5f;
            if (nv > 0) {
                float p = (float)(nv - nm) / (float)nv;
                if (p >= 0.74f)
                    p = 0.74f;
                const float d = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                core = (d > 0.0f) ? d : 0.0f;
            }
            EXPECT_EQ(jc69_distance<float>(nv, nm), core) << nv << " " << nm;
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
    EXPECT_EQ(distance<float>(Model::JC69, none), 1.5f);
    for (Model m : {Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC}) {
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

/* ---- Eigenvalue GTR+GC ------------------------------------------------------------------------------------ */

namespace {

template <class T>
T gtr_distance(const PairCounts &c, const SimulationModel &m, double alpha, EigenSettings settings = EigenSettings()) {
    SubstitutionModel dm;
    dm.model = Model::GTR_GC;
    for (int k = 0; k < N_GTR_RATES; ++k)
        dm.gtr_rates[k] = m.rates[k];
    dm.gtr_alpha = alpha;
    return pair_distance<T>(distance_params<T>(dm, settings), c.n);
}

} // namespace

/* With the true parameters, SymWLS and SymSVD return t from the expected counts under GTR+GC: each mode gives
 * g(sigma_k) / |lambda_k| = t. */
TEST(EigenDist, GtrGcConsistent) {
    for (const SimulationModel &m : {model_jc(), model_tn93(), model_gtr()})
        for (double alpha : {0.3, 1.0, 5.0})
            for (double t : TIMES) {
                const PairCounts c = expected_counts_gamma(m, t, alpha);
                for (GtrEstimator e : {GtrEstimator::Wls, GtrEstimator::Svd}) {
                    EigenSettings s;
                    s.estimator = e;
                    EXPECT_NEAR(gtr_distance<double>(c, m, alpha, s), t, 1e-5 * t)
                        << estimator_name(e) << " alpha=" << alpha << " t=" << t;
                    EXPECT_NEAR(gtr_distance<float>(c, m, alpha, s), t, 3e-3 * t)
                        << estimator_name(e) << " alpha=" << alpha << " t=" << t << " (float)";
                }
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

/* The library's mode 0 is consistent with equal base frequencies (its symmetrised matrix has the singular values
 * of P(t) only when diag(F) commutes with it). */
TEST(EigenDist, OrigConsistentAtEqualFrequencies) {
    EigenSettings s;
    s.estimator = GtrEstimator::Orig;
    for (double alpha : {0.5, 2.0})
        for (double t : TIMES) {
            const PairCounts c = expected_counts_gamma(model_k80(), t, alpha);
            EXPECT_NEAR(gtr_distance<double>(c, model_k80(), alpha, s), t, 1e-5 * t);
            EXPECT_NEAR(gtr_distance<float>(c, model_k80(), alpha, s), t, 3e-3 * t);
        }
}

/* The settings change what they should: on exact data the reweighting does not matter; on a noisy pair it does;
 * `pair` saturation sends a pair with det K <= 0 to the cap, `direction` does not. */
TEST(EigenDist, Settings) {
    const SimulationModel m = model_gtr();
    const PairCounts exact = expected_counts_gamma(m, 0.4, 0.7);
    EigenSettings none;
    none.reweight = 0;
    EXPECT_NEAR(gtr_distance<double>(exact, m, 0.7, none), 0.4, 4e-6);
    const PairCounts noisy = {{120, 9, 30, 11, 7, 41, 6, 19, 28, 4, 63, 8, 12, 22, 9, 95}};
    EigenSettings three;
    three.reweight = 3;
    const double d0 = gtr_distance<double>(noisy, m, 0.7, none), d2 = gtr_distance<double>(noisy, m, 0.7);
    const double d3 = gtr_distance<double>(noisy, m, 0.7, three);
    EXPECT_NE(d0, d2);
    EXPECT_NE(d2, d3);
    EXPECT_GT(d0, 0.0);
    const PairCounts flipped = {{2, 40, 3, 5, 38, 4, 6, 2, 3, 5, 37, 3, 6, 3, 4, 39}}; /* A, C swapped: det K < 0 */
    EigenSettings pair;
    pair.saturation = Saturation::Pair;
    EXPECT_EQ(gtr_distance<double>(flipped, m, 0.7, pair), 10.0);
    EXPECT_LT(gtr_distance<double>(flipped, m, 0.7), 10.0);
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
