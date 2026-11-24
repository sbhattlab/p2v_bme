// Helpers shared by the tests of the distance formulas (test_distances, test_eigen_dist): the expected pair counts
// of a model at distance t, written here without the code under test, and the models and distances they use.
#ifndef P2VBME_TESTS_DISTANCE_TEST_HELPERS_HPP
#define P2VBME_TESTS_DISTANCE_TEST_HELPERS_HPP

#include <cstdint>
#include <math.h>

#include "core.hpp"
#include "distances.hpp"
#include "simulations.hpp"

namespace {

using namespace p2vbme;

/* Expected counts of L sites at distance t under a model without rate variation: n_ij = round(L pi_i P_ij(t)). */
inline PairCounts expected_counts(const SimulationModel &m, double t, double L = 1e9) {
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
inline void rate_matrix(const SimulationModel &m, double Q[N_BASES][N_BASES]) {
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
inline void jacobi(double A[4][4], double V[4][4], double d[4]) {
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
inline PairCounts expected_counts_gamma(const SimulationModel &m, double t, double alpha, double L = 1e9) {
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

inline SimulationModel model_jc() { return SimulationModel(); }
inline SimulationModel model_k80() { /* kappa = 4 */
    SimulationModel m;
    m.rates[1] = m.rates[4] = 4.0;
    return m;
}
inline SimulationModel model_f81() {
    SimulationModel m;
    const double f[N_BASES] = {0.1, 0.2, 0.3, 0.4};
    for (int i = 0; i < N_BASES; ++i)
        m.freqs[i] = f[i];
    return m;
}
inline SimulationModel model_f84() { /* F84: s_AG = 1 + K / pi_R, s_CT = 1 + K / pi_Y, K = 2 */
    SimulationModel m = model_f81();
    m.rates[1] = 1.0 + 2.0 / (0.1 + 0.3);
    m.rates[4] = 1.0 + 2.0 / (0.2 + 0.4);
    return m;
}
inline SimulationModel model_tn93() {
    SimulationModel m = model_f81();
    m.rates[1] = 6.0;
    m.rates[4] = 2.5;
    return m;
}
inline SimulationModel model_gtr() {
    SimulationModel m;
    const double r[N_GTR_RATES] = {0.5, 2.0, 0.8, 1.2, 3.0, 0.7}, f[N_BASES] = {0.35, 0.15, 0.2, 0.3};
    for (int k = 0; k < N_GTR_RATES; ++k)
        m.rates[k] = r[k];
    for (int i = 0; i < N_BASES; ++i)
        m.freqs[i] = f[i];
    return m;
}

const double TIMES[] = {0.02, 0.1, 0.4, 1.0};

} // namespace

#endif // P2VBME_TESTS_DISTANCE_TEST_HELPERS_HPP
