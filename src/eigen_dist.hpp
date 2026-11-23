/*
 * eigen_dist.hpp
 *
 * The eigenvalue GTR+GC distance (Penn et al. 2025, Bull. Math. Biol. 87, 32, doi:10.1007/s11538-024-01403-z) with
 * given GTR exchangeabilities and gamma shape alpha (eigen_distance, which pair_distance in distances.hpp calls), and
 * the GTR rate matrix it needs. Per-pair code, inline and __host__ __device__, templated on the floating-point type.
 *
 * For a pair (a, b) with pair counts n: K = (n + 1e-5) / (L + 16e-5), F^a and F^b its row and column sums, and
 * sigma_1 >= sigma_2 >= sigma_3 the non-trivial singular values of
 *   K~ = diag(F^a)^(-1/2) K diag(F^b)^(-1/2) - sqrt(F^a) sqrt(F^b)^T,
 * clipped to [1e-4, 1). The rate matrix uses pair frequencies xi_i = sqrt(F^a_i F^b_i) / sum_j sqrt(F^a_j F^b_j);
 * lambda_k < 0 are its non-zero eigenvalues, by increasing magnitude. Each mode estimates the distance by
 * g(sigma_k) / |lambda_k|, g(sigma) = alpha (sigma^(-1/alpha) - 1). SymWLS (the distance of the paper's supplement)
 * combines the three modes by weighted least squares,
 *   d = sum_k w_k |lambda_k| g(sigma_k) / sum_k w_k lambda_k^2,  w_k = s_k^(2 + 2/alpha) / (1 - s_k^2 + 1e-4),
 * with s_k = sigma_k, then EIGEN_REWEIGHT_PASSES refits with s_k = (1 + |lambda_k| d / alpha)^(-alpha).
 * Saturation: a saturated mode (sigma at 1e-4) gets almost no weight in the first fit, but the reweighting passes
 * weight each mode by its model-implied s_k, so a fully saturated direction can still pull the distance to the cap (the
 * paper's GPU implementation behaves the same). pair_distance (distances.hpp) gives the pairs without a shared valid
 * site and the identical ones their values, and caps the distance at MAX_DISTANCE.
 *
 * The paper's GPU implementation, ported line by line (bitwise equal when both are compiled without FMA contraction,
 * for exchangeabilities with mean 1 as the paper's fit returns them; others are scaled to mean 1 first).
 * Not ported: the paper's C distance library's projection onto the bases present when a base is missing (its "reduced
 * alphabet"); here a base absent from both sequences gives a pseudocount-only row of K~ (singular value 1/4), which
 * reads as a strongly diverged mode, as in the paper's GPU implementation.
 */
#ifndef P2VBME_EIGEN_DIST_HPP
#define P2VBME_EIGEN_DIST_HPP

#include <stdint.h>

#include <cmath>

#include "config.hpp"
#include "core.hpp"
#include "host_device.hpp"

namespace p2vbme {

// The pseudocount added to each of the 16 counts, and the range of the singular values.
constexpr double EIGEN_PSEUDOCOUNT = 1e-5;
constexpr double EIGEN_SIGMA_MIN = 1e-4;
constexpr double EIGEN_SIGMA_MAX = 0.999999;
// ln(EIGEN_SIGMA_MIN) = ln(1e-4) = -4 ln(10): the floor of ln(mu_k) in the reweighting passes.
// The float path keeps the paper's GPU literal, one float step above this value rounded to float, so that it matches
// the paper's code bit for bit.
constexpr double LOG_EIGEN_SIGMA_MIN = -9.2103403719761836;
constexpr float LOG_EIGEN_SIGMA_MIN_FLOAT = -9.21034f;
// SymWLS refits after the first fit, each with the weights at the fitted singular values (the manuscript's two): they
// reduce the first fit's downward bias, since weights at the observed values favour the modes that look least diverged.
constexpr int32_t EIGEN_REWEIGHT_PASSES = 2;

// ---- GTR rate matrix -------------------------------------------------------------------------------------

// Symmetric form R = Pi^(1/2) Q Pi^(-1/2) of the GTR rate matrix Q with exchangeabilities s (AC, AG, AT, CG, CT, GT)
// and stationary frequencies xi, normalised to one expected substitution per unit time (mean rate sum_i xi_i sum_j s_ij
// xi_j = 1).
// R has the eigenvalues of Q.
template <class T>
P2V_HD inline void gtr_symmetric_rate_matrix(const T s[N_GTR_RATES], const T xi[N_BASES], T R[N_BASES * N_BASES]) {
    const T S[N_BASES * N_BASES] = {T(0), s[0], s[1], s[2], s[0], T(0), s[3], s[4],
                                    s[1], s[3], T(0), s[5], s[2], s[4], s[5], T(0)};
    T row_sum[N_BASES];
    T mu = T(0);
    for (int i = 0; i < N_BASES; ++i) {
        T rs = T(0);
        for (int j = 0; j < N_BASES; ++j)
            rs += S[i * N_BASES + j] * xi[j];
        row_sum[i] = rs;
        mu += xi[i] * rs;
    }
    const T inv_mu = (mu > T(1e-20)) ? (T(1) / mu) : T(1);
    T sxi[N_BASES];
    for (int i = 0; i < N_BASES; ++i)
        sxi[i] = std::sqrt(xi[i]);
    for (int i = 0; i < N_BASES; ++i) {
        for (int j = 0; j < N_BASES; ++j) {
            if (i == j) {
                R[i * N_BASES + i] = -row_sum[i] * inv_mu;
            } else {
                R[i * N_BASES + j] = S[i * N_BASES + j] * sxi[i] * sxi[j] * inv_mu;
            }
        }
    }
}

// ---- 4 x 4 symmetric eigenvalues ------------------------------------------------------------------------

// 2 pi / 3, the spacing of the eigenvalues' angles in the trigonometric solution below.
// Writing an eigenvalue as mu + 2 sqrt(p) cos(phi) turns the cubic into cos(3 phi) = r.
// Since cos repeats every 2 pi, 3 phi = acos(r) + 2 pi k, so the three solutions are phi = acos(r) / 3 + 2 pi k / 3,
// k = 0, 1, 2: 2 pi / 3 apart.
// Rounded to float it is the paper's GPU literal 2.09439510239f.
constexpr double TWO_PI_OVER_3 = 2.0943951023931955;

// The three eigenvalues of a symmetric 4 x 4 matrix M on the complement of its unit null vector v (M v = 0), in
// decreasing order, by the trigonometric solution of the deflated cubic (as the paper's GPU implementation).
// Both callers satisfy M v = 0: K~ sqrt(F^b) = 0 and R sqrt(xi) = 0.
template <class T> P2V_HD inline void sym4x4_rank3_eigenvalues(const T M[16], const T v[4], T evals3[3]) {
    T mu = (M[0] + M[5] + M[10] + M[15]) * (T(1) / T(3));
    T B[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T proj = ((i == j) ? T(1) : T(0)) - v[i] * v[j];
            B[i * 4 + j] = M[i * 4 + j] - mu * proj;
        }
    }
    T s2 = T(0);
    for (int idx = 0; idx < 16; ++idx)
        s2 += B[idx] * B[idx];
    T p = s2 * (T(1) / T(6));
    if (p <= T(1e-14)) {
        evals3[0] = mu;
        evals3[1] = mu;
        evals3[2] = mu;
        return;
    }
    T s3 = T(0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T b2_ij = B[i * 4 + 0] * B[0 * 4 + j] + B[i * 4 + 1] * B[1 * 4 + j] + B[i * 4 + 2] * B[2 * 4 + j] +
                      B[i * 4 + 3] * B[3 * 4 + j];
            s3 += B[i * 4 + j] * b2_ij;
        }
    }
    T q = s3 * (T(1) / T(6));
    T sp = std::sqrt(p);
    T r = q / (p * sp);
    if (r < T(-1))
        r = T(-1);
    if (r > T(1))
        r = T(1);
    T theta = std::acos(r) * (T(1) / T(3));
    T two_sp = T(2) * sp;
    // phi = theta, theta - 2 pi / 3 and theta + 2 pi / 3 (k = 0, 2, 1): the three solutions of cos(3 phi) = r
    evals3[0] = mu + two_sp * std::cos(theta);
    evals3[1] = mu + two_sp * std::cos(theta - T(TWO_PI_OVER_3));
    evals3[2] = mu + two_sp * std::cos(theta + T(TWO_PI_OVER_3));
}

// ---- The eigenvalue GTR+GC distance ----------------------------------------------------------------------

// s_rates: exchangeabilities (AC, AG, AT, CG, CT, GT); alpha > 0 and inv_alpha = 1 / alpha (computed once by the
// caller, as the paper's GPU implementation does).
// Assumes a pair that shares a valid site and is not identical, as pair_distance (distances.hpp) ensures.
// pair_distance also caps the result.
template <class T>
P2V_HD inline T eigen_distance(const uint32_t c[N_BASES * N_BASES], const T s_rates[N_GTR_RATES], T alpha,
                               T inv_alpha) {
    uint32_t r0 = c[0] + c[1] + c[2] + c[3];
    uint32_t r1 = c[4] + c[5] + c[6] + c[7];
    uint32_t r2 = c[8] + c[9] + c[10] + c[11];
    uint32_t r3 = c[12] + c[13] + c[14] + c[15];
    uint32_t total_valid = r0 + r1 + r2 + r3;

    const T eps = T(EIGEN_PSEUDOCOUNT);
    T inv_tot = T(1) / ((T)total_valid + T(N_BASES * N_BASES) * eps);
    T K[N_BASES * N_BASES];
    T Fa[N_BASES] = {T(0), T(0), T(0), T(0)};
    T Fb[N_BASES] = {T(0), T(0), T(0), T(0)};
    for (int i = 0; i < N_BASES; ++i) {
        for (int j = 0; j < N_BASES; ++j) {
            T kij = ((T)c[i * N_BASES + j] + eps) * inv_tot;
            K[i * N_BASES + j] = kij;
            Fa[i] += kij;
            Fb[j] += kij;
        }
    }

    T sfa[N_BASES], inv_sfa[N_BASES], sfb[N_BASES], inv_sfb[N_BASES], xi[N_BASES];
    T xi_sum = T(0);
    for (int i = 0; i < N_BASES; ++i) {
        sfa[i] = std::sqrt(Fa[i]);
        inv_sfa[i] = T(1) / sfa[i];
        sfb[i] = std::sqrt(Fb[i]);
        inv_sfb[i] = T(1) / sfb[i];
        xi[i] = sfa[i] * sfb[i];
        xi_sum += xi[i];
    }
    T inv_xi_sum = T(1) / xi_sum;
    for (int i = 0; i < N_BASES; ++i)
        xi[i] *= inv_xi_sum;

    // The rate matrix for the pair frequencies xi
    T R[N_BASES * N_BASES];
    gtr_symmetric_rate_matrix<T>(s_rates, xi, R);

    T Ltilde[N_BASES * N_BASES];
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j)
            Ltilde[i * N_BASES + j] = inv_sfa[i] * K[i * N_BASES + j] * inv_sfb[j] - sfa[i] * sfb[j];

    T G[N_BASES * N_BASES];
    for (int i = 0; i < N_BASES; ++i) {
        for (int j = i; j < N_BASES; ++j) {
            T s = T(0);
            for (int k = 0; k < N_BASES; ++k)
                s += Ltilde[k * N_BASES + i] * Ltilde[k * N_BASES + j];
            G[i * N_BASES + j] = s;
            G[j * N_BASES + i] = s;
        }
    }
    T g_evals[3];
    sym4x4_rank3_eigenvalues<T>(G, sfb, g_evals);

    T sigma[3];
    for (int k = 0; k < 3; ++k) {
        T sv = (g_evals[k] > T(0)) ? std::sqrt(g_evals[k]) : T(EIGEN_SIGMA_MIN);
        if (sv < T(EIGEN_SIGMA_MIN))
            sv = T(EIGEN_SIGMA_MIN);
        if (sv > T(EIGEN_SIGMA_MAX))
            sv = T(EIGEN_SIGMA_MAX);
        sigma[k] = sv;
    }

    // SymWLS
    T sxi[N_BASES];
    for (int i = 0; i < N_BASES; ++i)
        sxi[i] = std::sqrt(xi[i]);
    T r_evals[3];
    sym4x4_rank3_eigenvalues<T>(R, sxi, r_evals);
    T y[3], neg_lam[3], lw[3];
    T max_lw = T(-1e30);
    for (int k = 0; k < 3; ++k) {
        T sig = sigma[k];
        T nl = -r_evals[k];
        if (nl < T(1e-6))
            nl = T(1e-6);
        neg_lam[k] = nl;
        T log_sig = std::log(sig);
        T u = -log_sig * inv_alpha;
        if (u > T(20))
            u = T(20);
        y[k] = alpha * std::expm1(u);
        lw[k] = (T(2) + T(2) * inv_alpha) * log_sig - std::log(T(1) - sig * sig + T(1e-4));
        if (lw[k] > max_lw)
            max_lw = lw[k];
    }
    T num = T(0), den = T(0);
    for (int k = 0; k < 3; ++k) {
        T w_k = std::exp(lw[k] - max_lw);
        num += w_k * neg_lam[k] * y[k];
        den += w_k * neg_lam[k] * neg_lam[k];
    }
    T d_ab = (den > T(1e-16)) ? (num / den) : T(0);
    if (d_ab < T(0))
        d_ab = T(0);

    const T log_sigma_min = sizeof(T) == 4 ? T(LOG_EIGEN_SIGMA_MIN_FLOAT) : T(LOG_EIGEN_SIGMA_MIN);
    for (int it = 0; it < EIGEN_REWEIGHT_PASSES; ++it) {
        T d_curr = (d_ab > T(MAX_DISTANCE)) ? T(MAX_DISTANCE) : d_ab;
        max_lw = T(-1e30);
        for (int k = 0; k < 3; ++k) {
            T base = T(1) + (d_curr * neg_lam[k]) * inv_alpha;
            if (base < T(1))
                base = T(1);
            T log_mu = -alpha * std::log(base);
            if (log_mu > T(-1e-6))
                log_mu = T(-1e-6);
            if (log_mu < log_sigma_min)
                log_mu = log_sigma_min;
            T mu_k = std::exp(log_mu);
            lw[k] = (T(2) + T(2) * inv_alpha) * log_mu - std::log(T(1) - mu_k * mu_k + T(1e-4));
            if (lw[k] > max_lw)
                max_lw = lw[k];
        }
        num = T(0);
        den = T(0);
        for (int k = 0; k < 3; ++k) {
            T w_k = std::exp(lw[k] - max_lw);
            num += w_k * neg_lam[k] * y[k];
            den += w_k * neg_lam[k] * neg_lam[k];
        }
        d_ab = (den > T(1e-16)) ? (num / den) : d_ab;
        if (d_ab < T(0))
            d_ab = T(0);
    }
    return d_ab;
}

} // namespace p2vbme

#endif // P2VBME_EIGEN_DIST_HPP
