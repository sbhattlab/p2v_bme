/*
 * eigen_dist.hpp
 *
 * The eigenvalue GTR+GC distance (Penn et al. 2025, Bull. Math. Biol. 87, 32, doi:10.1007/s11538-024-01403-z) with
 * given GTR exchangeabilities and gamma shape alpha, and the dispatch of a pair's distance over all models
 * (pair_distance). Per-pair code, inline and __host__ __device__, templated on the floating-point type.
 *
 * For a pair (a, b) with pair counts n: K = (n + 1e-5) / (L + 16e-5), F^a and F^b its row and column sums, and
 * sigma_1 >= sigma_2 >= sigma_3 the non-trivial singular values of
 *   K~ = diag(F^a)^(-1/2) K diag(F^b)^(-1/2) - sqrt(F^a) sqrt(F^b)^T,
 * clipped to [1e-4, 1). The rate matrix uses pair frequencies xi_i = sqrt(F^a_i F^b_i) / sum_j sqrt(F^a_j F^b_j);
 * lambda_k < 0 are its non-zero eigenvalues, by increasing magnitude. Each mode estimates the distance by
 * g(sigma_k) / |lambda_k|, g(sigma) = alpha (sigma^(-1/alpha) - 1). SymWLS (the distance of the paper's supplement)
 * combines the three modes by weighted least squares,
 *   d = sum_k w_k |lambda_k| g(sigma_k) / sum_k w_k lambda_k^2,  w_k = s_k^(2 + 2/alpha) / (1 - s_k^2 + 1e-4),
 * with s_k = sigma_k, then `reweight` updates with s_k = (1 + |lambda_k| d / alpha)^(-alpha).
 * Saturation: a saturated mode (sigma at 1e-4) gets almost no weight in the first fit, but the reweighting passes
 * weight each mode by its model-implied s_k, so a fully saturated direction can still pull the distance to the cap (the
 * paper's GPU implementation behaves the same). Distances lie in [0, 10]; no shared valid site gives 2.0, identical
 * sequences 0.
 *
 * The paper's GPU implementation, ported line by line (bitwise equal when both are compiled without FMA contraction,
 * for exchangeabilities with mean 1 as the paper's fit returns them; others are scaled to mean 1 first).
 * Not ported: the paper's C distance library's projection onto the bases present when a base is missing (its "reduced
 * alphabet");
 * here a base absent from both sequences gives a pseudocount-only row of K~ (singular value 1/4), which reads as
 * a strongly diverged mode, as in the paper's GPU implementation.
 */
#ifndef P2VBME_EIGEN_DIST_HPP
#define P2VBME_EIGEN_DIST_HPP

#include <stdint.h>

#include <cmath>

#include "config.hpp"
#include "distances.hpp"

namespace p2vbme {

struct EigenSettings {
    int32_t reweight = DEFAULT_GTR_REWEIGHT; // SymWLS weight updates after the first fit
};

// The pseudocount added to each of the 16 counts, and the range of the singular values.
constexpr double EIGEN_PSEUDOCOUNT = 1e-5;
constexpr double EIGEN_SIGMA_MIN = 1e-4;
constexpr double EIGEN_SIGMA_MAX = 0.999999;

// ---- 4 x 4 symmetric eigenvalues ------------------------------------------------------------------------

// The three eigenvalues of a symmetric 4 x 4 matrix M on the complement of its unit null vector v (M v = 0), in
// decreasing order, by the trigonometric solution of the deflated cubic (as the paper's GPU implementation).
// Both callers satisfy M v = 0: K~ sqrt(F^b) = 0 and R sqrt(xi) = 0.
// Constants: the float path keeps the literals of the paper's GPU implementation; the double path the exact values.
template <class T> P2V_HD inline T two_pi_over_3() {
    return sizeof(T) == 4 ? T(2.09439510239f) : T(2.0943951023931955);
}
template <class T> P2V_HD inline T log_sigma_min() { return sizeof(T) == 4 ? T(-9.21034f) : T(-9.2103403719761836); }
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
    evals3[0] = mu + two_sp * std::cos(theta);
    evals3[1] = mu + two_sp * std::cos(theta - two_pi_over_3<T>());
    evals3[2] = mu + two_sp * std::cos(theta + two_pi_over_3<T>());
}

// ---- The eigenvalue GTR+GC distance ----------------------------------------------------------------------

// s_rates: exchangeabilities (AC, AG, AT, CG, CT, GT); alpha > 0 and inv_alpha = 1 / alpha (computed once by the
// caller, as the paper's GPU implementation does).
template <class T>
P2V_HD inline T eigen_distance(const uint32_t c[N_BASES * N_BASES], const T s_rates[N_GTR_RATES], T alpha, T inv_alpha,
                               const EigenSettings &opt) {
    uint32_t r0 = c[0] + c[1] + c[2] + c[3];
    uint32_t r1 = c[4] + c[5] + c[6] + c[7];
    uint32_t r2 = c[8] + c[9] + c[10] + c[11];
    uint32_t r3 = c[12] + c[13] + c[14] + c[15];
    uint32_t total_valid = r0 + r1 + r2 + r3;
    uint32_t diag_match = c[0] + c[5] + c[10] + c[15];
    if (total_valid == 0u)
        return T(NO_OVERLAP_DISTANCE);
    if (diag_match == total_valid)
        return T(0);

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

    for (int it = 0; it < opt.reweight; ++it) {
        T d_curr = (d_ab > T(MAX_DISTANCE)) ? T(MAX_DISTANCE) : d_ab;
        max_lw = T(-1e30);
        for (int k = 0; k < 3; ++k) {
            T base = T(1) + (d_curr * neg_lam[k]) * inv_alpha;
            if (base < T(1))
                base = T(1);
            T log_mu = -alpha * std::log(base);
            if (log_mu > T(-1e-6))
                log_mu = T(-1e-6);
            if (log_mu < log_sigma_min<T>())
                log_mu = log_sigma_min<T>();
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
    if (d_ab > T(MAX_DISTANCE))
        d_ab = T(MAX_DISTANCE);
    return d_ab;
}

// ---- One pair's distance under the run's model -----------------------------------------------------------

// The model in the precision of a device (the GPU: float).
template <class T> struct DistanceParams {
    Model model;
    T rates[N_GTR_RATES];
    T alpha, inv_alpha;
    EigenSettings eigen;
    T pi[N_BASES]; // base frequencies, normalised (uniform if not given)
};

template <class T> inline DistanceParams<T> distance_params(const SubstitutionModel &m, const EigenSettings &eigen) {
    DistanceParams<T> p;
    p.model = m.model;
    // The exchangeabilities scaled to mean 1, as the paper's fit returns them.
    // The distance depends only on their ratios; the scaling keeps the rate matrix's own normalisation in range.
    double rate_sum = 0.0;
    for (int k = 0; k < N_GTR_RATES; ++k)
        rate_sum += m.rates[k];
    const double rate_scale = 6.0 / rate_sum;
    for (int k = 0; k < N_GTR_RATES; ++k)
        p.rates[k] = (T)(m.rates[k] * rate_scale);
    p.alpha = (T)m.alpha;
    p.inv_alpha = T(1) / p.alpha; // as the paper's GPU implementation: 1 / alpha in the device's precision
    p.eigen = eigen;
    const double sum = m.base_freqs[0] + m.base_freqs[1] + m.base_freqs[2] + m.base_freqs[3];
    for (int i = 0; i < N_BASES; ++i)
        p.pi[i] = sum > 0.0 ? (T)(m.base_freqs[i] / sum) : T(0.25);
    return p;
}

// The distance of a pair with counts n under the model.
template <class T> P2V_HD inline T pair_distance(const DistanceParams<T> &p, const uint32_t n[N_BASES * N_BASES]) {
    switch (p.model) {
    case Model::JC69:
        return jc69_distance<T>((int32_t)detail::total_count(n), (int32_t)detail::diagonal_count(n));
    case Model::K80:
        return k80_distance<T>(n);
    case Model::F81:
        return f81_distance<T>(n, p.pi);
    case Model::F84:
        return f84_distance<T>(n, p.pi);
    case Model::TN93:
        return tn93_distance<T>(n, p.pi);
    case Model::LogDet:
        return logdet_distance<T>(n);
    case Model::GTR_GC:
        return eigen_distance<T>(n, p.rates, p.alpha, p.inv_alpha, p.eigen);
    }
    return T(MAX_DISTANCE);
}

} // namespace p2vbme

#endif // P2VBME_EIGEN_DIST_HPP
