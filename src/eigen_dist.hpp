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
 * g(sigma_k) / |lambda_k|, g(sigma) = alpha (sigma^(-1/alpha) - 1). Estimators:
 *   wls   SymWLS (the default; the distance of the paper's supplement): the weighted least-squares combination
 *         d = sum_k w_k |lambda_k| g(sigma_k) / sum_k w_k lambda_k^2, w_k = s_k^(2 + 2/alpha) / (1 - s_k^2 + 1e-4),
 *         with s_k = sigma_k, then `reweight` updates with s_k = (1 + |lambda_k| d / alpha)^(-alpha);
 *   svd   SymSVD: d = sum_k g(sigma_k) / (-trace Q);
 *   orig  the singular values of the (+1/2, -3/2) symmetrised matrix
 *         (diag(F^a)^(1/2) K diag(F^a)^(-3/2) + diag(F^b)^(-3/2) K diag(F^b)^(1/2)) / 2, all four in the sum.
 * Saturation: `direction` (default) clips each sigma to [1e-4, 1) (orig: [1e-4, 1]); a saturated mode gets almost no
 * weight in SymWLS's first fit, but the reweighting passes weight each mode by its model-implied s_k, so a fully
 * saturated direction can still pull the distance to the cap (the paper's GPU implementation behaves the same). `pair`
 * (the rule of the paper's C distance library) also sets all three sigma to 1e-4 when det K <= 0: the cap for alpha up
 * to about 14, 6.9 to 9.8 above. Distances lie in [0, 10]; no shared valid site gives 2.0, identical sequences 0.
 *
 * wls and svd with the default settings are the paper's GPU implementation, ported line by line (bitwise equal when
 * both are compiled without FMA contraction); orig and `pair` follow the paper's C distance library, with the GPU
 * implementation's pseudocount 1e-5 (library: 1e-8).
 * Not ported: the library's projection onto the bases present when a base is missing (its "reduced alphabet");
 * here a base absent from both sequences gives a pseudocount-only row of K~ (singular value 1/4), which reads as
 * a strongly diverged mode, as in the paper's GPU implementation.
 */
#ifndef P2VBME_EIGEN_DIST_HPP
#define P2VBME_EIGEN_DIST_HPP

#include <stdint.h>

#include "config.hpp"
#include "substitution_models.hpp"

namespace p2vbme {

enum class GtrEstimator : int32_t { Wls, Svd, Orig };
enum class Saturation : int32_t { Direction, Pair };

struct EigenSettings {
    GtrEstimator estimator = GtrEstimator::Wls;
    int32_t reweight = DEFAULT_GTR_REWEIGHT; /* SymWLS weight updates after the first fit (wls only) */
    Saturation saturation = Saturation::Direction;
};

const char *estimator_name(GtrEstimator e);            /* wls, svd, orig */
const char *saturation_name(Saturation s);             /* direction, pair */
GtrEstimator parse_estimator(const std::string &name); /* throws std::runtime_error */
Saturation parse_saturation(const std::string &name);

/* The pseudocount added to each of the 16 counts, and the range of the singular values. */
constexpr double EIGEN_PSEUDOCOUNT = 1e-5;
constexpr double EIGEN_SIGMA_MIN = 1e-4;
constexpr double EIGEN_SIGMA_MAX = 0.999999;

/* ---- 4 x 4 symmetric eigenvalues ------------------------------------------------------------------------ */

/* The three eigenvalues of a symmetric 4 x 4 matrix M on the complement of its unit null vector v (M v = 0), in
 * decreasing order, by the trigonometric solution of the deflated cubic (as the paper's GPU
 * implementation). Both callers satisfy M v = 0: K~ sqrt(F^b) = 0 and R sqrt(xi) = 0. */
/* Constants: the float path keeps the literals of the paper's GPU implementation; the double path the exact values. */
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
    T sp = fp::sqrt(p);
    T r = q / (p * sp);
    if (r < T(-1))
        r = T(-1);
    if (r > T(1))
        r = T(1);
    T theta = fp::acos(r) * (T(1) / T(3));
    T two_sp = T(2) * sp;
    evals3[0] = mu + two_sp * fp::cos(theta);
    evals3[1] = mu + two_sp * fp::cos(theta - two_pi_over_3<T>());
    evals3[2] = mu + two_sp * fp::cos(theta + two_pi_over_3<T>());
}

/* All four eigenvalues of a symmetric 4 x 4 matrix (overwritten), in decreasing order, by cyclic Jacobi rotations
 * (as the paper's C distance library, eigenvalues only). */
template <class T> P2V_HD inline void sym4x4_jacobi_eigenvalues(T A[16], T evals[4]) {
    const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int sweep = 0; sweep < 6; ++sweep) {
        T off = T(0);
        for (int k = 0; k < 6; ++k) {
            const T a = A[pairs[k][0] * 4 + pairs[k][1]];
            off += a * a;
        }
        if (off < T(1e-30))
            break;
        for (int k = 0; k < 6; ++k) {
            const int ip = pairs[k][0], iq = pairs[k][1];
            const T apq = A[ip * 4 + iq];
            if (apq < T(1e-18) && apq > T(-1e-18))
                continue;
            const T app = A[ip * 4 + ip], aqq = A[iq * 4 + iq];
            const T tau = (aqq - app) / (T(2) * apq);
            const T t = (tau >= T(0)) ? (T(1) / (tau + fp::sqrt(T(1) + tau * tau)))
                                      : (T(-1) / (-tau + fp::sqrt(T(1) + tau * tau)));
            const T c = T(1) / fp::sqrt(T(1) + t * t), s = t * c;
            A[ip * 4 + ip] = app - t * apq;
            A[iq * 4 + iq] = aqq + t * apq;
            A[ip * 4 + iq] = T(0);
            A[iq * 4 + ip] = T(0);
            for (int r = 0; r < 4; ++r) {
                if (r != ip && r != iq) {
                    const T arp = A[r * 4 + ip], arq = A[r * 4 + iq];
                    A[r * 4 + ip] = c * arp - s * arq;
                    A[ip * 4 + r] = A[r * 4 + ip];
                    A[r * 4 + iq] = s * arp + c * arq;
                    A[iq * 4 + r] = A[r * 4 + iq];
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i)
        evals[i] = A[i * 4 + i];
    for (int i = 0; i < 3; ++i) {
        int best = i;
        for (int j = i + 1; j < 4; ++j)
            if (evals[j] > evals[best])
                best = j;
        const T tmp = evals[i];
        evals[i] = evals[best];
        evals[best] = tmp;
    }
}

/* ---- The eigenvalue GTR+GC distance ---------------------------------------------------------------------- */

/* s_rates: exchangeabilities (AC, AG, AT, CG, CT, GT); alpha > 0 and inv_alpha = 1 / alpha (computed once by the
 * caller, as the paper's GPU implementation does). */
template <class T>
P2V_HD inline T eigen_distance(const uint32_t c[16], const T s_rates[6], T alpha, T inv_alpha,
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
    T inv_tot = T(1) / ((T)total_valid + T(16) * eps);
    T K[16];
    T Fa[4] = {T(0), T(0), T(0), T(0)};
    T Fb[4] = {T(0), T(0), T(0), T(0)};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T kij = ((T)c[i * 4 + j] + eps) * inv_tot;
            K[i * 4 + j] = kij;
            Fa[i] += kij;
            Fb[j] += kij;
        }
    }

    T sfa[4], inv_sfa[4], sfb[4], inv_sfb[4], xi[4];
    T xi_sum = T(0);
    for (int i = 0; i < 4; ++i) {
        sfa[i] = fp::sqrt(Fa[i]);
        inv_sfa[i] = T(1) / sfa[i];
        sfb[i] = fp::sqrt(Fb[i]);
        inv_sfb[i] = T(1) / sfb[i];
        xi[i] = sfa[i] * sfb[i];
        xi_sum += xi[i];
    }
    T inv_xi_sum = T(1) / xi_sum;
    for (int i = 0; i < 4; ++i)
        xi[i] *= inv_xi_sum;

    /* The rate matrix for the pair frequencies xi */
    T R[16];
    const T neg_tr_Q = gtr_symmetric_rate_matrix<T>(s_rates, xi, R);

    if (opt.estimator == GtrEstimator::Orig) {
        /* orig: all four singular values of the (+1/2, -3/2) symmetrised matrix */
        T G[16];
        {
            T Lbar[16];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const T t1 = sfa[i] * K[i * 4 + j] / (Fa[j] * sfa[j]);
                    const T t2 = K[i * 4 + j] * sfb[j] / (Fb[i] * sfb[i]);
                    Lbar[i * 4 + j] = T(0.5) * (t1 + t2);
                }
            for (int i = 0; i < 4; ++i)
                for (int j = i; j < 4; ++j) {
                    T s = T(0);
                    for (int k = 0; k < 4; ++k)
                        s += Lbar[k * 4 + i] * Lbar[k * 4 + j];
                    G[i * 4 + j] = s;
                    G[j * 4 + i] = s;
                }
        }
        T ev[4];
        sym4x4_jacobi_eigenvalues<T>(G, ev);
        T sig4[4];
        for (int k = 0; k < 4; ++k)
            sig4[k] = (ev[k] > T(0)) ? fp::sqrt(ev[k]) : T(0);
        if (opt.saturation == Saturation::Pair) {
            const T s0 = K[0] * K[5] - K[1] * K[4], s1 = K[0] * K[6] - K[2] * K[4], s2 = K[0] * K[7] - K[3] * K[4];
            const T s3 = K[1] * K[6] - K[2] * K[5], s4 = K[1] * K[7] - K[3] * K[5], s5 = K[2] * K[7] - K[3] * K[6];
            const T c5 = K[10] * K[15] - K[11] * K[14], c4 = K[9] * K[15] - K[11] * K[13];
            const T c3 = K[9] * K[14] - K[10] * K[13], c2 = K[8] * K[15] - K[11] * K[12];
            const T c1 = K[8] * K[14] - K[10] * K[12], c0 = K[8] * K[13] - K[9] * K[12];
            if (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0 <= T(0))
                sig4[1] = sig4[2] = sig4[3] = T(EIGEN_SIGMA_MIN);
        }
        T sum_gamma = T(0);
        for (int k = 0; k < 4; ++k) {
            T sig = sig4[k];
            if (sig > T(1))
                sig = T(1);
            if (sig < T(EIGEN_SIGMA_MIN))
                sig = T(EIGEN_SIGMA_MIN);
            T u = -fp::log(sig) * inv_alpha;
            if (u > T(30))
                u = T(30);
            sum_gamma += alpha * fp::expm1(u);
        }
        T d_ab = (neg_tr_Q > T(1e-12)) ? (sum_gamma / neg_tr_Q) : T(0);
        if (d_ab < T(0))
            d_ab = T(0);
        if (d_ab > T(MAX_DISTANCE))
            d_ab = T(MAX_DISTANCE);
        return d_ab;
    }

    T Ltilde[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Ltilde[i * 4 + j] = inv_sfa[i] * K[i * 4 + j] * inv_sfb[j] - sfa[i] * sfb[j];

    T G[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = i; j < 4; ++j) {
            T s = T(0);
            for (int k = 0; k < 4; ++k)
                s += Ltilde[k * 4 + i] * Ltilde[k * 4 + j];
            G[i * 4 + j] = s;
            G[j * 4 + i] = s;
        }
    }
    T g_evals[3];
    sym4x4_rank3_eigenvalues<T>(G, sfb, g_evals);

    T sigma[3];
    for (int k = 0; k < 3; ++k) {
        T sv = (g_evals[k] > T(0)) ? fp::sqrt(g_evals[k]) : T(EIGEN_SIGMA_MIN);
        if (sv < T(EIGEN_SIGMA_MIN))
            sv = T(EIGEN_SIGMA_MIN);
        if (sv > T(EIGEN_SIGMA_MAX))
            sv = T(EIGEN_SIGMA_MAX);
        sigma[k] = sv;
    }
    if (opt.saturation == Saturation::Pair) {
        /* the rule of the paper's C distance library: det K <= 0 (a mode has crossed zero) saturates the whole pair */
        const T s0 = K[0] * K[5] - K[1] * K[4], s1 = K[0] * K[6] - K[2] * K[4], s2 = K[0] * K[7] - K[3] * K[4];
        const T s3 = K[1] * K[6] - K[2] * K[5], s4 = K[1] * K[7] - K[3] * K[5], s5 = K[2] * K[7] - K[3] * K[6];
        const T c5 = K[10] * K[15] - K[11] * K[14], c4 = K[9] * K[15] - K[11] * K[13];
        const T c3 = K[9] * K[14] - K[10] * K[13], c2 = K[8] * K[15] - K[11] * K[12];
        const T c1 = K[8] * K[14] - K[10] * K[12], c0 = K[8] * K[13] - K[9] * K[12];
        if (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0 <= T(0))
            sigma[0] = sigma[1] = sigma[2] = T(EIGEN_SIGMA_MIN);
    }

    if (opt.estimator == GtrEstimator::Svd) {
        /* SymSVD: the sum of the three modes over -trace Q */
        T sum_gamma = T(0);
        for (int k = 0; k < 3; ++k) {
            T u = -fp::log(sigma[k]) * inv_alpha;
            if (u > T(20))
                u = T(20);
            sum_gamma += alpha * fp::expm1(u);
        }
        T d_ab = (neg_tr_Q > T(1e-8)) ? (sum_gamma / neg_tr_Q) : T(0);
        if (d_ab < T(0))
            d_ab = T(0);
        if (d_ab > T(MAX_DISTANCE))
            d_ab = T(MAX_DISTANCE);
        return d_ab;
    }

    /* SymWLS */
    T sxi[4];
    for (int i = 0; i < 4; ++i)
        sxi[i] = fp::sqrt(xi[i]);
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
        T log_sig = fp::log(sig);
        T u = -log_sig * inv_alpha;
        if (u > T(20))
            u = T(20);
        y[k] = alpha * fp::expm1(u);
        lw[k] = (T(2) + T(2) * inv_alpha) * log_sig - fp::log(T(1) - sig * sig + T(1e-4));
        if (lw[k] > max_lw)
            max_lw = lw[k];
    }
    T num = T(0), den = T(0);
    for (int k = 0; k < 3; ++k) {
        T w_k = fp::exp(lw[k] - max_lw);
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
            T log_mu = -alpha * fp::log(base);
            if (log_mu > T(-1e-6))
                log_mu = T(-1e-6);
            if (log_mu < log_sigma_min<T>())
                log_mu = log_sigma_min<T>();
            T mu_k = fp::exp(log_mu);
            lw[k] = (T(2) + T(2) * inv_alpha) * log_mu - fp::log(T(1) - mu_k * mu_k + T(1e-4));
            if (lw[k] > max_lw)
                max_lw = lw[k];
        }
        num = T(0);
        den = T(0);
        for (int k = 0; k < 3; ++k) {
            T w_k = fp::exp(lw[k] - max_lw);
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

/* ---- One pair's distance under the run's model ----------------------------------------------------------- */

/* The model in the precision of a device (the GPU: float). */
template <class T> struct DistanceParams {
    Model model;
    T rates[6];
    T alpha, inv_alpha;
    EigenSettings eigen;
    T pi[4]; /* base frequencies, normalised (uniform if not given) */
};

template <class T> inline DistanceParams<T> distance_params(const SubstitutionModel &m, const EigenSettings &eigen) {
    DistanceParams<T> p;
    p.model = m.model;
    for (int k = 0; k < 6; ++k)
        p.rates[k] = (T)m.gtr_rates[k];
    p.alpha = (T)m.gtr_alpha;
    p.inv_alpha = T(1) / p.alpha; /* as the paper's GPU implementation: 1 / alpha in the device's precision */
    p.eigen = eigen;
    const double sum = m.base_freqs[0] + m.base_freqs[1] + m.base_freqs[2] + m.base_freqs[3];
    for (int i = 0; i < 4; ++i)
        p.pi[i] = sum > 0.0 ? (T)(m.base_freqs[i] / sum) : T(0.25);
    return p;
}

/* The distance of a pair with counts n under the model. */
template <class T> P2V_HD inline T pair_distance(const DistanceParams<T> &p, const uint32_t n[16]) {
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
