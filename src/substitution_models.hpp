/*
 * substitution_models.hpp
 *
 * The substitution models of the distances: their names, the pair counts they start from, and
 * the closed-form distances of JC69, K80, F81, F84, TN93 and LogDet, plus the symmetric form of the GTR rate
 * matrix that the eigenvalue GTR+GC distance (eigen_dist.hpp) uses. The per-pair formulas are inline and
 * __host__ __device__, so the GPU kernels and the CPU device evaluate the same code; they are templates on the
 * floating-point type (the GPU uses float).
 *
 * Pair counts: n[4 * i + j] is the number of sites where sequence a has base i and sequence b has base j (order A,
 * C, G, T), over the sites valid in both. Distances are in expected substitutions per site.
 *
 * The formulas: JC69 is the search's (float); K80, F81, F84 and TN93 are those of ape's dist.dna, with the base
 * frequencies of the whole alignment (base_frequencies), as ape and FastME estimate them (comparable with other
 * programs); LogDet is the paper's C distance library's, which uses each sequence's own frequencies by definition.
 * Values other programs leave undefined (NaN or infinite) are made finite here, since the search needs a number for
 * every pair: a logarithm's argument is floored at 1e-4 and a distance capped at MAX_DISTANCE (the paper's C distance
 * library's rules); no shared valid site gives NO_OVERLAP_DISTANCE; an undefined LogDet gives MAX_DISTANCE.
 */
#pragma once

#include <math.h>
#include <stdint.h>

#include <string>

/* Marks the per-pair formulas host and device functions under nvcc (plain functions elsewhere). */
#ifndef P2V_HD
#ifdef __CUDACC__
#define P2V_HD __host__ __device__
#else
#define P2V_HD
#endif
#endif

namespace p2vbme {

/* ---- Models ---------------------------------------------------------------------------------------------- */

enum class Model : int32_t { JC69, K80, F81, F84, TN93, LogDet, GTR_GC };

/* Canonical name: JC69, K80, F81, F84, TN93, LogDet, GTR+GC. */
const char *model_name(Model model);

/* IQ-TREE-style names, case-insensitive: JC69 (JC), K80 (K2P), F81, F84, TN93 (TN), LogDet,
 * GTR+GC. Throws std::runtime_error for an unknown name, for +G (continuous gamma is +GC) and for GTR without
 * +GC. */
Model parse_model(const std::string &name);

/* ---- Constants ------------------------------------------------------------------------------------------- */

/* JC69 (the search's values): p is capped at JC69_P_MAX, so the largest distance is about 3.24; a pair with no
 * shared valid site gets JC69_NO_OVERLAP_DISTANCE. */
constexpr double JC69_P_MAX = 0.74;
constexpr double JC69_NO_OVERLAP_DISTANCE = 1.5;
/* Every other model: a pair with no shared valid site (the value of the paper's GPU implementation) */
constexpr double NO_OVERLAP_DISTANCE = 2.0;
/* and the largest distance (the cap of the paper's C distance library) */
constexpr double MAX_DISTANCE = 10.0;

/* ---- Pair counts ----------------------------------------------------------------------------------------- */

struct PairCounts {
    uint32_t n[16];
};

/* Counts of two rows of the 5-bitplane layout (Alignment::packed: planes A, C, G, T, valid; n_words each). */
PairCounts count_pair(const uint64_t *row_a, const uint64_t *row_b, int32_t n_words);

/* The base frequencies (A, C, G, T) of an alignment in the 5-bitplane layout: each base's share of all valid sites of
 * all sequences (as ape's base.freq and FastME). Uniform if the alignment has no valid site. */
void base_frequencies(const uint64_t *packed, int32_t n_taxa, int32_t n_words, double freqs[4]);

/* ---- Floating-point helpers (float and double overloads, host and device) --------------------------------- */

namespace fp {
P2V_HD inline float log(float x) { return ::logf(x); }
P2V_HD inline double log(double x) { return ::log(x); }
P2V_HD inline float exp(float x) { return ::expf(x); }
P2V_HD inline double exp(double x) { return ::exp(x); }
P2V_HD inline float expm1(float x) { return ::expm1f(x); }
P2V_HD inline double expm1(double x) { return ::expm1(x); }
P2V_HD inline float sqrt(float x) { return ::sqrtf(x); }
P2V_HD inline double sqrt(double x) { return ::sqrt(x); }
P2V_HD inline float acos(float x) { return ::acosf(x); }
P2V_HD inline double acos(double x) { return ::acos(x); }
P2V_HD inline float cos(float x) { return ::cosf(x); }
P2V_HD inline double cos(double x) { return ::cos(x); }
} // namespace fp

/* ---- Closed-form distances ------------------------------------------------------------------------------- */

/* JC69 from the number of shared valid sites and of matching sites (the search core's formula):
 * d = -3/4 ln(1 - 4/3 p), p = mismatches / valid, capped at JC69_P_MAX. */
template <class T> P2V_HD inline T jc69_distance(int32_t n_valid, int32_t n_match) {
    if (n_valid <= 0)
        return T(JC69_NO_OVERLAP_DISTANCE);
    T p = (T)(n_valid - n_match) / (T)n_valid;
    if (p >= T(JC69_P_MAX))
        p = T(JC69_P_MAX);
    T d = T(-0.75) * fp::log(T(1) - (T(4) / T(3)) * p);
    return (d > T(0)) ? d : T(0);
}

namespace detail {

/* -ln(w) with w clamped to [1e-4, 1], and 0 when w >= 1 - 1e-12 (the homogeneous link of the paper's C
 * distance library, its alpha = 0 case). The floor 1e-4 bounds each term at ln(1e4) = 9.21. */
template <class T> P2V_HD inline T neg_log_link(T w) {
    if (w > T(1))
        w = T(1);
    if (w < T(1e-4))
        w = T(1e-4);
    if (w >= T(1) - T(1e-12))
        return T(0);
    return -fp::log(w);
}

template <class T> P2V_HD inline T clamp_distance(T d) {
    if (d < T(0))
        d = T(0);
    if (d > T(MAX_DISTANCE))
        d = T(MAX_DISTANCE);
    return d;
}

/* What the closed forms share: proportions of a pair's sites. */
template <class T> struct Proportions {
    T p;      /* mismatches */
    T P1, P2; /* A <-> G and C <-> T differences (transitions) */
    T Q;      /* transversions, p - P1 - P2 */
};

template <class T> P2V_HD inline Proportions<T> proportions(const uint32_t n[16], uint32_t total) {
    Proportions<T> s;
    const T inv = T(1) / (T)total;
    const uint32_t same = n[0] + n[5] + n[10] + n[15];
    s.p = (T)(total - same) * inv;
    s.P1 = (T)(n[2] + n[8]) * inv;
    s.P2 = (T)(n[7] + n[13]) * inv;
    s.Q = s.p - s.P1 - s.P2;
    if (s.Q < T(0))
        s.Q = T(0);
    return s;
}

P2V_HD inline uint32_t total_count(const uint32_t n[16]) {
    uint32_t t = 0;
    for (int k = 0; k < 16; ++k)
        t += n[k];
    return t;
}

P2V_HD inline uint32_t diagonal_count(const uint32_t n[16]) { return n[0] + n[5] + n[10] + n[15]; }

/* F81's d = -E ln(1 - p / E), E = 1 - sum pi_i^2 (floored at 1e-4): also F84's and TN93's when the frequencies make
 * their own form undefined (a base, or both purines or both pyrimidines, absent from the alignment). */
template <class T> P2V_HD inline T f81_form(T p, const T pi[4]) {
    T E = T(1) - (pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2] + pi[3] * pi[3]);
    if (E < T(1e-4))
        E = T(1e-4);
    return E * neg_log_link(T(1) - p / E);
}

} // namespace detail

/* K80 (Kimura 1980): d = -1/2 ln(1 - 2P - Q) - 1/4 ln(1 - 2Q), P = transitions, Q = transversions. */
template <class T> P2V_HD inline T k80_distance(const uint32_t n[16]) {
    const uint32_t total = detail::total_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (detail::diagonal_count(n) == total)
        return T(0);
    const detail::Proportions<T> s = detail::proportions<T>(n, total);
    const T P = s.P1 + s.P2;
    const T d =
        T(0.5) * detail::neg_log_link(T(1) - T(2) * P - s.Q) + T(0.25) * detail::neg_log_link(T(1) - T(2) * s.Q);
    return detail::clamp_distance(d);
}

/* F81 (Felsenstein 1981; as ape's dist.dna): d = -E ln(1 - p / E), E = 1 - sum pi_i^2, pi the alignment's base
 * frequencies. */
template <class T> P2V_HD inline T f81_distance(const uint32_t n[16], const T pi[4]) {
    const uint32_t total = detail::total_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (detail::diagonal_count(n) == total)
        return T(0);
    const detail::Proportions<T> s = detail::proportions<T>(n, total);
    return detail::clamp_distance(detail::f81_form(s.p, pi));
}

/* F84 (Felsenstein & Churchill 1996; as ape's dist.dna): with the alignment's base frequencies pi,
 *   A = pi_A pi_G / pi_R + pi_C pi_T / pi_Y,  B = pi_A pi_G + pi_C pi_T,  C = pi_R pi_Y,
 *   d = -2A ln(1 - P / (2A) - (A - B) Q / (2AC)) - 2(C + B - A) ln(1 - Q / (2C)).
 * Falls back to F81's form when A or C is below 1e-6. */
template <class T> P2V_HD inline T f84_distance(const uint32_t n[16], const T pi[4]) {
    const uint32_t total = detail::total_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (detail::diagonal_count(n) == total)
        return T(0);
    const detail::Proportions<T> s = detail::proportions<T>(n, total);
    const T piA = pi[0], piC = pi[1], piG = pi[2], piT = pi[3];
    const T piR = piA + piG, piY = piC + piT;
    const T A = (piR > T(1e-6) ? piA * piG / piR : T(0)) + (piY > T(1e-6) ? piC * piT / piY : T(0));
    const T B = piA * piG + piC * piT;
    const T C = piR * piY;
    if (A < T(1e-6) || C < T(1e-6))
        return detail::clamp_distance(detail::f81_form(s.p, pi));
    const T P = s.P1 + s.P2;
    const T g1 = detail::neg_log_link(T(1) - P / (T(2) * A) - ((A - B) * s.Q) / (T(2) * A * C));
    const T g2 = detail::neg_log_link(T(1) - s.Q / (T(2) * C));
    return detail::clamp_distance(T(2) * A * g1 + T(2) * (C + B - A) * g2);
}

/* TN93 (Tamura & Nei 1993; as ape's dist.dna): with the alignment's base frequencies pi, P1 = A <-> G, P2 = C <-> T,
 *   k1 = 2 pi_A pi_G / pi_R,  k2 = 2 pi_C pi_T / pi_Y,  k3 = 2 pi_R pi_Y,
 *   d = -k1 ln(w1) - k2 ln(w2) - (k3 - k1 pi_Y - k2 pi_R) ln(w3),
 *   w1 = 1 - P1 / k1 - Q / (2 pi_R),  w2 = 1 - P2 / k2 - Q / (2 pi_Y),  w3 = 1 - Q / k3.
 * Falls back to F81's form when k1, k2 or k3 is below 1e-6. */
template <class T> P2V_HD inline T tn93_distance(const uint32_t n[16], const T pi[4]) {
    const uint32_t total = detail::total_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (detail::diagonal_count(n) == total)
        return T(0);
    const detail::Proportions<T> s = detail::proportions<T>(n, total);
    const T piA = pi[0], piC = pi[1], piG = pi[2], piT = pi[3];
    const T piR = piA + piG, piY = piC + piT;
    const T k1 = (piR > T(1e-6)) ? T(2) * piA * piG / piR : T(0);
    const T k2 = (piY > T(1e-6)) ? T(2) * piC * piT / piY : T(0);
    const T k3 = T(2) * piR * piY;
    if (k1 < T(1e-6) || k2 < T(1e-6) || k3 < T(1e-6))
        return detail::clamp_distance(detail::f81_form(s.p, pi));
    const T g1 = detail::neg_log_link(T(1) - s.P1 / k1 - s.Q / (T(2) * piR));
    const T g2 = detail::neg_log_link(T(1) - s.P2 / k2 - s.Q / (T(2) * piY));
    const T g3 = detail::neg_log_link(T(1) - s.Q / k3);
    return detail::clamp_distance(k1 * g1 + k2 * g2 + (k3 - k1 * piY - k2 * piR) * g3);
}

/* LogDet / paralinear (Lockhart et al. 1994): d = -1/4 (ln det J - 1/2 ln prod_i F^a_i F^b_i), J the 4 x 4 matrix
 * of pair proportions. Undefined (MAX_DISTANCE here; NaN in the paper's C distance library) when a base has frequency
 * below 1e-3 in either sequence or det J <= 1e-30. */
template <class T> P2V_HD inline T logdet_distance(const uint32_t n[16]) {
    const uint32_t total = detail::total_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (detail::diagonal_count(n) == total)
        return T(0);
    const T inv = T(1) / (T)total;
    T J[16];
    for (int k = 0; k < 16; ++k)
        J[k] = (T)n[k] * inv;
    T prod_f = T(1);
    for (int i = 0; i < 4; ++i) {
        const T fa = J[4 * i] + J[4 * i + 1] + J[4 * i + 2] + J[4 * i + 3];
        const T fb = J[i] + J[4 + i] + J[8 + i] + J[12 + i];
        if (fa < T(1e-3) || fb < T(1e-3))
            return T(MAX_DISTANCE);
        prod_f *= fa * fb;
    }
    /* determinant by 2 x 2 minors of the top and bottom row pairs (Laplace expansion) */
    const T s0 = J[0] * J[5] - J[1] * J[4], s1 = J[0] * J[6] - J[2] * J[4], s2 = J[0] * J[7] - J[3] * J[4];
    const T s3 = J[1] * J[6] - J[2] * J[5], s4 = J[1] * J[7] - J[3] * J[5], s5 = J[2] * J[7] - J[3] * J[6];
    const T c5 = J[10] * J[15] - J[11] * J[14], c4 = J[9] * J[15] - J[11] * J[13], c3 = J[9] * J[14] - J[10] * J[13];
    const T c2 = J[8] * J[15] - J[11] * J[12], c1 = J[8] * J[14] - J[10] * J[12], c0 = J[8] * J[13] - J[9] * J[12];
    const T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (!(det > T(1e-30)) || !(prod_f > T(1e-30)))
        return T(MAX_DISTANCE);
    return detail::clamp_distance(T(-0.25) * (fp::log(det) - T(0.5) * fp::log(prod_f)));
}

/* ---- GTR rate matrix ------------------------------------------------------------------------------------- */

/* Symmetric form R = Pi^(1/2) Q Pi^(-1/2) of the GTR rate matrix Q with exchangeabilities s (AC, AG, AT, CG, CT,
 * GT) and stationary frequencies xi, normalised to one expected substitution per unit time (mean rate
 * sum_i xi_i sum_j s_ij xi_j = 1). R has the eigenvalues of Q. Returns -trace(Q). */
template <class T> P2V_HD inline T gtr_symmetric_rate_matrix(const T s[6], const T xi[4], T R[16]) {
    const T S[16] = {T(0), s[0], s[1], s[2], s[0], T(0), s[3], s[4], s[1], s[3], T(0), s[5], s[2], s[4], s[5], T(0)};
    T row_sum[4];
    T mu = T(0);
    for (int i = 0; i < 4; ++i) {
        T rs = T(0);
        for (int j = 0; j < 4; ++j)
            rs += S[i * 4 + j] * xi[j];
        row_sum[i] = rs;
        mu += xi[i] * rs;
    }
    const T inv_mu = (mu > T(1e-20)) ? (T(1) / mu) : T(1);
    T sxi[4];
    for (int i = 0; i < 4; ++i)
        sxi[i] = fp::sqrt(xi[i]);
    T neg_tr_Q = T(0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                const T q_ii = -row_sum[i] * inv_mu;
                R[i * 4 + i] = q_ii;
                neg_tr_Q -= q_ii;
            } else {
                R[i * 4 + j] = S[i * 4 + j] * sxi[i] * sxi[j] * inv_mu;
            }
        }
    }
    return neg_tr_Q;
}

} // namespace p2vbme
