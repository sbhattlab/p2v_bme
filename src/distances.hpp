/*
 * distances.hpp
 *
 * The distance of a pair of sequences under each model, from the pair's counts: n[4 * i + j] is the number of sites
 * where sequence a has base i and sequence b has base j (A, C, G, T), over the sites valid in both.
 * In order: the models and their parameters, the pair counts, one formula per model (GTR's and GTR+GC's in
 * eigen_dist.hpp), and pair_distance, which applies the run's model.
 * Distances are in expected substitutions per site, and always finite (config.hpp: NO_OVERLAP_DISTANCE, MAX_DISTANCE).
 * The formulas are inline host/device templates on the floating-point type, so the GPU kernels (float) and the CPU
 * device run the same code.
 */
#ifndef P2VBME_DISTANCES_HPP
#define P2VBME_DISTANCES_HPP

#include <stdint.h>

#include <cmath>
#include <string>

#include "config.hpp"
#include "core.hpp"
#include "eigen_dist.hpp"
#include "host_device.hpp"

namespace p2vbme {

// ---- Models ----------------------------------------------------------------------------------------------

enum class Model : int32_t { JC69, K80, F81, F84, TN93, LogDet, GTR, GTR_GC };

// Canonical name: JC69, K80, F81, F84, TN93, LogDet, GTR, GTR+GC.
const char *model_name(Model model);

// IQ-TREE-style names, case-insensitive: JC69 (JC), K80 (K2P), F81, F84, TN93 (TN), LogDet, GTR+GC.
// Throws std::runtime_error for an unknown name, for +G (continuous gamma is +GC) and for GTR without +GC.
Model parse_model(const std::string &name);

// The substitution model of a run: the model and its parameters.
struct SubstitutionModel {
    Model model = Model::JC69;
    double rates[N_GTR_RATES] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // GTR, GTR+GC exchangeabilities AC, AG, AT, CG, CT, GT
    double alpha = 1.0;                                         // GTR+GC: gamma shape of the rates across sites
    double base_freqs[N_BASES] = {0.0, 0.0, 0.0, 0.0}; // F81, F84, TN93: the alignment's base frequencies (A, C,
                                                       // G, T); all 0: not given (count_base_frequencies)
};

// ---- Pair counts -----------------------------------------------------------------------------------------

struct PairCounts {
    uint32_t n[N_BASES * N_BASES];
};

// Counts of two rows of the 5-bitplane layout (Alignment::packed: planes A, C, G, T, valid; n_words each).
PairCounts count_pair(const uint64_t *row_a, const uint64_t *row_b, int32_t n_words);

// The base frequencies (A, C, G, T) of an alignment in the 5-bitplane layout: each base's share of all valid sites of
// all sequences (as ape's base.freq and FastME).
// Uniform if the alignment has no valid site.
void base_frequencies(const uint64_t *packed, int32_t n_taxa, int32_t n_words, double freqs[N_BASES]);

// ---- Closed-form distances -------------------------------------------------------------------------------

// Each formula assumes a pair that shares a valid site and is not identical, and leaves the range to pair_distance,
// which handles those cases for every model.
// JC69 is the exception: the GPU's fast kernels call it directly, so it checks for itself.
// First the helpers the formulas share, then one formula per model.

// -ln(w) with w floored at LOG_ARG_MIN, and 0 when w >= 1 (the homogeneous link of the paper's C distance library, its
// alpha = 0 case).
// The floor bounds each term at ln(1e4) = 9.21.
template <class T> P2V_HD inline T neg_log_link(T w) {
    if (w >= T(1))
        return T(0);
    if (w < T(LOG_ARG_MIN))
        w = T(LOG_ARG_MIN);
    return -std::log(w);
}

template <class T> P2V_HD inline T clamp_distance(T d) {
    if (d < T(0))
        d = T(0);
    if (d > T(MAX_DISTANCE))
        d = T(MAX_DISTANCE);
    return d;
}

// What the closed forms share: proportions of a pair's sites.
template <class T> struct Proportions {
    T p;      // mismatches
    T P1, P2; // A <-> G and C <-> T differences (transitions)
    T Q;      // transversions, p - P1 - P2
};

template <class T> P2V_HD inline Proportions<T> proportions(const uint32_t n[N_BASES * N_BASES], uint32_t total) {
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

P2V_HD inline uint32_t total_count(const uint32_t n[N_BASES * N_BASES]) {
    uint32_t t = 0;
    for (int k = 0; k < N_BASES * N_BASES; ++k)
        t += n[k];
    return t;
}

P2V_HD inline uint32_t diagonal_count(const uint32_t n[16]) { return n[0] + n[5] + n[10] + n[15]; }

// F81's d = -E ln(1 - p / E), E = 1 - sum pi_i^2 (floored at 1e-4): also F84's and TN93's when the frequencies make
// their own form undefined (a base, or both purines or both pyrimidines, absent from the alignment).
template <class T> P2V_HD inline T f81_form(T p, const T pi[N_BASES]) {
    T E = T(1) - (pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2] + pi[3] * pi[3]);
    if (E < T(1e-4))
        E = T(1e-4);
    return E * neg_log_link(T(1) - p / E);
}

// JC69 from the number of shared valid sites and of matching sites: d = -3/4 ln(1 - 4/3 p), p = mismatches / valid.
// Two counts instead of 16, so the GPU has faster kernels for it.
template <class T> P2V_HD inline T jc69_distance(int32_t n_valid, int32_t n_match) {
    if (n_valid <= 0)
        return T(NO_OVERLAP_DISTANCE);
    const T p = (T)(n_valid - n_match) / (T)n_valid;
    return clamp_distance(T(0.75) * neg_log_link(T(1) - (T(4) / T(3)) * p));
}

// K80 (Kimura 1980): d = -1/2 ln(1 - 2P - Q) - 1/4 ln(1 - 2Q), P = transitions, Q = transversions.
template <class T> P2V_HD inline T k80_distance(const uint32_t n[N_BASES * N_BASES]) {
    const uint32_t total = total_count(n);
    const Proportions<T> s = proportions<T>(n, total);
    const T P = s.P1 + s.P2;
    return T(0.5) * neg_log_link(T(1) - T(2) * P - s.Q) + T(0.25) * neg_log_link(T(1) - T(2) * s.Q);
}

// F81 (Felsenstein 1981; as ape's dist.dna): d = -E ln(1 - p / E), E = 1 - sum pi_i^2, pi the alignment's base
// frequencies.
template <class T> P2V_HD inline T f81_distance(const uint32_t n[N_BASES * N_BASES], const T pi[N_BASES]) {
    const uint32_t total = total_count(n);
    const Proportions<T> s = proportions<T>(n, total);
    return f81_form(s.p, pi);
}

// F84 (Felsenstein & Churchill 1996; as ape's dist.dna): with the alignment's base frequencies pi,
//   A = pi_A pi_G / pi_R + pi_C pi_T / pi_Y,  B = pi_A pi_G + pi_C pi_T,  C = pi_R pi_Y,
//   d = -2A ln(1 - P / (2A) - (A - B) Q / (2AC)) - 2(C + B - A) ln(1 - Q / (2C)).
// Falls back to F81's form when A or C is below FREQ_TERM_MIN.
template <class T> P2V_HD inline T f84_distance(const uint32_t n[N_BASES * N_BASES], const T pi[N_BASES]) {
    const uint32_t total = total_count(n);
    const Proportions<T> s = proportions<T>(n, total);
    const T piA = pi[0], piC = pi[1], piG = pi[2], piT = pi[3];
    const T piR = piA + piG, piY = piC + piT;
    const T A = (piR > T(FREQ_TERM_MIN) ? piA * piG / piR : T(0)) + (piY > T(FREQ_TERM_MIN) ? piC * piT / piY : T(0));
    const T B = piA * piG + piC * piT;
    const T C = piR * piY;
    if (A < T(FREQ_TERM_MIN) || C < T(FREQ_TERM_MIN))
        return f81_form(s.p, pi);
    const T P = s.P1 + s.P2;
    const T g1 = neg_log_link(T(1) - P / (T(2) * A) - ((A - B) * s.Q) / (T(2) * A * C));
    const T g2 = neg_log_link(T(1) - s.Q / (T(2) * C));
    return T(2) * A * g1 + T(2) * (C + B - A) * g2;
}

// TN93 (Tamura & Nei 1993; as ape's dist.dna): with the alignment's base frequencies pi, P1 = A <-> G, P2 = C <-> T,
//   k1 = 2 pi_A pi_G / pi_R,  k2 = 2 pi_C pi_T / pi_Y,  k3 = 2 pi_R pi_Y,
//   d = -k1 ln(w1) - k2 ln(w2) - (k3 - k1 pi_Y - k2 pi_R) ln(w3),
//   w1 = 1 - P1 / k1 - Q / (2 pi_R),  w2 = 1 - P2 / k2 - Q / (2 pi_Y),  w3 = 1 - Q / k3.
// Falls back to F81's form when k1, k2 or k3 is below FREQ_TERM_MIN.
template <class T> P2V_HD inline T tn93_distance(const uint32_t n[N_BASES * N_BASES], const T pi[N_BASES]) {
    const uint32_t total = total_count(n);
    const Proportions<T> s = proportions<T>(n, total);
    const T piA = pi[0], piC = pi[1], piG = pi[2], piT = pi[3];
    const T piR = piA + piG, piY = piC + piT;
    const T k1 = (piR > T(FREQ_TERM_MIN)) ? T(2) * piA * piG / piR : T(0);
    const T k2 = (piY > T(FREQ_TERM_MIN)) ? T(2) * piC * piT / piY : T(0);
    const T k3 = T(2) * piR * piY;
    if (k1 < T(FREQ_TERM_MIN) || k2 < T(FREQ_TERM_MIN) || k3 < T(FREQ_TERM_MIN))
        return f81_form(s.p, pi);
    const T g1 = neg_log_link(T(1) - s.P1 / k1 - s.Q / (T(2) * piR));
    const T g2 = neg_log_link(T(1) - s.P2 / k2 - s.Q / (T(2) * piY));
    const T g3 = neg_log_link(T(1) - s.Q / k3);
    return k1 * g1 + k2 * g2 + (k3 - k1 * piY - k2 * piR) * g3;
}

// LogDet / paralinear (Lockhart et al. 1994): d = -1/4 (ln det J - 1/2 ln prod_i F^a_i F^b_i), J the 4 x 4 matrix of
// pair proportions.
// Undefined (MAX_DISTANCE here; NaN in the paper's C distance library) when a base has frequency below 1e-3 in either
// sequence or det J <= EPS.
template <class T> P2V_HD inline T logdet_distance(const uint32_t n[N_BASES * N_BASES]) {
    const uint32_t total = total_count(n);
    const T inv = T(1) / (T)total;
    T J[N_BASES * N_BASES];
    for (int k = 0; k < N_BASES * N_BASES; ++k)
        J[k] = (T)n[k] * inv;
    T prod_f = T(1);
    for (int i = 0; i < N_BASES; ++i) {
        const T fa = J[4 * i] + J[4 * i + 1] + J[4 * i + 2] + J[4 * i + 3];
        const T fb = J[i] + J[4 + i] + J[8 + i] + J[12 + i];
        if (fa < T(1e-3) || fb < T(1e-3))
            return T(MAX_DISTANCE);
        prod_f *= fa * fb;
    }
    // determinant by 2 x 2 minors of the top and bottom row pairs (Laplace expansion)
    const T s0 = J[0] * J[5] - J[1] * J[4], s1 = J[0] * J[6] - J[2] * J[4], s2 = J[0] * J[7] - J[3] * J[4];
    const T s3 = J[1] * J[6] - J[2] * J[5], s4 = J[1] * J[7] - J[3] * J[5], s5 = J[2] * J[7] - J[3] * J[6];
    const T c5 = J[10] * J[15] - J[11] * J[14], c4 = J[9] * J[15] - J[11] * J[13], c3 = J[9] * J[14] - J[10] * J[13];
    const T c2 = J[8] * J[15] - J[11] * J[12], c1 = J[8] * J[14] - J[10] * J[12], c0 = J[8] * J[13] - J[9] * J[12];
    const T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (!(det > T(EPS)) || !(prod_f > T(EPS)))
        return T(MAX_DISTANCE);
    return T(-0.25) * (std::log(det) - T(0.5) * std::log(prod_f));
}

// ---- One pair's distance under the run's model -----------------------------------------------------------

// The model in the precision of a device (the GPU: float).
template <class T> struct DistanceParams {
    Model model;
    T rates[N_GTR_RATES];
    T alpha, inv_alpha;
    T pi[N_BASES]; // base frequencies, normalised (uniform if not given)
};

template <class T> inline DistanceParams<T> distance_params(const SubstitutionModel &m) {
    DistanceParams<T> p;
    p.model = m.model;
    // The exchangeabilities scaled to mean 1, as the paper's fit returns them.
    // The distance depends only on their ratios; the scaling keeps the rate matrix's own normalisation in range.
    double rate_sum = 0.0;
    for (int k = 0; k < N_GTR_RATES; ++k)
        rate_sum += m.rates[k];
    const double rate_scale = N_GTR_RATES / rate_sum;
    for (int k = 0; k < N_GTR_RATES; ++k)
        p.rates[k] = (T)(m.rates[k] * rate_scale);
    p.alpha = (T)m.alpha;
    p.inv_alpha = T(1) / p.alpha; // as the paper's GPU implementation: 1 / alpha in the device's precision
    const double sum = m.base_freqs[0] + m.base_freqs[1] + m.base_freqs[2] + m.base_freqs[3];
    for (int i = 0; i < N_BASES; ++i)
        p.pi[i] = sum > 0.0 ? (T)(m.base_freqs[i] / sum) : T(1.0 / N_BASES);
    return p;
}

// The distance of a pair with counts n under the model.
// What every model shares is handled here: no shared valid site gives NO_OVERLAP_DISTANCE, identical sequences 0, and
// every distance is kept in [0, MAX_DISTANCE].
template <class T> P2V_HD inline T pair_distance(const DistanceParams<T> &p, const uint32_t n[N_BASES * N_BASES]) {
    const uint32_t total = total_count(n);
    const uint32_t same = diagonal_count(n);
    if (total == 0)
        return T(NO_OVERLAP_DISTANCE);
    if (same == total)
        return T(0);
    T d = T(MAX_DISTANCE);
    switch (p.model) {
    case Model::JC69:
        d = jc69_distance<T>((int32_t)total, (int32_t)same);
        break;
    case Model::K80:
        d = k80_distance<T>(n);
        break;
    case Model::F81:
        d = f81_distance<T>(n, p.pi);
        break;
    case Model::F84:
        d = f84_distance<T>(n, p.pi);
        break;
    case Model::TN93:
        d = tn93_distance<T>(n, p.pi);
        break;
    case Model::LogDet:
        d = logdet_distance<T>(n);
        break;
    case Model::GTR:
        d = eigen_distance<T, false>(n, p.rates, p.alpha, p.inv_alpha);
        break;
    case Model::GTR_GC:
        d = eigen_distance<T>(n, p.rates, p.alpha, p.inv_alpha);
        break;
    }
    return clamp_distance(d);
}

} // namespace p2vbme

#endif // P2VBME_DISTANCES_HPP
