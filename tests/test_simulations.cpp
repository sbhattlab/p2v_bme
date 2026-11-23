// Tests of src/simulations: transition probabilities against closed forms, random trees, simulated alignments.
#include <gtest/gtest.h>

#include <math.h>

#include <stdexcept>
#include <vector>

#include "simulations.hpp"

namespace {

using p2vbme::SimulationModel;
using p2vbme::WORD_BITS;

// p-distance between taxa i and j over sites valid in both (bitplanes as in the .bin bundles).
double p_distance(const p2vbme::Alignment &a, int i, int j, int64_t *n_valid = nullptr) {
    const uint64_t *x = &a.packed[(size_t)i * p2vbme::N_PLANES * a.n_words],
                   *y = &a.packed[(size_t)j * p2vbme::N_PLANES * a.n_words];
    int64_t valid = 0, match = 0;
    for (int w = 0; w < a.n_words; ++w) {
        valid += __builtin_popcountll(x[4 * a.n_words + w] & y[4 * a.n_words + w]);
        uint64_t m = 0;
        for (int b = 0; b < p2vbme::N_BASES; ++b)
            m |= x[b * a.n_words + w] & y[b * a.n_words + w];
        match += __builtin_popcountll(m);
    }
    if (n_valid)
        *n_valid = valid;
    return (double)(valid - match) / (double)valid;
}

// Sum of the branch lengths on the path between leaves i and j.
double path_length(const p2vbme::Tree &tree, int32_t i, int32_t j) {
    const int32_t n_nodes = (int32_t)tree.c1.size();
    std::vector<int32_t> parent(n_nodes, -1);
    for (int32_t m = (n_nodes + 1) / 2; m < n_nodes; ++m)
        parent[tree.c1[m]] = parent[tree.c2[m]] = m;
    std::vector<double> up(n_nodes, -1.0); // distance from i up to each of its ancestors
    double d = 0.0;
    for (int32_t u = i; u != -1; u = parent[u]) {
        up[u] = d;
        if (parent[u] != -1)
            d += tree.branch_lengths[u];
    }
    d = 0.0;
    for (int32_t u = j; u != -1; u = parent[u]) {
        if (up[u] >= 0.0)
            return d + up[u];
        d += tree.branch_lengths[u];
    }
    throw std::logic_error("path_length: no common ancestor");
}

TEST(TransitionMatrix, Jc69ClosedForm) {
    SimulationModel m;
    for (double t : {0.0, 0.01, 0.1, 0.5, 2.0}) {
        double P[p2vbme::N_BASES][p2vbme::N_BASES];
        p2vbme::transition_matrix(m, t, P);
        const double same = 0.25 + 0.75 * exp(-4.0 * t / 3.0), diff = 0.25 - 0.25 * exp(-4.0 * t / 3.0);
        for (int i = 0; i < p2vbme::N_BASES; ++i)
            for (int j = 0; j < p2vbme::N_BASES; ++j)
                EXPECT_NEAR(P[i][j], i == j ? same : diff, 1e-12) << "t=" << t;
    }
}

TEST(TransitionMatrix, K80ClosedForm) {
    const double kappa = 4.0;
    SimulationModel m;
    m.rates[1] = m.rates[4] = kappa; // AG, CT
    const double alpha = kappa / (kappa + 2.0), beta = 1.0 / (kappa + 2.0);
    for (double t : {0.01, 0.1, 0.5, 2.0}) {
        double P[p2vbme::N_BASES][p2vbme::N_BASES];
        p2vbme::transition_matrix(m, t, P);
        const double transversion = 0.25 * (1.0 - exp(-4.0 * beta * t));
        const double transition = 0.25 + 0.25 * exp(-4.0 * beta * t) - 0.5 * exp(-2.0 * (alpha + beta) * t);
        EXPECT_NEAR(P[0][2], transition, 1e-12);   // A -> G
        EXPECT_NEAR(P[1][3], transition, 1e-12);   // C -> T
        EXPECT_NEAR(P[0][1], transversion, 1e-12); // A -> C
        EXPECT_NEAR(P[2][3], transversion, 1e-12); // G -> T
        EXPECT_NEAR(P[0][0], 1.0 - transition - 2.0 * transversion, 1e-12);
    }
}

TEST(TransitionMatrix, F81ClosedForm) {
    SimulationModel m;
    const double pi[p2vbme::N_BASES] = {0.1, 0.2, 0.3, 0.4};
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        m.freqs[i] = pi[i];
    const double mu = 1.0 - (0.01 + 0.04 + 0.09 + 0.16);
    for (double t : {0.01, 0.3, 1.5}) {
        double P[p2vbme::N_BASES][p2vbme::N_BASES];
        p2vbme::transition_matrix(m, t, P);
        const double e = exp(-t / mu);
        for (int i = 0; i < p2vbme::N_BASES; ++i)
            for (int j = 0; j < p2vbme::N_BASES; ++j)
                EXPECT_NEAR(P[i][j], pi[j] * (1.0 - e) + (i == j ? e : 0.0), 1e-12);
    }
}

TEST(TransitionMatrix, GtrIsAStochasticReversibleSemigroup) {
    SimulationModel m;
    const double rates[p2vbme::N_GTR_RATES] = {0.5, 2.0, 0.3, 0.7, 3.1, 1.0},
                 pi[p2vbme::N_BASES] = {0.35, 0.15, 0.2, 0.3};
    for (int k = 0; k < p2vbme::N_GTR_RATES; ++k)
        m.rates[k] = rates[k];
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        m.freqs[i] = pi[i];
    double P0[p2vbme::N_BASES][p2vbme::N_BASES], Ps[p2vbme::N_BASES][p2vbme::N_BASES],
        Pt[p2vbme::N_BASES][p2vbme::N_BASES], Pst[p2vbme::N_BASES][p2vbme::N_BASES],
        Pinf[p2vbme::N_BASES][p2vbme::N_BASES];
    p2vbme::transition_matrix(m, 0.0, P0);
    p2vbme::transition_matrix(m, 0.2, Ps);
    p2vbme::transition_matrix(m, 0.7, Pt);
    p2vbme::transition_matrix(m, 0.9, Pst);
    p2vbme::transition_matrix(m, 200.0, Pinf);
    for (int i = 0; i < p2vbme::N_BASES; ++i) {
        double row = 0.0;
        for (int j = 0; j < p2vbme::N_BASES; ++j) {
            EXPECT_NEAR(P0[i][j], i == j ? 1.0 : 0.0, 1e-12);
            EXPECT_GT(Ps[i][j], 0.0);
            row += Ps[i][j];
            EXPECT_NEAR(pi[i] * Ps[i][j], pi[j] * Ps[j][i], 1e-12); // reversibility
            EXPECT_NEAR(Pinf[i][j], pi[j], 1e-9);                   // stationary distribution
            double prod = 0.0;                                      // Chapman-Kolmogorov
            for (int k = 0; k < p2vbme::N_BASES; ++k)
                prod += Ps[i][k] * Pt[k][j];
            EXPECT_NEAR(prod, Pst[i][j], 1e-12);
        }
        EXPECT_NEAR(row, 1.0, 1e-12);
    }
    // One expected substitution per unit time: -sum_i pi_i Q_ii = 1, with Q ~ (P(h) - I) / h.
    const double h = 1e-6;
    double Ph[p2vbme::N_BASES][p2vbme::N_BASES], rate = 0.0;
    p2vbme::transition_matrix(m, h, Ph);
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        rate += pi[i] * (1.0 - Ph[i][i]) / h;
    EXPECT_NEAR(rate, 1.0, 1e-5);
}

TEST(RandomTree, IsAPostOrderBinaryTree) {
    const int n = 50;
    const p2vbme::Tree t = p2vbme::random_tree(n, 0.01, 0.2, 7);
    ASSERT_EQ(t.c1.size(), 2u * n - 1);
    ASSERT_EQ(t.branch_lengths.size(), 2u * n - 1);
    std::vector<int> n_parents(2 * n - 1, 0);
    for (int m = n; m < 2 * n - 1; ++m) {
        EXPECT_LT(t.c1[m], m);
        EXPECT_LT(t.c2[m], m);
        EXPECT_GE(t.c1[m], 0);
        EXPECT_NE(t.c1[m], t.c2[m]);
        ++n_parents[t.c1[m]];
        ++n_parents[t.c2[m]];
    }
    for (int u = 0; u < 2 * n - 2; ++u) {
        EXPECT_EQ(n_parents[u], 1) << u;
        EXPECT_GE(t.branch_lengths[u], 0.01f);
        EXPECT_LE(t.branch_lengths[u], 0.2f);
    }
    EXPECT_EQ(n_parents[2 * n - 2], 0);
    const p2vbme::Tree again = p2vbme::random_tree(n, 0.01, 0.2, 7);
    EXPECT_EQ(again.c1, t.c1);
    EXPECT_EQ(again.branch_lengths, t.branch_lengths);
    EXPECT_NE(p2vbme::random_tree(n, 0.01, 0.2, 8).c1, t.c1);
}

TEST(PathLength, SmallTree) {
    // ((0:1,1:2):3,2:4); internal 3 = (0,1), root 4 = (3,2)
    p2vbme::Tree t;
    t.c1 = {-1, -1, -1, 0, 3};
    t.c2 = {-1, -1, -1, 1, 2};
    t.branch_lengths = {1.0f, 2.0f, 4.0f, 3.0f, 0.0f};
    EXPECT_DOUBLE_EQ(path_length(t, 0, 1), 3.0);
    EXPECT_DOUBLE_EQ(path_length(t, 0, 2), 8.0);
    EXPECT_DOUBLE_EQ(path_length(t, 2, 1), 9.0);
    EXPECT_DOUBLE_EQ(path_length(t, 1, 1), 0.0);
}

TEST(Simulate, LayoutAndDeterminism) {
    const p2vbme::Tree t = p2vbme::random_tree(20, 0.05, 0.3, 3);
    SimulationModel m;
    m.alpha = 0.7;
    const int L = 130; // not a multiple of WORD_BITS: the last word has unused bits
    const p2vbme::Alignment a = p2vbme::simulate_alignment(t, L, m, 11);
    ASSERT_EQ(a.n_taxa, 20);
    ASSERT_EQ(a.n_sites, L);
    ASSERT_EQ(a.n_words, 3);
    ASSERT_EQ(a.packed.size(), 20u * p2vbme::N_PLANES * 3);
    ASSERT_EQ(a.names.size(), 20u);
    EXPECT_EQ(a.names[0], "t0");
    EXPECT_EQ(a.names[19], "t19");
    for (int i = 0; i < a.n_taxa; ++i)
        for (int s = 0; s < 3 * WORD_BITS; ++s) {
            const int w = s / WORD_BITS, k = s % WORD_BITS; // bit k of word w
            int n_bases = 0;
            for (int b = 0; b < p2vbme::N_BASES; ++b)
                n_bases += (a.packed[((size_t)i * p2vbme::N_PLANES + b) * 3 + w] >> k) & 1;
            const int valid = (a.packed[((size_t)i * p2vbme::N_PLANES + p2vbme::VALID_PLANE) * 3 + w] >> k) & 1;
            EXPECT_EQ(n_bases, s < L ? 1 : 0);
            EXPECT_EQ(valid, s < L ? 1 : 0);
        }
    EXPECT_EQ(p2vbme::simulate_alignment(t, L, m, 11).packed, a.packed);
    EXPECT_NE(p2vbme::simulate_alignment(t, L, m, 12).packed, a.packed);
}

// JC69 distances of simulated sequences estimate the path lengths (within 4 standard errors).
TEST(Simulate, Jc69DistancesMatchPathLengths) {
    const int n = 8, L = 200000;
    const p2vbme::Tree t = p2vbme::random_tree(n, 0.02, 0.25, 21);
    const p2vbme::Alignment a = p2vbme::simulate_alignment(t, L, SimulationModel(), 5);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double p = p_distance(a, i, j), d = -0.75 * log(1.0 - 4.0 * p / 3.0);
            const double se = sqrt(p * (1.0 - p) / L) / (1.0 - 4.0 * p / 3.0);
            EXPECT_NEAR(d, path_length(t, i, j), 4.0 * se) << i << "," << j;
        }
}

TEST(Simulate, BaseFrequencies) {
    SimulationModel m;
    const double pi[p2vbme::N_BASES] = {0.1, 0.2, 0.3, 0.4};
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        m.freqs[i] = pi[i];
    m.rates[1] = 3.0;
    const int n = 10, L = 50000;
    const p2vbme::Alignment a = p2vbme::simulate_alignment(p2vbme::random_tree(n, 0.05, 0.5, 2), L, m, 9);
    for (int b = 0; b < p2vbme::N_BASES; ++b) {
        int64_t count = 0;
        for (int i = 0; i < n; ++i)
            for (int w = 0; w < a.n_words; ++w)
                count += __builtin_popcountll(a.packed[((size_t)i * p2vbme::N_PLANES + b) * a.n_words + w]);
        const double f = (double)count / ((double)n * L);
        // leaves are correlated through the tree, so allow a wider band than for independent sites
        EXPECT_NEAR(f, pi[b], 6.0 * sqrt(pi[b] * (1.0 - pi[b]) / L)) << "base " << b;
    }
}

// Continuous gamma: the expected JC69 p-distance at path length d is 3/4 (1 - (1 + 4 d / (3 alpha))^-alpha).
TEST(Simulate, ContinuousGammaMatchesJc69GammaPDistance) {
    const double alpha = 0.5;
    p2vbme::Tree t; // two leaves, path length 0.2 + 0.4
    t.c1 = {-1, -1, 0};
    t.c2 = {-1, -1, 1};
    t.branch_lengths = {0.2f, 0.4f, 0.0f};
    SimulationModel m;
    m.alpha = alpha;
    const int L = 400000;
    const p2vbme::Alignment a = p2vbme::simulate_alignment(t, L, m, 17);
    const double d = 0.6, expected = 0.75 * (1.0 - pow(1.0 + 4.0 * d / (3.0 * alpha), -alpha));
    const double p = p_distance(a, 0, 1);
    EXPECT_NEAR(p, expected, 4.0 * sqrt(expected * (1.0 - expected) / L));
    // without rate variation the p-distance is clearly larger
    EXPECT_GT(0.75 * (1.0 - exp(-4.0 * d / 3.0)), expected + 0.05);
}

TEST(Simulate, RejectsBadInput) {
    const p2vbme::Tree t = p2vbme::random_tree(4, 0.1, 0.1, 1);
    SimulationModel bad_freq;
    bad_freq.freqs[2] = 0.0;
    EXPECT_THROW(p2vbme::simulate_alignment(t, 10, bad_freq, 1), std::invalid_argument);
    SimulationModel bad_rate;
    bad_rate.rates[0] = -1.0;
    EXPECT_THROW(p2vbme::simulate_alignment(t, 10, bad_rate, 1), std::invalid_argument);
    EXPECT_THROW(p2vbme::simulate_alignment(t, 0, SimulationModel(), 1), std::invalid_argument);
    p2vbme::Tree no_lengths = t;
    no_lengths.branch_lengths.clear();
    EXPECT_THROW(p2vbme::simulate_alignment(no_lengths, 10, SimulationModel(), 1), std::invalid_argument);
    EXPECT_THROW(p2vbme::random_tree(1, 0.1, 0.2, 1), std::invalid_argument);
    EXPECT_THROW(p2vbme::random_tree(5, 0.3, 0.2, 1), std::invalid_argument);
}

} // namespace
