/*
 * test_support.cpp
 *
 * The supports of a reference tree's branches (support.hpp) against a brute force that compares every split of the
 * reference with every subtree of each replicate, on random trees (also with the internal ids shuffled, as the search
 * leaves them); identical replicates; the root's branch; the format.
 */
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "random.hpp"
#include "simulations.hpp"
#include "support.hpp"

using namespace vecme;

namespace {

// The leaves below each node: (tree) -> membership[node][leaf].
std::vector<std::vector<uint8_t>> leaf_sets(const Tree &t) {
    const int32_t K = t.n_leaves(), n_nodes = 2 * K - 1;
    std::vector<std::vector<uint8_t>> below(n_nodes, std::vector<uint8_t>(K, 0));
    std::vector<int32_t> order = {n_nodes - 1}; // parents before children
    for (size_t i = 0; i < order.size(); ++i)
        if (order[i] >= K) {
            order.push_back(t.c1[order[i]]);
            order.push_back(t.c2[order[i]]);
        }
    for (int32_t i = (int32_t)order.size() - 1; i >= 0; --i) {
        const int32_t u = order[i];
        if (u < K)
            below[u][u] = 1;
        else
            for (int32_t l = 0; l < K; ++l)
                below[u][l] = below[t.c1[u]][l] | below[t.c2[u]][l];
    }
    return below;
}

// The supports by definition: for each reference split S, delta = min over the replicate's non-root nodes C of
// min(|S xor C|, K - |S xor C|); TBE = 100 (1 - mean delta / (p - 1)), exact = share of delta = 0; p < 2: none.
void brute_force(const Tree &ref, const std::vector<Tree> &reps, std::vector<double> &tbe, std::vector<double> &exact) {
    const int32_t K = ref.n_leaves(), n_nodes = 2 * K - 1, root = n_nodes - 1;
    const auto s_ref = leaf_sets(ref);
    std::vector<int64_t> sum(n_nodes, 0);
    std::vector<int32_t> same(n_nodes, 0);
    for (const Tree &rep : reps) {
        const auto s_rep = leaf_sets(rep);
        for (int32_t u = K; u < root; ++u) {
            int32_t best = K;
            for (int32_t v = 0; v < root; ++v) {
                int32_t diff = 0;
                for (int32_t l = 0; l < K; ++l)
                    diff += s_ref[u][l] != s_rep[v][l];
                best = std::min(best, std::min(diff, K - diff));
            }
            sum[u] += best;
            same[u] += best == 0;
        }
    }
    tbe.assign(n_nodes, NO_SUPPORT);
    exact.assign(n_nodes, NO_SUPPORT);
    const double B = (double)reps.size();
    for (int32_t u = K; u < root; ++u) {
        const int32_t s = std::accumulate(s_ref[u].begin(), s_ref[u].end(), 0), p = std::min(s, K - s);
        if (p < 2)
            continue;
        exact[u] = 100.0 * same[u] / B;
        tbe[u] = 100.0 * (1.0 - ((double)sum[u] / B) / (double)(p - 1));
    }
}

// The same tree with its internal ids (all but the root's) shuffled, as the search leaves them.
Tree shuffle_internal_ids(const Tree &t, uint64_t seed) {
    const int32_t K = t.n_leaves(), n_nodes = 2 * K - 1, root = n_nodes - 1;
    std::vector<int32_t> ids(root - K);
    std::iota(ids.begin(), ids.end(), K);
    uint64_t state = seed;
    shuffle(ids.begin(), ids.end(), state);
    std::vector<int32_t> to(n_nodes);
    std::iota(to.begin(), to.end(), 0);
    for (int32_t i = 0; i < root - K; ++i)
        to[K + i] = ids[i];
    Tree out;
    out.c1.assign(n_nodes, -1);
    out.c2.assign(n_nodes, -1);
    for (int32_t u = K; u < n_nodes; ++u) {
        out.c1[to[u]] = to[t.c1[u]];
        out.c2[to[u]] = to[t.c2[u]];
    }
    return out;
}

// One node's support, compared: NO_SUPPORT on both sides or equal up to rounding.
void expect_support(double got, double want, const char *what, int32_t u) {
    if (want == NO_SUPPORT)
        EXPECT_EQ(got, NO_SUPPORT) << what << " at node " << u;
    else
        EXPECT_NEAR(got, want, 1e-9) << what << " at node " << u;
}

} // namespace

/* Random reference and replicate trees, from 4 to 300 leaves, some with shuffled internal ids: the supports equal the
 * brute force's. */
TEST(Support, EqualsTheBruteForce) {
    for (int32_t K : {4, 5, 8, 17, 40, 300}) {
        SCOPED_TRACE("K = " + std::to_string(K));
        const Tree ref = shuffle_internal_ids(random_tree(K, 0.1, 0.1, 1000 + K), 7);
        std::vector<Tree> reps;
        for (int r = 0; r < 6; ++r) {
            const Tree t = random_tree(K, 0.1, 0.1, 2000 + 10 * K + r);
            reps.push_back(r % 2 ? shuffle_internal_ids(t, 100 + r) : t);
        }
        reps.push_back(ref); // one replicate equal to the reference
        SupportCounter counter(ref);
        for (const Tree &t : reps)
            counter.add(t);
        EXPECT_EQ(counter.n_replicates(), (int32_t)reps.size());
        std::vector<double> tbe, exact;
        brute_force(ref, reps, tbe, exact);
        const std::vector<double> got_tbe = counter.tbe(), got_exact = counter.exact();
        for (int32_t u = 0; u < 2 * K - 1; ++u) {
            expect_support(got_tbe[u], tbe[u], "TBE", u);
            expect_support(got_exact[u], exact[u], "exact", u);
        }
    }
}

/* Replicates identical to the reference (in any id order) support every branch fully; the root's two children, one
 * branch of the unrooted tree, get the same value; the means count that branch once. */
TEST(Support, IdenticalReplicatesAndTheRootBranch) {
    const int32_t K = 30;
    const Tree ref = random_tree(K, 0.1, 0.1, 5);
    SupportCounter counter(ref);
    counter.add(ref);
    counter.add(shuffle_internal_ids(ref, 3));
    const std::vector<double> tbe = counter.tbe(), exact = counter.exact();
    for (int32_t u = 0; u < 2 * K - 1; ++u) {
        if (tbe[u] == NO_SUPPORT)
            continue;
        EXPECT_EQ(tbe[u], 100.0) << u;
        EXPECT_EQ(exact[u], 100.0) << u;
    }
    EXPECT_EQ(counter.mean_tbe(), 100.0);
    EXPECT_EQ(counter.mean_exact(), 100.0);

    // a tree whose root has two internal children, against a different replicate
    Tree r2 = random_tree(K, 0.1, 0.1, 6);
    SupportCounter c2(r2);
    c2.add(random_tree(K, 0.1, 0.1, 8));
    const int32_t root = 2 * K - 2, a = r2.c1[root], b = r2.c2[root];
    if (a >= K && b >= K) {
        EXPECT_EQ(c2.tbe()[a], c2.tbe()[b]);
        EXPECT_EQ(c2.exact()[a], c2.exact()[b]);
    }
}

/* Fewer than 4 leaves, or a replicate on other leaves, are refused. */
TEST(Support, BadInputRefused) {
    EXPECT_THROW(SupportCounter(random_tree(3, 0.1, 0.1, 1)), std::invalid_argument);
    SupportCounter counter(random_tree(10, 0.1, 0.1, 1));
    EXPECT_THROW(counter.add(random_tree(11, 0.1, 0.1, 2)), std::invalid_argument);
    EXPECT_EQ(counter.n_replicates(), 0);
    EXPECT_EQ(counter.tbe()[15], NO_SUPPORT); // no replicate: no support
}

/* Supports are written as the paper's code writes them: at most 2 decimals, no trailing zeros, within [0, 100]. */
TEST(Support, Format) {
    EXPECT_EQ(format_support(97.333333), "97.33");
    EXPECT_EQ(format_support(85.5), "85.5");
    EXPECT_EQ(format_support(100.0), "100");
    EXPECT_EQ(format_support(0.0), "0");
    EXPECT_EQ(format_support(100.0000001), "100");
    EXPECT_EQ(format_support(-0.1), "0");
}
