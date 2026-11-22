// Tests of src/phylo2vec: ordered vectors (phylo2vec's ordered trees, 0 <= v[i] <= i) of rooted trees.
//   - to_skeleton decodes as phylo2vec's to_ancestry (reference rows from the phylo2vec crate 1.7.0), children
//     swapped, and children get smaller ids than their parent;
//   - queue_shuffle_rooted encodes any rooted tree, whatever its internal ids, as an ordered vector with a label
//     mapping, in both labellings, and decoding gives the tree back;
//   - at each node the child holding the smaller leaf keeps the label, or with flip the other child;
//   - with flip, on the tree of an ordered vector, it gives phylo2vec's queue_shuffle (reference values from the
//     phylo2vec crate 1.7.0).
#include <gtest/gtest.h>

#include <stdint.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "phylo2vec.hpp"

namespace {

using p2vbme::queue_shuffle_rooted;
using p2vbme::to_skeleton;

// Builds a random rooted tree on n leaves by random joins: (c1, c2), root 2n - 2, the other internal ids shuffled.
void random_rooted_tree(int32_t n, std::mt19937_64 &rng, std::vector<int32_t> &c1, std::vector<int32_t> &c2) {
    std::vector<int32_t> a(2 * n - 1, 0), b(2 * n - 1, 0), pool(n);
    std::iota(pool.begin(), pool.end(), 0);
    int32_t next = n;
    while (pool.size() > 1) {
        const size_t i = rng() % pool.size();
        const int32_t x = pool[i];
        pool[i] = pool.back();
        pool.pop_back();
        const size_t j = rng() % pool.size();
        a[next] = x;
        b[next] = pool[j];
        pool[j] = next++;
    }
    // shuffle the internal ids n .. 2n - 3 (the root keeps 2n - 2), as the search leaves them
    std::vector<int32_t> id(2 * n - 1);
    std::iota(id.begin(), id.end(), 0);
    if (n > 2)
        std::shuffle(id.begin() + n, id.end() - 1, rng);
    c1.assign(2 * n - 1, 0);
    c2.assign(2 * n - 1, 0);
    for (int32_t u = n; u < 2 * n - 1; ++u) {
        c1[id[u]] = id[a[u]];
        c2[id[u]] = id[b[u]];
    }
}

// Lists the clades of a rooted tree: one sorted set of tip labels (label[tip]) per internal node.
std::set<std::vector<int32_t>> clades(const std::vector<int32_t> &c1, const std::vector<int32_t> &c2, int32_t n,
                                      const std::vector<int32_t> &label) {
    std::set<std::vector<int32_t>> out;
    for (int32_t u = n; u < 2 * n - 1; ++u) {
        std::vector<int32_t> tips, st = {u};
        while (!st.empty()) {
            const int32_t x = st.back();
            st.pop_back();
            if (x < n) {
                tips.push_back(label[x]);
            } else {
                st.push_back(c1[x]);
                st.push_back(c2[x]);
            }
        }
        std::sort(tips.begin(), tips.end());
        out.insert(tips);
    }
    return out;
}

} // namespace

// to_skeleton gives phylo2vec's to_ancestry rows {a, b, parent} as c1[parent] = b, c2[parent] = a
TEST(Phylo2Vec, ToSkeletonMatchesPhylo2vecAncestry) {
    struct Case {
        std::vector<int64_t> v;
        std::vector<std::array<int32_t, 3>> ancestry; // phylo2vec 1.7.0, vector::convert::to_ancestry
    };
    const std::vector<Case> cases = {
        {{0, 0, 1}, {{1, 3, 4}, {0, 2, 5}, {5, 4, 6}}},
        {{0, 0, 0, 1}, {{1, 4, 5}, {0, 3, 6}, {6, 2, 7}, {7, 5, 8}}},
        {{0, 1, 2, 3, 4}, {{4, 5, 6}, {3, 6, 7}, {2, 7, 8}, {1, 8, 9}, {0, 9, 10}}},
        {{0, 0, 0, 0, 0, 0}, {{0, 6, 7}, {7, 5, 8}, {8, 4, 9}, {9, 3, 10}, {10, 2, 11}, {11, 1, 12}}},
        {{0, 1, 0, 2, 1, 5, 3}, {{3, 7, 8}, {5, 6, 9}, {1, 9, 10}, {2, 4, 11}, {0, 8, 12}, {10, 11, 13}, {12, 13, 14}}},
    };
    for (const Case &c : cases) {
        const int32_t n = (int32_t)c.v.size() + 1;
        std::vector<int32_t> c1(2 * n - 1, -1), c2(2 * n - 1, -1);
        to_skeleton(c.v.data(), n, c1.data(), c2.data());
        ASSERT_EQ((int32_t)c.ancestry.size(), n - 1);
        for (const auto &row : c.ancestry) {
            EXPECT_EQ(c1[row[2]], row[1]) << "parent " << row[2];
            EXPECT_EQ(c2[row[2]], row[0]) << "parent " << row[2];
        }
        for (int32_t x = 0; x < n; ++x) {
            EXPECT_EQ(c1[x], 0);
            EXPECT_EQ(c2[x], 0);
        }
    }
}

// Every ordered vector decodes into a rooted tree whose children have smaller ids than their parent
TEST(Phylo2Vec, ToSkeletonChildrenBeforeParents) {
    std::mt19937_64 rng(3);
    for (int32_t n : {2, 3, 4, 7, 50, 1000}) {
        for (int rep = 0; rep < 50; ++rep) {
            std::vector<int64_t> v(n - 1);
            for (int32_t i = 0; i < n - 1; ++i)
                v[i] = (int64_t)(rng() % (uint64_t)(i + 1));
            std::vector<int32_t> c1(2 * n - 1), c2(2 * n - 1);
            to_skeleton(v.data(), n, c1.data(), c2.data());
            std::vector<int32_t> seen(2 * n - 1, 0);
            for (int32_t m = n; m < 2 * n - 1; ++m) {
                EXPECT_LT(c1[m], m);
                EXPECT_LT(c2[m], m);
                ++seen[c1[m]];
                ++seen[c2[m]];
            }
            for (int32_t x = 0; x < 2 * n - 2; ++x)
                EXPECT_EQ(seen[x], 1) << "node " << x << " has " << seen[x] << " parents";
        }
    }
}

// queue_shuffle_rooted gives an ordered vector and a label mapping, and decoding gives the tree back: the same clades
// once each decoded tip a is read as the original leaf perm[a]; in both labellings
TEST(Phylo2Vec, QueueShuffleRoundTrip) {
    std::mt19937_64 rng(7);
    long flips_differ = 0;
    for (int32_t n : {2, 3, 4, 5, 10, 100, 1000}) {
        for (int rep = 0; rep < (n > 100 ? 20 : 200); ++rep) {
            std::vector<int32_t> c1, c2;
            random_rooted_tree(n, rng, c1, c2);
            std::vector<int32_t> identity(n);
            std::iota(identity.begin(), identity.end(), 0);
            const auto want = clades(c1, c2, n, identity);
            std::vector<int64_t> v_flip[2];
            for (const bool flip : {false, true}) {
                std::vector<int64_t> v(n - 1, -1);
                std::vector<int32_t> perm(n, -1);
                queue_shuffle_rooted(c1.data(), c2.data(), n, flip, v.data(), perm.data());
                for (int32_t i = 0; i < n - 1; ++i) {
                    EXPECT_GE(v[i], 0);
                    EXPECT_LE(v[i], i) << "not ordered at row " << i;
                }
                std::vector<int32_t> sorted = perm;
                std::sort(sorted.begin(), sorted.end());
                EXPECT_EQ(sorted, identity) << "perm is not a permutation";
                std::vector<int32_t> d1(2 * n - 1), d2(2 * n - 1);
                to_skeleton(v.data(), n, d1.data(), d2.data());
                EXPECT_EQ(clades(d1, d2, n, perm), want) << "n " << n << " flip " << flip;
                v_flip[flip] = v;
            }
            flips_differ += v_flip[0] != v_flip[1];
        }
    }
    EXPECT_GT(flips_differ, 0) << "the two labellings never gave different vectors";
}

// Without flip, the child holding the smaller leaf keeps its parent's label; with flip, the other child.
// In the decoded tree the child that kept the label is c2 (to_skeleton), so the smallest original leaf (perm[tip]) on
// c2's side is below c1's at every node without flip, above it with flip.
TEST(Phylo2Vec, QueueShuffleFlipChoosesWhichChildKeepsTheLabel) {
    std::mt19937_64 rng(11);
    for (int32_t n : {2, 3, 4, 5, 10, 100, 1000}) {
        for (int rep = 0; rep < (n > 100 ? 20 : 200); ++rep) {
            std::vector<int32_t> c1, c2;
            random_rooted_tree(n, rng, c1, c2);
            for (const bool flip : {false, true}) {
                std::vector<int64_t> v(n - 1);
                std::vector<int32_t> perm(n);
                queue_shuffle_rooted(c1.data(), c2.data(), n, flip, v.data(), perm.data());
                std::vector<int32_t> d1(2 * n - 1), d2(2 * n - 1);
                to_skeleton(v.data(), n, d1.data(), d2.data());
                // smallest original leaf below each node of the decoded tree (children have smaller ids)
                std::vector<int32_t> min_leaf(2 * n - 1);
                for (int32_t a = 0; a < n; ++a)
                    min_leaf[a] = perm[a];
                for (int32_t m = n; m < 2 * n - 1; ++m) {
                    min_leaf[m] = std::min(min_leaf[d1[m]], min_leaf[d2[m]]);
                    if (flip)
                        EXPECT_GT(min_leaf[d2[m]], min_leaf[d1[m]]) << "n " << n << " node " << m;
                    else
                        EXPECT_LT(min_leaf[d2[m]], min_leaf[d1[m]]) << "n " << n << " node " << m;
                }
            }
        }
    }
}

// With flip, queue_shuffle_rooted on the tree of an ordered vector gives phylo2vec's queue_shuffle(v, false): the same
// vector and label mapping (phylo2vec crate 1.7.0, vector::ops::queue_shuffle; the last two vectors drawn at random)
TEST(Phylo2Vec, QueueShuffleWithFlipMatchesPhylo2vec) {
    struct Case {
        std::vector<int64_t> v;
        std::vector<int64_t> v_qs;
        std::vector<int32_t> label_mapping;
    };
    const std::vector<Case> cases = {
        {{0, 0, 1}, {0, 0, 1}, {3, 2, 1, 0}},
        {{0, 0, 0, 1}, {0, 0, 1, 3}, {4, 2, 1, 3, 0}},
        {{0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, {5, 0, 1, 2, 3, 4}},
        {{0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 0}},
        {{0, 1, 0, 2, 1, 5, 3}, {0, 0, 1, 0, 2, 1, 2}, {4, 7, 6, 0, 2, 1, 3, 5}},
        {{0, 1, 1, 0, 3, 2, 6, 1, 4, 9, 7}, {0, 0, 1, 0, 2, 1, 0, 2, 5, 1, 0}, {11, 10, 5, 0, 2, 8, 4, 6, 3, 1, 9, 7}},
        {{0, 0, 2, 1, 4, 4, 0, 7, 3, 2, 10, 5, 12, 8, 1},
         {0, 0, 1, 0, 2, 1, 3, 0, 4, 1, 6, 3, 0, 6, 3},
         {13, 9, 15, 14, 6, 1, 11, 0, 5, 4, 3, 2, 7, 12, 10, 8}},
    };
    for (const Case &c : cases) {
        const int32_t n = (int32_t)c.v.size() + 1;
        std::vector<int32_t> c1(2 * n - 1), c2(2 * n - 1);
        to_skeleton(c.v.data(), n, c1.data(), c2.data());
        std::vector<int64_t> v(n - 1);
        std::vector<int32_t> perm(n);
        queue_shuffle_rooted(c1.data(), c2.data(), n, true, v.data(), perm.data());
        EXPECT_EQ(v, c.v_qs) << "n " << n;
        EXPECT_EQ(perm, c.label_mapping) << "n " << n;
    }
}
