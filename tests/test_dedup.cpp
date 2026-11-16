// Unit tests for identical sequences: which copies are kept, reattach and prune.
#include <gtest/gtest.h>

#include <set>
#include <stdint.h>
#include <string>
#include <vector>

#include "dedup.hpp"
#include "fasta.hpp"
#include "io_test_helpers.hpp"

namespace {

// Every node of a post-order tree on n leaves (root 2n-2) has one parent; children come before parents.
bool valid_tree(const p2vbme::Tree &t, int n) {
    if ((int)t.c1.size() != 2 * n - 1)
        return false;
    std::vector<int> parents(2 * n - 1, 0);
    for (int u = n; u < 2 * n - 1; ++u) {
        if (t.c1[u] < 0 || t.c2[u] < 0 || t.c1[u] >= u || t.c2[u] >= u)
            return false;
        ++parents[t.c1[u]];
        ++parents[t.c2[u]];
    }
    for (int u = 0; u < 2 * n - 2; ++u)
        if (parents[u] != 1)
            return false;
    return parents[2 * n - 2] == 0;
}

} // namespace

// Groups of 1..4 identical rows: the first two copies stay; same bases but different missing sites differ.
TEST(Dedup, KeepsTwoCopiesPerGroup) {
    const std::string fa = temp_path("d.fa");
    write_file(fa, ">a\nACGT\n>b\nACGT\n>c\nACG-\n>d\nACGT\n>e\nTTTT\n>f\nACGT\n>g\nTTTT\n>h\nGGGG\n>i\nTTTT\n");
    const p2vbme::Alignment aln = p2vbme::read_fasta(fa);
    p2vbme::Deduplication dd;
    p2vbme::Alignment red = aln;
    p2vbme::deduplicate(red, dd);
    // ACGT: a b d f -> a b kept; ACG-: c; TTTT: e g i -> e g kept; GGGG: h
    EXPECT_EQ(dd.kept, (std::vector<int32_t>{0, 1, 2, 4, 6, 7}));
    EXPECT_EQ(dd.copy_of, (std::vector<int32_t>{-1, -1, -1, 0, -1, 0, -1, -1, 3}));
    EXPECT_EQ(dd.n_removed(), 3);
    EXPECT_EQ(red.names, (std::vector<std::string>{"a", "b", "c", "e", "g", "h"}));
    EXPECT_EQ(red.n_taxa, 6);
    EXPECT_EQ(red.packed.size(), (size_t)6 * 5 * red.n_words);
    for (int k = 0; k < 6; ++k) // each kept row is the input row it came from
        for (int w = 0; w < 5 * red.n_words; ++w)
            EXPECT_EQ(red.packed[(size_t)k * 5 * red.n_words + w],
                      aln.packed[(size_t)dd.kept[k] * 5 * aln.n_words + w]);
    p2vbme::Alignment same = p2vbme::read_fasta(fa); // nothing to remove: unchanged
    write_file(fa, ">a\nACGT\n>b\nACGT\n>c\nTTTT\n");
    same = p2vbme::read_fasta(fa);
    const std::vector<uint64_t> before = same.packed;
    p2vbme::deduplicate(same, dd);
    EXPECT_EQ(dd.n_removed(), 0);
    EXPECT_EQ(same.packed, before);
}

// reattach puts every removed copy back next to its first copy with zero-length branches; prune undoes it.
TEST(Dedup, ReattachAndPrune) {
    p2vbme::Deduplication dd;
    dd.kept = {0, 1, 2, 4};              // 6 sequences, 4 kept
    dd.copy_of = {-1, -1, -1, 0, -1, 0}; // 3 and 5 are copies of kept position 0 (sequence 0)
    p2vbme::Tree red;                    // ((0,1),(2,3)) on the kept positions
    red.c1 = {-1, -1, -1, -1, 0, 2, 4};
    red.c2 = {-1, -1, -1, -1, 1, 3, 5};
    red.branch_lengths = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}; // 2M - 2 edges, as compute_bme_branch_lengths
    const p2vbme::Tree full = p2vbme::reattach(red, dd);
    ASSERT_TRUE(valid_tree(full, 6));
    EXPECT_EQ(full.branch_lengths.size(), 10u); // 2N - 2
    // sequence 0's former edge (0.1) is above the chain ((0,3),5); the copies' edges are 0
    EXPECT_EQ(full.branch_lengths[0], 0.0f);
    EXPECT_EQ(full.branch_lengths[3], 0.0f);
    EXPECT_EQ(full.branch_lengths[5], 0.0f);
    std::set<std::set<int>> c = clades(full, 6);
    EXPECT_TRUE(c.count({0, 3}));
    EXPECT_TRUE(c.count({0, 3, 5}));
    EXPECT_TRUE(c.count({0, 3, 5, 1}));
    EXPECT_TRUE(c.count({2, 4}));
    float sum_red = 0.0f, sum_full = 0.0f;
    for (float x : red.branch_lengths)
        sum_red += x;
    for (float x : full.branch_lengths)
        sum_full += x;
    EXPECT_FLOAT_EQ(sum_full, sum_red);
    // prune removes the copies again: the same clades as the reduced tree
    const p2vbme::Tree back = p2vbme::prune(full, dd);
    ASSERT_TRUE(valid_tree(back, 4));
    EXPECT_EQ(clades(back, 4), clades(red, 4));

    // a search result reuses internal ids: here node 4 is the parent of node 5 (child id > parent id)
    p2vbme::Tree shuffled;
    shuffled.c1 = {-1, -1, -1, -1, 5, 0, 4};
    shuffled.c2 = {-1, -1, -1, -1, 1, 2, 3};
    shuffled.branch_lengths = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    const p2vbme::Tree full2 = p2vbme::reattach(shuffled, dd); // (((((0,3),5),2),1),4)
    ASSERT_TRUE(valid_tree(full2, 6));
    std::set<std::set<int>> c2 = clades(full2, 6);
    EXPECT_TRUE(c2.count({0, 3, 5}));
    EXPECT_TRUE(c2.count({0, 3, 5, 2}));
    EXPECT_TRUE(c2.count({0, 3, 5, 2, 1}));
    EXPECT_EQ(p2vbme::prune(full2, dd).c1.size(), 7u);
}
