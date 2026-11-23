// Unit tests for the alignment before inference and the trees around it: base frequencies, identical sequences, a
// given start tree and the copies put back.
#include <gtest/gtest.h>

#include <set>
#include <stdint.h>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "io_test_helpers.hpp"
#include "processing.hpp"

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

// ---- The alignment ----

// F81: the base frequencies of the whole input, counted before the copies are set aside; given ones are kept, and
// other models get none.
TEST(Processing, BaseFrequenciesOfTheWholeInput) {
    const std::string fa = temp_path("p.fa");
    write_file(fa, ">a\nAAAA\n>b\nAAAA\n>c\nAAAA\n>d\nAAAA\n>e\nCCCC\n>f\nGGGG\n>g\nTTTA\n");
    p2vbme::Alignment aln = p2vbme::read_fasta(fa);
    p2vbme::SubstitutionModel f81;
    f81.model = p2vbme::Model::F81;
    p2vbme::Deduplication dd;
    p2vbme::prepare_alignment(aln, f81, true, dd);
    EXPECT_EQ(dd.n_removed(), 2);
    EXPECT_EQ(aln.n_taxa, 5);
    const double expect[p2vbme::N_BASES] = {17.0 / 28, 4.0 / 28, 4.0 / 28, 3.0 / 28}; // all 7 sequences, not the 5 kept
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        EXPECT_DOUBLE_EQ(f81.base_freqs[i], expect[i]) << i;

    p2vbme::SubstitutionModel given = f81;
    given.base_freqs[0] = 0.1, given.base_freqs[1] = 0.2, given.base_freqs[2] = 0.3, given.base_freqs[3] = 0.4;
    p2vbme::count_base_frequencies(aln, given);
    EXPECT_EQ(given.base_freqs[3], 0.4);
    p2vbme::SubstitutionModel jc;
    p2vbme::count_base_frequencies(aln, jc);
    for (double f : jc.base_freqs)
        EXPECT_EQ(f, 0.0);
}

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
    EXPECT_EQ(red.packed.size(), (size_t)6 * p2vbme::N_PLANES * red.n_words);
    for (int k = 0; k < 6; ++k) // each kept row is the input row it came from
        for (int w = 0; w < p2vbme::N_PLANES * red.n_words; ++w)
            EXPECT_EQ(red.packed[(size_t)k * p2vbme::N_PLANES * red.n_words + w],
                      aln.packed[(size_t)dd.kept[k] * p2vbme::N_PLANES * aln.n_words + w]);
    p2vbme::Alignment same = p2vbme::read_fasta(fa); // nothing to remove: unchanged
    write_file(fa, ">a\nACGT\n>b\nACGT\n>c\nTTTT\n");
    same = p2vbme::read_fasta(fa);
    const std::vector<uint64_t> before = same.packed;
    p2vbme::deduplicate(same, dd);
    EXPECT_EQ(dd.n_removed(), 0);
    EXPECT_EQ(same.packed, before);
}

// ---- Trees ----

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

// A start tree on all names is pruned to the kept sequences, and the copies come back after; without copies set
// aside (or without dedup) both leave the tree as it is.
TEST(Processing, StartTreeAndCopies) {
    const std::string fa = temp_path("q.fa");
    write_file(fa, ">a\nACGT\n>b\nACGT\n>c\nACGT\n>d\nTTTT\n>e\nGGGG\n");
    p2vbme::Alignment aln = p2vbme::read_fasta(fa);
    p2vbme::SubstitutionModel jc;
    p2vbme::Deduplication dd;
    p2vbme::prepare_alignment(aln, jc, true, dd);
    ASSERT_EQ(dd.n_removed(), 1); // c
    // (((a,c),b),(d,e)): leaves 0..4, nodes 5 = (0,2), 6 = (5,1), 7 = (3,4), root 8 = (6,7)
    p2vbme::Tree full;
    full.c1 = {-1, -1, -1, -1, -1, 0, 5, 3, 6};
    full.c2 = {-1, -1, -1, -1, -1, 2, 1, 4, 7};
    p2vbme::Tree start = full;
    p2vbme::prepare_start_tree(start, dd);
    EXPECT_EQ(clades(start, 4), (std::set<std::set<int>>{{0, 1}, {2, 3}})); // c gone: (a,b),(d,e) on the kept rows
    p2vbme::Tree back = start;
    p2vbme::restore_copies(back, dd);
    EXPECT_EQ(back.c1.size(), full.c1.size());
    EXPECT_EQ(clades(back, 5), (std::set<std::set<int>>{{0, 2}, {0, 1, 2}, {3, 4}})); // c next to its first copy a

    p2vbme::Deduplication none;
    p2vbme::Tree same = full;
    p2vbme::prepare_start_tree(same, none);
    p2vbme::restore_copies(same, none);
    EXPECT_EQ(same.c1, full.c1);
    EXPECT_EQ(same.c2, full.c2);
}
