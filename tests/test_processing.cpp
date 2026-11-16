// Unit tests for the input before inference and the tree after it.
#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "io_test_helpers.hpp"
#include "processing.hpp"

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
    const double expect[4] = {17.0 / 28, 4.0 / 28, 4.0 / 28, 3.0 / 28}; // all 7 sequences, not the 5 kept
    for (int i = 0; i < 4; ++i)
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
