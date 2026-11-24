// Unit tests for the alignment before inference and the trees around it: base frequencies, the GTR+GC fit,
// identical sequences, a given start tree and the copies put back.
#include <gtest/gtest.h>

#include <set>
#include <stdint.h>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "io_test_helpers.hpp"
#include "processing.hpp"
#include "random.hpp"
#include "simulations.hpp"

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
    p2vbme::prepare_alignment(aln, f81, p2vbme::DEFAULT_GTR_FIT_PAIRS, p2vbme::DEFAULT_SEED, 0, true, dd);
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

// GTR+GC: the rates and alpha are fitted on the whole input, before the copies are set aside (the same fit with and
// without dedup), from the run seed's own stream, and set in the model; GTR: its rates alone, from the same pairs;
// other models get no fit.
TEST(Processing, GtrGcFitOfTheWholeInput) {
    p2vbme::SimulationModel m;
    m.rates[1] = 3.0; // transitions AG and CT
    m.rates[4] = 3.0;
    m.alpha = 0.8;
    p2vbme::Alignment aln = p2vbme::simulate_alignment(p2vbme::random_tree(20, 0.02, 0.2, 5), 400, m, 6);
    // two more copies of t0, so that dedup sets one aside
    const size_t row_words = (size_t)p2vbme::N_PLANES * aln.n_words;
    const std::vector<uint64_t> t0(aln.packed.begin(), aln.packed.begin() + row_words);
    for (int c = 1; c <= 2; ++c) {
        aln.packed.insert(aln.packed.end(), t0.begin(), t0.end());
        aln.names.push_back("t0_copy" + std::to_string(c));
        ++aln.n_taxa;
    }
    const int32_t n_pairs = 100; // fewer than the 231 pairs
    const uint64_t seed = 9;
    const p2vbme::GtrFit whole =
        p2vbme::fit_gtr_gc(aln, n_pairs, p2vbme::derive_seed(seed, p2vbme::SALT_GTR_FIT_PAIRS));
    ASSERT_EQ(whole.n_pairs, n_pairs);
    for (bool dedup : {true, false}) {
        p2vbme::Alignment a = aln;
        p2vbme::SubstitutionModel gtr;
        gtr.model = p2vbme::Model::GTR_GC;
        p2vbme::Deduplication dd;
        const p2vbme::GtrFit fit = p2vbme::prepare_alignment(a, gtr, n_pairs, seed, 0, dedup, dd);
        EXPECT_EQ(dd.n_removed(), dedup ? 1 : 0);
        EXPECT_EQ(fit.alpha, whole.alpha) << "dedup " << dedup;
        EXPECT_EQ(gtr.alpha, whole.alpha) << "dedup " << dedup;
        for (int k = 0; k < p2vbme::N_GTR_RATES; ++k) {
            EXPECT_EQ(fit.rates[k], whole.rates[k]) << "dedup " << dedup << ", rate " << k;
            EXPECT_EQ(gtr.rates[k], whole.rates[k]) << "dedup " << dedup << ", rate " << k;
        }
    }
    const p2vbme::GtrFit whole_gtr =
        p2vbme::fit_gtr(aln, n_pairs, p2vbme::derive_seed(seed, p2vbme::SALT_GTR_FIT_PAIRS));
    p2vbme::Alignment a = aln;
    p2vbme::SubstitutionModel plain;
    plain.model = p2vbme::Model::GTR;
    p2vbme::Deduplication dd;
    const p2vbme::GtrFit fit = p2vbme::prepare_alignment(a, plain, n_pairs, seed, 0, true, dd);
    EXPECT_EQ(fit.n_pairs, n_pairs);
    EXPECT_EQ(plain.alpha, 1.0); // left as it was
    for (int k = 0; k < p2vbme::N_GTR_RATES; ++k) {
        EXPECT_EQ(fit.rates[k], whole_gtr.rates[k]) << "rate " << k;
        EXPECT_EQ(plain.rates[k], whole_gtr.rates[k]) << "rate " << k;
    }
    a = aln;
    p2vbme::SubstitutionModel jc;
    EXPECT_EQ(p2vbme::prepare_alignment(a, jc, n_pairs, seed, 0, true, dd).n_pairs, 0);
    for (double r : jc.rates)
        EXPECT_EQ(r, 1.0);
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
    std::vector<int32_t> back_map;
    const p2vbme::Tree back = p2vbme::prune(full, dd, &back_map);
    ASSERT_TRUE(valid_tree(back, 4));
    EXPECT_EQ(clades(back, 4), clades(red, 4));
    // its node map: each node of the pruned tree stands for the node of the full tree with the same kept leaves below
    ASSERT_EQ(back_map.size(), back.c1.size());
    for (int32_t u = 0; u < 7; ++u) {
        std::set<int> kept_below; // the full tree's kept leaves below back_map[u], as pruned leaves
        std::vector<int32_t> st = {back_map[u]};
        while (!st.empty()) {
            const int32_t v = st.back();
            st.pop_back();
            if (v < 6) {
                for (int k = 0; k < 4; ++k)
                    if (dd.kept[k] == v)
                        kept_below.insert(k);
            } else {
                st.insert(st.end(), {full.c1[v], full.c2[v]});
            }
        }
        std::set<int> below_u;
        st = {u};
        while (!st.empty()) {
            const int32_t v = st.back();
            st.pop_back();
            if (v < 4)
                below_u.insert(v);
            else
                st.insert(st.end(), {back.c1[v], back.c2[v]});
        }
        EXPECT_EQ(kept_below, below_u) << "pruned node " << u;
    }

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

    // the node map: each node of the reduced tree stands for the node of the full tree below which are its kept
    // sequences and their copies
    std::vector<int32_t> map;
    const p2vbme::Tree full3 = p2vbme::reattach(shuffled, dd, &map);
    ASSERT_EQ(map.size(), shuffled.c1.size());
    auto below = [](const p2vbme::Tree &t, int32_t n, int32_t u) {
        std::set<int> out;
        std::vector<int32_t> st = {u};
        while (!st.empty()) {
            const int32_t v = st.back();
            st.pop_back();
            if (v < n)
                out.insert(v);
            else
                st.insert(st.end(), {t.c1[v], t.c2[v]});
        }
        return out;
    };
    for (int32_t u = 0; u < 7; ++u) {
        std::set<int> want;
        for (int k : below(shuffled, 4, u))
            for (int x = 0; x < 6; ++x)
                if (x == dd.kept[k] || dd.copy_of[x] == k)
                    want.insert(x);
        EXPECT_EQ(below(full3, 6, map[u]), want) << "reduced node " << u;
    }
    // without copies set aside, the map is the identity
    p2vbme::Deduplication none;
    p2vbme::Tree same = shuffled;
    p2vbme::restore_copies(same, none, &map);
    EXPECT_EQ(map, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}));
}

// A start tree on all names is pruned to the kept sequences, and the copies come back after; without copies set
// aside (or without dedup) both leave the tree as it is.
TEST(Processing, StartTreeAndCopies) {
    const std::string fa = temp_path("q.fa");
    write_file(fa, ">a\nACGT\n>b\nACGT\n>c\nACGT\n>d\nTTTT\n>e\nGGGG\n");
    p2vbme::Alignment aln = p2vbme::read_fasta(fa);
    p2vbme::SubstitutionModel jc;
    p2vbme::Deduplication dd;
    p2vbme::prepare_alignment(aln, jc, p2vbme::DEFAULT_GTR_FIT_PAIRS, p2vbme::DEFAULT_SEED, 0, true, dd);
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
