/*
 * test_bootstrap.cpp
 *
 * The standard bootstrap (bootstrap.hpp): replicate seeds, the column draw, resampled alignments bit by bit, a small
 * bootstrap on the CPU device, and the support labels of a tree with copies joined back.
 */
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "bootstrap.hpp"
#include "infer_test_helpers.hpp"
#include "simulations.hpp"
#include "support.hpp"

using namespace p2vbme;

namespace {

// Bit `site` of plane `plane` of sequence x.
int get_bit(const Alignment &aln, int32_t x, int32_t plane, int32_t site) {
    const uint64_t w = aln.packed[((size_t)x * N_PLANES + plane) * aln.n_words + site / WORD_BITS];
    return (int)((w >> (site % WORD_BITS)) & 1ULL);
}

// A small simulated alignment: 40 sequences x 700 sites (11 words, the last one partly used) under JC69.
Alignment small_alignment() { return simulate_alignment(random_tree(40, 0.02, 0.2, 3), 700, SimulationModel(), 4); }

} // namespace

/* Replicate seeds differ from each other and from the run seed, and follow the run seed. */
TEST(Bootstrap, ReplicateSeeds) {
    std::set<uint64_t> seen;
    for (int32_t r = 1; r <= 1000; ++r)
        seen.insert(replicate_seed(1, r));
    EXPECT_EQ(seen.size(), 1000u);
    EXPECT_FALSE(seen.count(1));
    EXPECT_NE(replicate_seed(1, 1), replicate_seed(2, 1));
}

/* Columns are drawn with replacement in 0..L-1, the same for the same seed. */
TEST(Bootstrap, DrawColumns) {
    const std::vector<int32_t> c = draw_columns(1000, 7);
    ASSERT_EQ(c.size(), 1000u);
    std::set<int32_t> distinct;
    for (int32_t x : c) {
        ASSERT_TRUE(x >= 0 && x < 1000) << x;
        distinct.insert(x);
    }
    EXPECT_LT(distinct.size(), 1000u); // with replacement: about 63% distinct
    EXPECT_GT(distinct.size(), 550u);
    EXPECT_EQ(draw_columns(1000, 7), c);
    EXPECT_NE(draw_columns(1000, 8), c);
}

/* Each site of the replicate is its column of the source, in every plane, and the unused bits stay 0. */
TEST(Bootstrap, ResampleColumns) {
    const Alignment aln = small_alignment();
    std::vector<int32_t> identity(aln.n_sites);
    for (int32_t j = 0; j < aln.n_sites; ++j)
        identity[j] = j;
    const Alignment same = resample_columns(aln, identity);
    EXPECT_EQ(same.packed, aln.packed);
    EXPECT_EQ(same.names, aln.names);

    const std::vector<int32_t> cols = draw_columns(aln.n_sites, 11);
    const Alignment rep = resample_columns(aln, cols);
    ASSERT_EQ(rep.n_sites, aln.n_sites);
    ASSERT_EQ(rep.n_words, aln.n_words);
    for (int32_t x = 0; x < aln.n_taxa; ++x)
        for (int32_t p = 0; p < N_PLANES; ++p) {
            for (int32_t j = 0; j < aln.n_sites; ++j)
                ASSERT_EQ(get_bit(rep, x, p, j), get_bit(aln, x, p, cols[j])) << x << " " << p << " " << j;
            for (int32_t j = aln.n_sites; j < aln.n_words * WORD_BITS; ++j)
                ASSERT_EQ(get_bit(rep, x, p, j), 0) << "padding bit " << j;
        }
    EXPECT_THROW(resample_columns(aln, {0, aln.n_sites}), std::invalid_argument);
}

/* A bootstrap on the CPU: supports for every branch, within [0, 100], the same for the same seed; replicates do not
 * need the main search's given start tree. */
TEST(Bootstrap, SupportsOnTheCpu) {
    const Alignment aln = small_alignment();
    Settings opt;
    opt.accelerator = Accelerator::Cpu;
    opt.threads = 2;
    const Result main_run = run_infer(aln, opt);
    const int32_t K = aln.n_taxa;
    Devices devices = devices_for(opt);
    const BootstrapResult boot = bootstrap(aln, opt, devices, main_run.tree, 4);
    EXPECT_EQ(boot.n_replicates, 4);
    ASSERT_EQ(boot.tbe.size(), (size_t)(2 * K - 1));
    int32_t n_supported = 0;
    for (int32_t u = 0; u < 2 * K - 1; ++u) {
        if (boot.tbe[u] == NO_SUPPORT)
            continue;
        ++n_supported;
        EXPECT_TRUE(boot.tbe[u] >= 0.0 && boot.tbe[u] <= 100.0) << u;
        EXPECT_TRUE(boot.exact[u] >= 0.0 && boot.exact[u] <= 100.0) << u;
        EXPECT_GE(boot.tbe[u], boot.exact[u]) << u; // TBE also credits near matches
    }
    EXPECT_GE(n_supported, K - 3);  // every internal branch (the root's counted twice when both children are internal)
    EXPECT_GT(boot.mean_tbe, 50.0); // short branches, 700 sites: most splits are recovered
    EXPECT_GT(boot.time, 0.0);

    const BootstrapResult again = bootstrap(aln, opt, devices, main_run.tree, 4);
    EXPECT_EQ(again.tbe, boot.tbe);
    EXPECT_EQ(again.exact, boot.exact);

    Settings given = opt; // the main search started from a given tree: replicates use sample-and-place
    given.tree_init = TreeInit::Given;
    EXPECT_NO_THROW(bootstrap(aln, given, devices, main_run.tree, 1));
    EXPECT_THROW(bootstrap(aln, opt, devices, main_run.tree, 0), std::invalid_argument);
}

/* Replicates run in ranges (1-2, then 3-3) give the bootstrap's trees, in order, and its supports. */
TEST(Bootstrap, ReplicateRangesGiveTheSameTreesAndSupports) {
    const Alignment aln = small_alignment();
    Settings opt;
    opt.accelerator = Accelerator::Cpu;
    opt.threads = 2;
    const Result main_run = run_infer(aln, opt);
    Devices devices = devices_for(opt);
    std::vector<int32_t> boot_order;
    std::vector<Tree> boot_trees, range_trees;
    const BootstrapResult boot =
        bootstrap(aln, opt, devices, main_run.tree, 3, nullptr, [&](int32_t r, const Result &res) {
            boot_order.push_back(r);
            boot_trees.push_back(res.tree);
        });
    EXPECT_EQ(boot_order, (std::vector<int32_t>{1, 2, 3}));
    SupportCounter counter(main_run.tree);
    const ReplicateCallback keep = [&](int32_t, const Result &res) {
        EXPECT_TRUE(res.tree.branch_lengths.empty());
        counter.add(res.tree);
        range_trees.push_back(res.tree);
    };
    bootstrap_replicates(aln, opt, devices, 1, 2, keep);
    bootstrap_replicates(aln, opt, devices, 3, 3, keep);
    ASSERT_EQ(range_trees.size(), boot_trees.size());
    for (size_t i = 0; i < boot_trees.size(); ++i) {
        EXPECT_EQ(range_trees[i].c1, boot_trees[i].c1);
        EXPECT_EQ(range_trees[i].c2, boot_trees[i].c2);
    }
    EXPECT_EQ(counter.tbe(), boot.tbe);
    EXPECT_EQ(counter.exact(), boot.exact);
    EXPECT_THROW(bootstrap_replicate(aln, opt, devices, 0), std::invalid_argument);
    EXPECT_THROW(bootstrap_replicates(aln, opt, devices, 0, 1, keep), std::invalid_argument);
    EXPECT_THROW(bootstrap_replicates(aln, opt, devices, 3, 2, keep), std::invalid_argument);
}

/* Labels of the full tree: the searched tree's supports through the node map, "100/100" on the nodes joining copies,
 * none on leaves and trivial branches, and the root's two children sharing one label. */
TEST(Bootstrap, SupportLabels) {
    // searched tree on 4 kept sequences: (0,(1,(2,3))): node 4 = (2,3), node 5 = (1,4), root 6 = (0,5)
    Tree searched;
    searched.c1 = {-1, -1, -1, -1, 2, 1, 0};
    searched.c2 = {-1, -1, -1, -1, 3, 4, 5};
    Deduplication dd; // sequence 4 is a copy of kept sequence 0
    dd.kept = {0, 1, 2, 3};
    dd.copy_of = {-1, -1, -1, -1, 0};
    std::vector<int32_t> map;
    const Tree full = reattach(searched, dd, &map); // ((0,4),(1,(2,3))): node 5 = (0,4), 6 = (2,3), 7 = (1,6), root 8
    BootstrapResult boot; // node 4's split {2,3} has a support; node 5's {1,2,3} is trivial on 4 leaves
    boot.tbe = {NO_SUPPORT, NO_SUPPORT, NO_SUPPORT, NO_SUPPORT, 97.333, NO_SUPPORT, NO_SUPPORT};
    boot.exact = {NO_SUPPORT, NO_SUPPORT, NO_SUPPORT, NO_SUPPORT, 85.0, NO_SUPPORT, NO_SUPPORT};
    const std::vector<std::string> labels = support_labels(boot, map, full);
    ASSERT_EQ(labels.size(), 9u);
    for (int32_t leaf = 0; leaf < 5; ++leaf)
        EXPECT_EQ(labels[leaf], "") << leaf;
    EXPECT_EQ(labels[map[4]], "97.33/85");
    EXPECT_EQ(labels[map[0]], "100/100"); // the node joining sequence 0 and its copy
    // the root's other child: on the full tree its split {1,2,3} | {0,4} is the copies' branch, so the same label
    EXPECT_EQ(labels[map[5]], "100/100");
    EXPECT_EQ(labels[8], ""); // the root
    EXPECT_THROW(support_labels(boot, {0, 1}, full), std::invalid_argument);
}
