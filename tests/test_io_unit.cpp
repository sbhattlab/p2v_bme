// Unit tests for p2vbme I/O: FASTA packing and errors, Newick input, unrooted Newick output.
#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#include "dedup.hpp"
#include "fasta.hpp"
#include "io.hpp"
#include "maple.hpp"

namespace {

std::string temp_path(const std::string &name) { return testing::TempDir() + "p2vbme_" + name; }

void write_file(const std::string &path, const std::string &text) { std::ofstream(path) << text; }

std::string read_file(const std::string &path) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

bool bit(const p2vbme::Alignment &a, int taxon, int plane, int site) {
    return (a.packed[((size_t)taxon * 5 + plane) * a.n_words + site / 64] >> (site % 64)) & 1ULL;
}

// Leaf sets below every internal node except the root: the rooted splits of a tree.
std::set<std::set<int>> clades(const p2vbme::Tree &t, int n) {
    std::vector<std::set<int>> below(2 * n - 1);
    for (int i = 0; i < n; ++i)
        below[i] = {i};
    std::set<std::set<int>> out;
    for (int u = n; u < 2 * n - 1; ++u) {
        below[u] = below[t.c1[u]];
        below[u].insert(below[t.c2[u]].begin(), below[t.c2[u]].end());
        if (u != 2 * n - 2)
            out.insert(below[u]);
    }
    return out;
}

// Unrooted splits, each stored as the side without leaf 0.
std::set<std::set<int>> splits(const p2vbme::Tree &t, int n) {
    std::set<std::set<int>> out;
    for (const auto &c : clades(t, n)) {
        if ((int)c.size() < 2 || (int)c.size() > n - 2)
            continue;
        if (c.count(0)) {
            std::set<int> other;
            for (int i = 0; i < n; ++i)
                if (!c.count(i))
                    other.insert(i);
            out.insert(other);
        } else {
            out.insert(c);
        }
    }
    return out;
}

} // namespace

TEST(Fasta, PacksPlanesAsTheBundles) {
    const std::string p = temp_path("pack.fa");
    write_file(p, ">a desc\nACGTU-\nNnRY\n>b\nacgtac\ngtac\n");
    const auto aln = p2vbme::read_fasta(p);
    ASSERT_EQ(aln.n_taxa, 2);
    ASSERT_EQ(aln.n_sites, 10);
    ASSERT_EQ(aln.n_words, 1);
    EXPECT_EQ(aln.names, (std::vector<std::string>{"a", "b"}));
    // taxon a: A C G T U - N n R Y
    EXPECT_TRUE(bit(aln, 0, 0, 0));
    EXPECT_TRUE(bit(aln, 0, 1, 1));
    EXPECT_TRUE(bit(aln, 0, 2, 2));
    EXPECT_TRUE(bit(aln, 0, 3, 3));
    EXPECT_TRUE(bit(aln, 0, 3, 4)); // U counts as T
    for (int site = 5; site < 10; ++site)
        for (int plane = 0; plane < 5; ++plane)
            EXPECT_FALSE(bit(aln, 0, plane, site)) << site << " " << plane;
    for (int site = 0; site < 5; ++site)
        EXPECT_TRUE(bit(aln, 0, 4, site));
    // taxon b: lowercase is read as uppercase
    EXPECT_TRUE(bit(aln, 1, 0, 0));
    EXPECT_TRUE(bit(aln, 1, 1, 9));
}

TEST(Fasta, RejectsBadInput) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {">a\nACGT\n>b\nACG\n", "must be aligned"},
        {">a\nACGT\n>a\nACGT\n", "duplicate name"},
        {"ACGT\n>a\nACGT\n", "before the first"},
        {"", "no sequences"},
    };
    for (const auto &[text, message] : cases) {
        const std::string p = temp_path("bad.fa");
        write_file(p, text);
        try {
            p2vbme::read_fasta(p);
            ADD_FAILURE() << "no error for: " << text;
        } catch (const std::runtime_error &e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
}

TEST(Newick, RoundTripKeepsTheTopology) {
    const std::vector<std::string> names = {"t0", "t1", "t2", "t3", "t4", "t5"};
    const std::string in = temp_path("in.nwk"), out = temp_path("out.nwk");
    write_file(in, "((t0:1,(t1,t2)x:0.5),((t3,t4),t5));\n");
    const auto t = p2vbme::read_newick(in, names);
    p2vbme::write_newick_unrooted(out, names, t);
    const auto back = p2vbme::read_newick(out, names); // unrooted input is accepted
    EXPECT_EQ(splits(t, 6), splits(back, 6));
    EXPECT_EQ(splits(t, 6).size(), 3u);
}

TEST(Newick, RejectsBadStartTrees) {
    const std::vector<std::string> names = {"a", "b", "c", "d"};
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"(a,b,(c,x));", "is not in the alignment"},
        {"(a,b,c);", "is not in the tree"},
        {"(a,b,c,d);", "must be binary"},
        {"((a,b),(c,d)", "ends before"},
    };
    for (const auto &[text, message] : cases) {
        const std::string p = temp_path("bad.nwk");
        write_file(p, text);
        try {
            p2vbme::read_newick(p, names);
            ADD_FAILURE() << "no error for: " << text;
        } catch (const std::runtime_error &e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
}

TEST(Newick, UnrootedOutputMergesTheRootEdgesAndClampsNegatives) {
    // ((a,b),(c,d)) rooted: leaves 0..3, node 4 = (0,1), node 5 = (2,3), root 6 = (4,5)
    p2vbme::Tree t;
    t.c1 = {-1, -1, -1, -1, 0, 2, 4};
    t.c2 = {-1, -1, -1, -1, 1, 3, 5};
    t.branch_lengths = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, 0.25f, 0.0f};
    const std::string out = temp_path("bl.nwk");
    p2vbme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t);
    // node 4 becomes the trifurcation; node 5 hangs from it by the merged edge 0.5 + 0.25
    EXPECT_EQ(read_file(out), "(a:0.1,b:0,(c:0.3,d:0.4):0.75);\n");
}

TEST(Newick, LengthsAreShortestRoundTripFloats) {
    p2vbme::Tree t;
    t.c1 = {-1, -1, -1, -1, 0, 2, 4};
    t.c2 = {-1, -1, -1, -1, 1, 3, 5};
    t.branch_lengths = {1.0f / 3.0f, -0.2f, 1.5e-7f, 0.4f, 0.5f, 0.25f, 0.0f};
    const std::string out = temp_path("lengths.nwk");
    p2vbme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t);
    EXPECT_EQ(read_file(out), "(a:0.33333334,b:0,(c:1.5e-07,d:0.4):0.75);\n");
    // --no-neg-as-zero: raw lengths
    p2vbme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t, false);
    EXPECT_EQ(read_file(out), "(a:0.33333334,b:-0.2,(c:1.5e-07,d:0.4):0.75);\n");
    // each length reads back to the same float
    EXPECT_EQ(std::stof("0.33333334"), 1.0f / 3.0f);
    EXPECT_EQ(std::stof("1.5e-07"), 1.5e-7f);
}

TEST(Newick, QuotesNamesThatNeedIt) {
    p2vbme::Tree t;
    t.c1 = {-1, -1, -1, 0, 3};
    t.c2 = {-1, -1, -1, 1, 2};
    const std::string out = temp_path("q.nwk");
    p2vbme::write_newick_unrooted(out, {"plain", "with space", "it's"}, t);
    EXPECT_EQ(read_file(out), "(plain,'with space','it''s');\n");
}

// ---- .maple, identical sequences ----------------------------------------------------------------------------

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

// The same alignment as FASTA and as .maple (reference + differences) packs to the same bitplanes.
TEST(Maple, SameAlignmentAsFasta) {
    const std::string fa = temp_path("m.fa"), mp = temp_path("m.maple");
    write_file(fa, ">s1\nACGTACGTAC\n>s2 descr\nACGTnnnTAC\n>s3\nTCGTAC-tAR\n>s4\nACGTACGTAC\n");
    write_file(mp, ">ref\nACGTAC\nGTAC\n"
                   ">s1\n"
                   ">s2 descr\nn 5 3\n"
                   ">s3\nT 1\n- 7\nr 10\nt 8\n\n"
                   ">s4\n");
    const p2vbme::Alignment a = p2vbme::read_fasta(fa), b = p2vbme::read_alignment(mp);
    EXPECT_EQ(b.n_taxa, 4);
    EXPECT_EQ(b.n_sites, 10);
    EXPECT_EQ(b.names, a.names);
    EXPECT_EQ(b.packed, a.packed);
}

TEST(Maple, RejectsBadInput) {
    const std::string p = temp_path("bad.maple");
    write_file(p, "A 3\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // no reference
    write_file(p, ">ref\nACGT\n>s\nA 5\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // outside the reference
    write_file(p, ">ref\nACGT\n>s\nn 3 3\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // runs past the end
    write_file(p, ">ref\nACGT\n>s\nn 2 9223372036854775807\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // a length that would overflow
    write_file(p, ">ref\nACGT\n>s\nA x\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // not a position
    write_file(p, ">ref\nACGT\n>s\n>s\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // duplicate name
    write_file(p, ">ref\nACGT\n");
    EXPECT_THROW(p2vbme::read_maple(p), std::runtime_error); // no sequences
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
