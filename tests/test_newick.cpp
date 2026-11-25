// Unit tests for Newick input and unrooted Newick output.
#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io_test_helpers.hpp"
#include "newick.hpp"

namespace {

// Unrooted splits, each stored as the side without leaf 0.
std::set<std::set<int>> splits(const vecme::Tree &t, int n) {
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

// A tree read with its own names and lengths is written back as it was.
TEST(Newick, ReadWithNamesKeepsTheLengths) {
    const std::string in = temp_path("named.nwk"), out = temp_path("named_out.nwk");
    const std::string text = "((b:0.1,'x y':0.25):0.5,a:1e-05,(d:0,c:3):0.125);\n";
    write_file(in, text);
    std::vector<std::string> names;
    const vecme::Tree t = vecme::read_newick_with_names(in, names);
    EXPECT_EQ(names, (std::vector<std::string>{"b", "x y", "a", "d", "c"}));
    EXPECT_TRUE(t.double_precision_lengths);
    vecme::write_newick_unrooted(out, names, t);
    EXPECT_EQ(read_file(out), text);
    write_file(in, "((a,b),(c,d));");
    EXPECT_TRUE(vecme::read_newick_with_names(in, names).branch_lengths.empty());      // no lengths: none kept
    EXPECT_EQ(vecme::read_newick(in, {"d", "c", "b", "a"}).branch_lengths.size(), 0u); // a start tree's are ignored
}

// Several trees in one file, one or more per line or over several lines: the first sets the leaves.
TEST(Newick, ReadsSeveralTrees) {
    const std::string p = temp_path("trees.nwk");
    write_file(p, "((a,b),(c,d));((a,c),(b,d));\n(a,b,\n(c,d));\n");
    std::vector<std::string> names;
    std::vector<std::set<std::set<int>>> got;
    EXPECT_EQ(vecme::read_newick_trees(p, names, [&](const vecme::Tree &t) { got.push_back(splits(t, 4)); }), 3);
    EXPECT_EQ(names, (std::vector<std::string>{"a", "b", "c", "d"}));
    EXPECT_EQ(got, (std::vector<std::set<std::set<int>>>{{{2, 3}}, {{1, 3}}, {{2, 3}}}));
    const std::vector<std::pair<std::string, std::string>> bad = {
        {"((a,b),(c,d));((a,b),(c,e));", "(tree 2): leaf 'e' is not in the alignment"},
        {"((a,b),(c,d));((a,b),c);", "(tree 2): alignment sequence 'd' is not in the tree"},
        {"((a,b),(c,d));((a,b),(c,d))", "Newick ends before ';'"},
    };
    for (const auto &[text, message] : bad) {
        write_file(p, text);
        try {
            vecme::read_newick_trees(p, names, [](const vecme::Tree &) {});
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
    const auto t = vecme::read_newick(in, names);
    vecme::write_newick_unrooted(out, names, t);
    const auto back = vecme::read_newick(out, names); // unrooted input is accepted
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
            vecme::read_newick(p, names);
            ADD_FAILURE() << "no error for: " << text;
        } catch (const std::runtime_error &e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
}

TEST(Newick, UnrootedOutputMergesTheRootEdgesAndClampsNegatives) {
    // ((a,b),(c,d)) rooted: leaves 0..3, node 4 = (0,1), node 5 = (2,3), root 6 = (4,5)
    vecme::Tree t;
    t.c1 = {-1, -1, -1, -1, 0, 2, 4};
    t.c2 = {-1, -1, -1, -1, 1, 3, 5};
    t.branch_lengths = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, 0.25f, 0.0f};
    const std::string out = temp_path("bl.nwk");
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t);
    // node 4 becomes the trifurcation; node 5 hangs from it by the merged edge 0.5 + 0.25
    EXPECT_EQ(read_file(out), "(a:0.1,b:0,(c:0.3,d:0.4):0.75);\n");
}

TEST(Newick, LabelsAfterInternalNodes) {
    // (((a,b),c),(d,e)): leaves 0..4, node 5 = (0,1), node 6 = (5,2), node 7 = (3,4), root 8 = (6,7)
    vecme::Tree t;
    t.c1 = {-1, -1, -1, -1, -1, 0, 5, 3, 6};
    t.c2 = {-1, -1, -1, -1, -1, 1, 2, 4, 7};
    t.branch_lengths = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.25f, 0.75f, 0.0f};
    const std::string out = temp_path("labels.nwk");
    // node 6 becomes the trifurcation (no label of its own); node 7 carries the merged root edge and its label
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d", "e"}, t, true,
                                 {"", "", "", "", "", "97.33/85", "x", "100/100", "root"});
    EXPECT_EQ(read_file(out), "((a:0.1,b:0.2)97.33/85:0.6,c:0.3,(d:0.4,e:0.5)100/100:1);\n");
    // no labels: as before
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d", "e"}, t);
    EXPECT_EQ(read_file(out), "((a:0.1,b:0.2):0.6,c:0.3,(d:0.4,e:0.5):1);\n");
    // one label per node, or none
    EXPECT_THROW(vecme::write_newick_unrooted(out, {"a", "b", "c", "d", "e"}, t, true, {"1"}), std::runtime_error);
}

TEST(Newick, LengthsAreShortestRoundTripFloats) {
    vecme::Tree t;
    t.c1 = {-1, -1, -1, -1, 0, 2, 4};
    t.c2 = {-1, -1, -1, -1, 1, 3, 5};
    t.branch_lengths = {1.0f / 3.0f, -0.2f, 1.5e-7f, 0.4f, 0.5f, 0.25f, 0.0f};
    const std::string out = temp_path("lengths.nwk");
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t);
    EXPECT_EQ(read_file(out), "(a:0.33333334,b:0,(c:1.5e-07,d:0.4):0.75);\n");
    // --no-neg-as-zero: raw lengths
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t, false);
    EXPECT_EQ(read_file(out), "(a:0.33333334,b:-0.2,(c:1.5e-07,d:0.4):0.75);\n");
    // each length reads back to the same float
    EXPECT_EQ(std::stof("0.33333334"), 1.0f / 3.0f);
    EXPECT_EQ(std::stof("1.5e-07"), 1.5e-7f);
}

// --precision double: the lengths keep double digits (shortest round trip of the double values).
TEST(Newick, DoublePrecisionLengthsKeepDoubleDigits) {
    vecme::Tree t;
    t.c1 = {-1, -1, -1, -1, 0, 2, 4};
    t.c2 = {-1, -1, -1, -1, 1, 3, 5};
    t.branch_lengths = {1.0 / 3.0, 0.1, 0.2, 0.4, 0.5, 0.25, 0.0};
    t.double_precision_lengths = true;
    const std::string out = temp_path("double_lengths.nwk");
    vecme::write_newick_unrooted(out, {"a", "b", "c", "d"}, t);
    EXPECT_EQ(read_file(out), "(a:0.3333333333333333,b:0.1,(c:0.2,d:0.4):0.75);\n");
    EXPECT_EQ(std::stod("0.3333333333333333"), 1.0 / 3.0);
}

TEST(Newick, QuotesNamesThatNeedIt) {
    vecme::Tree t;
    t.c1 = {-1, -1, -1, 0, 3};
    t.c2 = {-1, -1, -1, 1, 2};
    const std::string out = temp_path("q.nwk");
    vecme::write_newick_unrooted(out, {"plain", "with space", "it's"}, t);
    EXPECT_EQ(read_file(out), "(plain,'with space','it''s');\n");
}
