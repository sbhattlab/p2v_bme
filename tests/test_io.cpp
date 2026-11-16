// Unit tests for Newick input and unrooted Newick output.
#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io.hpp"
#include "io_test_helpers.hpp"

namespace {

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
