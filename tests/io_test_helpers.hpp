// Helpers shared by the input/output unit tests (test_fasta, test_maple, test_dedup, test_newick).
#ifndef P2VBME_TESTS_IO_TEST_HELPERS_HPP
#define P2VBME_TESTS_IO_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "inference.hpp"

namespace {

std::string temp_path(const std::string &name) { return testing::TempDir() + "p2vbme_" + name; }

void write_file(const std::string &path, const std::string &text) { std::ofstream(path) << text; }

std::string read_file(const std::string &path) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
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

} // namespace

#endif // P2VBME_TESTS_IO_TEST_HELPERS_HPP
