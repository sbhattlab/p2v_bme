/*
 * bme_scores.cpp
 *
 * Exact BME lengths of given trees, for the FastME comparison: the distance matrix of an alignment computed as
 * p2v-bme's search computes it (the model's pair distance in single precision, as on the GPU), optionally written as
 * a square PHYLIP matrix for FastME, and the balanced minimum evolution length of each tree from the full matrix in
 * double precision.
 * Ported from the paper's scoring programs (bin2phylip_dist.c, exact_bme.py), which share no code with the search.
 *
 *   bme_scores -s ALIGNMENT -m MODEL [--phylip OUT.phy] [TREE.nwk ...]
 *
 * Prints one JSON line per tree: {"tree": path, "bme_length": L}.
 * The length is Pauplin's: L = sum over pairs i < j of 2^(1 - tau_ij) d_ij, tau_ij the number of edges between i and
 * j; on the rooted tree it is D~(c1(r), c2(r)) + 1/2 sum over the other internal nodes u of D~(c1(u), c2(u)), where
 * D~(A, B) is the balanced (2^-depth weighted) average distance between the clades A and B.
 * O(N^2) time; memory: the N x N matrix (float) and one row per clade waiting for its sibling.
 *
 * Reference
 *   Pauplin, Y. (2000). Direct calculation of a tree length using a distance matrix. J. Mol. Evol. 51, 41-47.
 *     doi:10.1007/s002390010065
 */
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distances.hpp"
#include "io.hpp"
#include "newick.hpp"

namespace {

using p2vbme::Alignment;
using p2vbme::Tree;

// Computes the distance matrix as p2v-bme's search does: (aln, model) -> N x N floats, row major.
std::vector<float> distance_matrix(const Alignment &aln, p2vbme::SubstitutionModel model) {
    if (model.model == p2vbme::Model::GTR_GC)
        throw std::runtime_error("GTR+GC needs the fitted rates and alpha; not supported here");
    p2vbme::base_frequencies(aln.packed.data(), aln.n_taxa, aln.n_words, model.base_freqs);
    const p2vbme::DistanceParams<float> params = p2vbme::distance_params<float>(model);
    const int32_t n = aln.n_taxa;
    const size_t row = (size_t)p2vbme::N_PLANES * aln.n_words;
    std::vector<float> D((size_t)n * n, 0.0f);
#pragma omp parallel for schedule(dynamic, 16)
    for (int32_t i = 0; i < n; ++i)
        for (int32_t j = i + 1; j < n; ++j) {
            const p2vbme::PairCounts c = p2vbme::count_pair(&aln.packed[i * row], &aln.packed[j * row], aln.n_words);
            const float d = p2vbme::pair_distance(params, c.n);
            D[(size_t)i * n + j] = d;
            D[(size_t)j * n + i] = d;
        }
    return D;
}

// Writes the matrix as a square PHYLIP file (names, then the row): (names, D, path) -> file.
void write_phylip(const std::vector<std::string> &names, const std::vector<float> &D, const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    const size_t n = names.size();
    std::fprintf(f, "%zu\n", n);
    for (size_t i = 0; i < n; ++i) {
        std::fputs(names[i].c_str(), f);
        // %.9g reads back as the same float
        for (size_t j = 0; j < n; ++j)
            std::fprintf(f, " %.9g", (double)D[i * n + j]);
        std::fputc('\n', f);
    }
    if (std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
}

// Computes the exact BME length of a rooted binary tree: (tree, D) -> L in double precision.
double exact_bme_length(const Tree &tree, const std::vector<float> &D) {
    const int32_t n = tree.n_leaves();
    const int32_t root = 2 * n - 2;
    // Internal nodes in post-order (children first), from the root.
    std::vector<int32_t> order, stack = {root};
    while (!stack.empty()) {
        const int32_t u = stack.back();
        stack.pop_back();
        if (u < n)
            continue;
        order.push_back(u);
        stack.push_back(tree.c1[u]);
        stack.push_back(tree.c2[u]);
    }
    std::vector<std::pair<int32_t, double>> leaf_weights;   // reused: the leaves of a clade and 2^-depth
    std::unordered_map<int32_t, std::vector<double>> delta; // balanced average distance of a clade to every leaf
    std::unordered_map<int32_t, std::vector<std::pair<int32_t, double>>> leaves;
    std::vector<double> leaf_a(n), leaf_b(n);
    // A clade's row of delta: a leaf's row of D (copied into scratch), else its stored row.
    auto row_of = [&](int32_t v, std::vector<double> &scratch) -> const double * {
        if (v >= n)
            return delta[v].data();
        for (int32_t y = 0; y < n; ++y)
            scratch[y] = D[(size_t)v * n + y];
        return scratch.data();
    };
    double total = 0.0;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int32_t u = *it, a = tree.c1[u], b = tree.c2[u];
        const double *ra = row_of(a, leaf_a), *rb = row_of(b, leaf_b);
        // D~(a, b): the leaves of b, weighted by their depth in b, against a's averages
        double term = 0.0;
        if (b < n)
            term = ra[b];
        else
            for (const auto &[y, w] : leaves[b])
                term += w * ra[y];
        total += u == root ? term : 0.5 * term;
        if (u == root)
            break;
        std::vector<double> row(n);
        for (int32_t y = 0; y < n; ++y)
            row[y] = 0.5 * (ra[y] + rb[y]);
        leaf_weights.clear();
        for (int32_t c : {a, b}) {
            if (c < n)
                leaf_weights.emplace_back(c, 0.5);
            else
                for (const auto &[y, w] : leaves[c])
                    leaf_weights.emplace_back(y, 0.5 * w);
        }
        for (int32_t c : {a, b}) {
            delta.erase(c);
            leaves.erase(c);
        }
        delta[u] = std::move(row);
        leaves[u] = leaf_weights;
    }
    return total;
}

// Escapes a path for a JSON string: path -> its JSON form.
std::string json_string(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out + '"';
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string alignment, model_name, phylip;
        std::vector<std::string> trees;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "-s" || arg == "-m" || arg == "--phylip") && i + 1 < argc)
                (arg == "-s" ? alignment : arg == "-m" ? model_name : phylip) = argv[++i];
            else if (!arg.empty() && arg[0] != '-')
                trees.push_back(arg);
            else
                throw std::runtime_error("usage: bme_scores -s ALIGNMENT -m MODEL [--phylip OUT.phy] [TREE.nwk ...]");
        }
        if (alignment.empty() || model_name.empty())
            throw std::runtime_error("usage: bme_scores -s ALIGNMENT -m MODEL [--phylip OUT.phy] [TREE.nwk ...]");
        const Alignment aln = p2vbme::read_alignment(alignment);
        p2vbme::SubstitutionModel model;
        model.model = p2vbme::parse_model(model_name);
        const std::vector<float> D = distance_matrix(aln, model);
        if (!phylip.empty())
            write_phylip(aln.names, D, phylip);
        for (const std::string &path : trees) {
            const Tree tree = p2vbme::read_newick(path, aln.names);
            std::printf("{\"tree\": %s, \"bme_length\": %.17g}\n", json_string(path).c_str(),
                        exact_bme_length(tree, D));
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "bme_scores: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
