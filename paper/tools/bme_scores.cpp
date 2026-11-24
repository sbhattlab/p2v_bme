/*
 * bme_scores.cpp
 *
 * BME lengths of given trees, scored independently of the search.
 * Exact (the FastME comparison, the benchmarks up to 10^4 sequences): the distance matrix of an alignment computed as
 * p2v-bme's search computes it (the model's pair distance in single precision, as on the GPU), optionally written as
 * a square PHYLIP matrix for FastME, and the balanced minimum evolution length of each tree from the full matrix in
 * double precision.
 * With --cap K (the sweep of the cap on representatives, 10^5 and 10^6 sequences): the paper's evaluator, which keeps
 * at most K representatives per clade and needs no matrix; JC69 only, K >= N gives the exact length.
 * Ported from the paper's scoring programs (bin2phylip_dist.c, exact_bme.py, dipper_helper.c --eval), which share no
 * code with the search.
 *
 *   bme_scores -s ALIGNMENT -m MODEL [--phylip OUT.phy] [--cap K] [TREE.nwk ...]
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

#include <algorithm>
#include <cmath>
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
using p2vbme::N_BASES;
using p2vbme::N_PLANES;
using p2vbme::Tree;
using p2vbme::VALID_PLANE;

constexpr float EVALUATOR_P_MAX = 0.74f; // the evaluator's cap on a pair's mismatch proportion

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

// Lists the internal nodes of a rooted binary tree, children before parents: tree -> nodes, the root last.
std::vector<int32_t> internal_postorder(const Tree &tree) {
    const int32_t n = tree.n_leaves();
    std::vector<int32_t> order, stack = {2 * n - 2};
    while (!stack.empty()) {
        const int32_t u = stack.back();
        stack.pop_back();
        if (u < n)
            continue;
        order.push_back(u);
        stack.push_back(tree.c1[u]);
        stack.push_back(tree.c2[u]);
    }
    std::reverse(order.begin(), order.end());
    return order;
}

// Computes the exact BME length of a rooted binary tree: (tree, D) -> L in double precision.
double exact_bme_length(const Tree &tree, const std::vector<float> &D) {
    const int32_t n = tree.n_leaves();
    const int32_t root = 2 * n - 2;
    const std::vector<int32_t> order = internal_postorder(tree);
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
    for (const int32_t u : order) {
        const int32_t a = tree.c1[u], b = tree.c2[u];
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

// Computes the JC69 distance of two packed sequences as the paper's evaluator does: (a, b, words) -> d.
// p = mismatches / valid sites shared, capped at EVALUATOR_P_MAX; d = -3/4 ln(1 - 4/3 p) in single precision; 0 when
// they share no valid site.
float evaluator_jc69(const uint64_t *a, const uint64_t *b, int32_t n_words) {
    int32_t valid = 0, match = 0;
    for (int32_t w = 0; w < n_words; ++w) {
        const uint64_t v = a[VALID_PLANE * n_words + w] & b[VALID_PLANE * n_words + w];
        uint64_t m = 0;
        for (int32_t p = 0; p < N_BASES; ++p)
            m |= a[p * n_words + w] & b[p * n_words + w];
        valid += __builtin_popcountll(v);
        match += __builtin_popcountll(m & v);
    }
    if (valid <= 0)
        return 0.0f;
    float p = (float)(valid - match) / (float)valid;
    if (p >= EVALUATOR_P_MAX)
        p = EVALUATOR_P_MAX;
    const float d = -0.75f * std::log(1.0f - (4.0f / 3.0f) * p);
    return d > 0.0f ? d : 0.0f;
}

// A clade's representatives: leaves and their weights, which add up to 1.
struct Representatives {
    std::vector<int32_t> leaves;
    std::vector<double> weights;
};

// Appends k of a child's representatives at even strides through its list, reweighted to add up to 1/2:
// (child, k) -> out.
void take_half(const Representatives &child, int32_t k, Representatives &out) {
    const int32_t l = (int32_t)child.leaves.size();
    const size_t first = out.leaves.size();
    double sum = 0.0;
    for (int32_t i = 0; i < k; ++i) {
        const int32_t idx = (int32_t)(((int64_t)i * l) / k);
        out.leaves.push_back(child.leaves[idx]);
        out.weights.push_back(child.weights[idx]);
        sum += child.weights[idx];
    }
    const double scale = sum > 0.0 ? 0.5 / sum : 0.0;
    for (size_t i = first; i < out.weights.size(); ++i)
        out.weights[i] *= scale;
}

// Computes the BME length with at most cap representatives per clade, as the paper's evaluator (dipper_helper.c
// --eval) does: (tree, alignment, cap) -> L.
// Bottom-up, a clade keeps both children's representatives with their weights halved while they number at most cap;
// above, each child gives up to cap / 2 (one with fewer leaves its spare share to the other), chosen by take_half.
// L = D~(c1(r), c2(r)) + 1/2 sum over the other internal nodes u of D~(c1(u), c2(u)), D~ over the representatives.
// cap >= N gives the exact length.
double capped_bme_length(const Tree &tree, const Alignment &aln, int32_t cap) {
    const int32_t n = tree.n_leaves();
    const int32_t root = 2 * n - 2;
    const std::vector<int32_t> order = internal_postorder(tree);
    std::vector<Representatives> reps(2 * n - 1);
    for (int32_t x = 0; x < n; ++x)
        reps[x] = {{x}, {1.0}};
    for (const int32_t u : order) {
        if (u == root)
            break;
        const Representatives &a = reps[tree.c1[u]], &b = reps[tree.c2[u]];
        const int32_t la = (int32_t)a.leaves.size(), lb = (int32_t)b.leaves.size();
        Representatives &out = reps[u];
        if (la + lb <= cap) {
            for (const Representatives *c : {&a, &b})
                for (size_t i = 0; i < c->leaves.size(); ++i) {
                    out.leaves.push_back(c->leaves[i]);
                    out.weights.push_back(0.5 * c->weights[i]);
                }
            continue;
        }
        const int32_t half = cap / 2;
        int32_t ka = std::min(la, half), kb = std::min(lb, half);
        if (ka < half && lb > kb)
            kb = std::min(lb, cap - ka);
        else if (kb < half && la > ka)
            ka = std::min(la, cap - kb);
        take_half(a, ka, out);
        take_half(b, kb, out);
    }
    // D~ of each internal node's children, in parallel; summed in a fixed order
    const int32_t n_words = aln.n_words;
    const size_t row = (size_t)N_PLANES * n_words;
    std::vector<double> terms(order.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t k = 0; k < order.size(); ++k) {
        const int32_t u = order[k];
        const Representatives &a = reps[tree.c1[u]], &b = reps[tree.c2[u]];
        double d_ab = 0.0;
        for (size_t i = 0; i < a.leaves.size(); ++i) {
            const uint64_t *pa = &aln.packed[a.leaves[i] * row];
            double acc = 0.0;
            for (size_t j = 0; j < b.leaves.size(); ++j)
                acc += b.weights[j] * (double)evaluator_jc69(pa, &aln.packed[b.leaves[j] * row], n_words);
            d_ab += a.weights[i] * acc;
        }
        terms[k] = (u == root ? 1.0 : 0.5) * d_ab;
    }
    double total = 0.0;
    for (const double t : terms)
        total += t;
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
        const std::string usage = "usage: bme_scores -s ALIGNMENT -m MODEL [--phylip OUT.phy] [--cap K] [TREE.nwk ...]";
        std::string alignment, model_name, phylip, cap_arg;
        std::vector<std::string> trees;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "-s" || arg == "-m" || arg == "--phylip" || arg == "--cap") && i + 1 < argc)
                (arg == "-s" ? alignment : arg == "-m" ? model_name : arg == "--phylip" ? phylip : cap_arg) = argv[++i];
            else if (!arg.empty() && arg[0] != '-')
                trees.push_back(arg);
            else
                throw std::runtime_error(usage);
        }
        if (alignment.empty() || model_name.empty())
            throw std::runtime_error(usage);
        const int32_t cap = cap_arg.empty() ? 0 : std::stoi(cap_arg);
        if (!cap_arg.empty() && cap < 2)
            throw std::runtime_error("--cap needs K >= 2, not " + cap_arg);
        const Alignment aln = p2vbme::read_alignment(alignment);
        p2vbme::SubstitutionModel model;
        model.model = p2vbme::parse_model(model_name);
        if (cap > 0 && model.model != p2vbme::Model::JC69)
            throw std::runtime_error("--cap scores as the paper's evaluator, which has JC69 only");
        std::vector<float> D;
        if (cap == 0 || !phylip.empty())
            D = distance_matrix(aln, model);
        if (!phylip.empty())
            write_phylip(aln.names, D, phylip);
        for (const std::string &path : trees) {
            const Tree tree = p2vbme::read_newick(path, aln.names);
            const double length = cap > 0 ? capped_bme_length(tree, aln, cap) : exact_bme_length(tree, D);
            std::printf("{\"tree\": %s, \"bme_length\": %.17g}\n", json_string(path).c_str(), length);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "bme_scores: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
