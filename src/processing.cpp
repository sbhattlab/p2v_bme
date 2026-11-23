/*
 * processing.cpp
 *
 * The alignment before inference (base frequencies, GTR+GC fit, identical sequences) and the trees around it (a given
 * start tree, the copies joined back).
 * Tree traversals are iterative so that trees with millions of leaves do not overflow the stack.
 */
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "processing.hpp"
#include "random.hpp"

namespace p2vbme {

namespace {

// The internal nodes of a rooted tree (root 2n-2) in post-order, found from the root.
// The search reuses internal node ids, so ids alone do not give children before parents.
std::vector<int32_t> internal_postorder(const Tree &tree, int32_t n) {
    std::vector<int32_t> order, stack = {2 * n - 2};
    order.reserve(n - 1);
    while (!stack.empty()) { // parents before children, then reversed
        const int32_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        if (tree.c1[u] >= n)
            stack.push_back(tree.c1[u]);
        if (tree.c2[u] >= n)
            stack.push_back(tree.c2[u]);
    }
    if ((int32_t)order.size() != n - 1)
        throw std::invalid_argument("the tree does not have " + std::to_string(n - 1) +
                                    " internal nodes below its root");
    std::reverse(order.begin(), order.end());
    return order;
}

} // namespace

// ---- The alignment: base frequencies, GTR+GC fit, identical sequences set aside ----

void count_base_frequencies(const Alignment &aln, SubstitutionModel &model) {
    const double sum = model.base_freqs[0] + model.base_freqs[1] + model.base_freqs[2] + model.base_freqs[3];
    if ((model.model == Model::F81 || model.model == Model::F84 || model.model == Model::TN93) && !(sum > 0.0))
        base_frequencies(aln.packed.data(), aln.n_taxa, aln.n_words, model.base_freqs);
}

GtrFit fit_model_parameters(const Alignment &aln, SubstitutionModel &model, int32_t n_pairs, uint64_t seed,
                            int threads) {
    if (model.model != Model::GTR_GC)
        return GtrFit{};
    const GtrFit fit = fit_gtr_gc(aln, n_pairs, derive_seed(seed, SALT_GTR_FIT_PAIRS), threads);
    for (int k = 0; k < N_GTR_RATES; ++k)
        model.rates[k] = fit.rates[k];
    model.alpha = fit.alpha;
    return fit;
}

void deduplicate(Alignment &aln, Deduplication &dd) {
    const size_t row_words = (size_t)N_PLANES * aln.n_words;
    dd.kept.clear();
    dd.copy_of.assign(aln.n_taxa, -1);
    // rows grouped by a hash of their words, then compared word by word
    std::unordered_map<uint64_t, std::vector<int32_t>> by_hash; // hash -> kept positions of the first copies
    std::vector<int32_t> group_size;                            // per kept position: copies of its row seen
    for (int32_t x = 0; x < aln.n_taxa; ++x) {
        const uint64_t *row = &aln.packed[(size_t)x * row_words];
        uint64_t h = 1469598103934665603ULL;
        for (size_t w = 0; w < row_words; ++w)
            h = (h ^ row[w]) * 1099511628211ULL;
        int32_t first = -1;
        for (int32_t k : by_hash[h])
            if (memcmp(row, &aln.packed[(size_t)dd.kept[k] * row_words], row_words * sizeof(uint64_t)) == 0) {
                first = k;
                break;
            }
        if (first >= 0 && group_size[first] >= 2) { // a third or later copy: removed
            dd.copy_of[x] = first;
            ++group_size[first];
            continue;
        }
        if (first >= 0)
            ++group_size[first];
        dd.kept.push_back(x);
        group_size.push_back(first >= 0 ? 0 : 1); // only the first copy of a group counts the copies
        if (first < 0)
            by_hash[h].push_back((int32_t)dd.kept.size() - 1);
    }
    // compact the kept rows in place (no second copy of a large alignment)
    if (dd.kept.size() == (size_t)aln.n_taxa)
        return;
    for (size_t k = 0; k < dd.kept.size(); ++k) {
        const int32_t x = dd.kept[k];
        if ((size_t)x != k) {
            std::copy(aln.packed.begin() + (size_t)x * row_words, aln.packed.begin() + (size_t)(x + 1) * row_words,
                      aln.packed.begin() + k * row_words);
            aln.names[k] = aln.names[x];
        }
    }
    aln.n_taxa = (int32_t)dd.kept.size();
    aln.packed.resize((size_t)aln.n_taxa * row_words);
    aln.names.resize(aln.n_taxa);
}

GtrFit prepare_alignment(Alignment &aln, SubstitutionModel &model, int32_t fit_pairs, uint64_t seed, int threads,
                         bool dedup, Deduplication &dd) {
    // both before identical sequences are set aside (as FastME counts the base frequencies)
    count_base_frequencies(aln, model);
    const GtrFit fit = fit_model_parameters(aln, model, fit_pairs, seed, threads);
    if (dedup)
        deduplicate(aln, dd);
    return fit;
}

// ---- Trees: a given start tree restricted, the copies put back ----

Tree prune(const Tree &tree, const Deduplication &dd) {
    const int32_t N = (int32_t)dd.copy_of.size(), M = (int32_t)dd.kept.size();
    if ((int32_t)tree.c1.size() != 2 * N - 1)
        throw std::invalid_argument("prune: the tree is not on all sequences");
    std::vector<int32_t> new_id(2 * N - 1, -1); // reduced id of a kept leaf or of a kept internal node; -1 = gone
    for (int32_t k = 0; k < M; ++k)
        new_id[dd.kept[k]] = k;
    Tree out;
    out.c1.assign(2 * M - 1, -1);
    out.c2.assign(2 * M - 1, -1);
    int32_t next = M;
    for (int32_t u : internal_postorder(tree, N)) { // children before parents
        const int32_t a = new_id[tree.c1[u]], b = new_id[tree.c2[u]];
        if (a >= 0 && b >= 0) {
            out.c1[next] = a;
            out.c2[next] = b;
            new_id[u] = next++;
        } else {
            new_id[u] = a >= 0 ? a : b; // a node with a removed child is replaced by the other child
        }
    }
    if (next != 2 * M - 1)
        throw std::logic_error("prune: wrong number of nodes");
    return out;
}

void prepare_start_tree(Tree &tree, const Deduplication &dd) {
    if (dd.n_removed() > 0)
        tree = prune(tree, dd);
}

Tree reattach(const Tree &tree, const Deduplication &dd, std::vector<int32_t> *node_map) {
    const int32_t M = (int32_t)dd.kept.size(), N = (int32_t)dd.copy_of.size();
    if ((int32_t)tree.c1.size() != 2 * M - 1)
        throw std::invalid_argument("reattach: the tree is not on the kept sequences");
    const bool with_bl = !tree.branch_lengths.empty();
    if (with_bl && (int32_t)tree.branch_lengths.size() < 2 * M - 2)
        throw std::invalid_argument("reattach: fewer branch lengths than edges");
    std::vector<std::vector<int32_t>> extra(M); // removed copies, by the kept position of their first copy
    for (int32_t x = 0; x < N; ++x)
        if (dd.copy_of[x] >= 0)
            extra[dd.copy_of[x]].push_back(x);
    Tree full;
    full.double_precision_lengths = tree.double_precision_lengths;
    full.c1.assign(2 * N - 1, -1);
    full.c2.assign(2 * N - 1, -1);
    if (with_bl) // the 2N - 2 edges above nodes 0..2N-3, as compute_bme_branch_lengths
        full.branch_lengths.assign(2 * N - 2, 0.0);
    std::vector<int32_t> top(2 * M - 1, -1); // node of the full tree standing for each node of the reduced tree
    int32_t next = N;
    for (int32_t k = 0; k < M; ++k) {
        // the kept sequence, then each removed copy joined to it by zero-length branches
        int32_t node = dd.kept[k];
        for (int32_t x : extra[k]) {
            full.c1[next] = node;
            full.c2[next] = x;
            node = next++;
        }
        top[k] = node;
        if (with_bl)
            full.branch_lengths[node] = tree.branch_lengths[k];
    }
    for (int32_t u : internal_postorder(tree, M)) { // children before parents
        full.c1[next] = top[tree.c1[u]];
        full.c2[next] = top[tree.c2[u]];
        top[u] = next;
        if (with_bl && u != 2 * M - 2) // the root has no edge above it
            full.branch_lengths[next] = tree.branch_lengths[u];
        ++next;
    }
    if (node_map)
        *node_map = top;
    return full;
}

void restore_copies(Tree &tree, const Deduplication &dd, std::vector<int32_t> *node_map) {
    if (dd.n_removed() > 0) {
        tree = reattach(tree, dd, node_map);
    } else if (node_map) {
        node_map->resize(tree.c1.size());
        std::iota(node_map->begin(), node_map->end(), 0);
    }
}

} // namespace p2vbme
