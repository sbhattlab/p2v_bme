/*
 * bootstrap.cpp
 *
 * The standard bootstrap (bootstrap.hpp): replicate seeds and columns, the replicate loop, and the support labels.
 */
#include "bootstrap.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "random.hpp"
#include "support.hpp"
#include "timing.hpp"

namespace p2vbme {

uint64_t replicate_seed(uint64_t seed, int32_t replicate) {
    return derive_seed(derive_seed(seed, SALT_BOOTSTRAP), (uint64_t)replicate);
}

std::vector<int32_t> draw_columns(int32_t n_sites, uint64_t seed) {
    std::vector<int32_t> columns(n_sites > 0 ? n_sites : 0);
    uint64_t state = seed;
    for (int32_t &c : columns)
        c = (int32_t)uniform_below(state, (uint64_t)n_sites);
    return columns;
}

Alignment resample_columns(const Alignment &aln, const std::vector<int32_t> &columns) {
    Alignment out;
    out.n_taxa = aln.n_taxa;
    out.n_sites = (int32_t)columns.size();
    out.n_words = (out.n_sites + WORD_BITS - 1) / WORD_BITS;
    out.names = aln.names;
    const int32_t L = out.n_sites;
    std::vector<int32_t> word(L), bit(L); // where each new site is in a source row
    for (int32_t j = 0; j < L; ++j) {
        if (columns[j] < 0 || columns[j] >= aln.n_sites)
            throw std::invalid_argument("resample_columns: column " + std::to_string(columns[j]) + " is not in 0.." +
                                        std::to_string(aln.n_sites - 1));
        word[j] = columns[j] / WORD_BITS;
        bit[j] = columns[j] % WORD_BITS;
    }
    const size_t in_row = (size_t)N_PLANES * aln.n_words, out_row = (size_t)N_PLANES * out.n_words;
    out.packed.assign((size_t)aln.n_taxa * out_row, 0);
#pragma omp parallel for schedule(static)
    for (int32_t x = 0; x < aln.n_taxa; ++x)
        for (int32_t p = 0; p < N_PLANES; ++p) {
            const uint64_t *src = &aln.packed[(size_t)x * in_row + (size_t)p * aln.n_words];
            uint64_t *dst = &out.packed[(size_t)x * out_row + (size_t)p * out.n_words];
            for (int32_t j = 0; j < L; ++j)
                dst[j / WORD_BITS] |= ((src[word[j]] >> bit[j]) & 1ULL) << (j % WORD_BITS);
        }
    return out;
}

Result bootstrap_replicate(const Alignment &aln, const Settings &settings, Devices &devices, int32_t replicate,
                           const LogCallback &log) {
    if (replicate < 1)
        throw std::invalid_argument("bootstrap: replicates are numbered from 1");
    Settings rep = settings;
    rep.branch_lengths = false;
    if (rep.tree_init == TreeInit::Given)
        rep.tree_init = TreeInit::Sample;
    rep.seed = replicate_seed(settings.seed, replicate);
    const Alignment columns =
        resample_columns(aln, draw_columns(aln.n_sites, derive_seed(rep.seed, SALT_BOOTSTRAP_COLUMNS)));
    return infer(columns, rep, devices, nullptr, log);
}

void bootstrap_replicates(const Alignment &aln, const Settings &settings, Devices &devices, int32_t first, int32_t last,
                          const ReplicateCallback &on_replicate, const LogCallback &log) {
    if (first < 1 || last < first)
        throw std::invalid_argument("bootstrap: replicates are numbered from 1, the first no later than the last");
    for (int32_t r = first; r <= last; ++r) {
        const double t_rep = wall_sec();
        const Result res = bootstrap_replicate(aln, settings, devices, r);
        if (on_replicate)
            on_replicate(r, res);
        if (log) {
            char line[128];
            std::snprintf(line, sizeof(line), "bootstrap replicate %d/%d: %d rounds in %.2f s", r, last, res.rounds_run,
                          wall_sec() - t_rep);
            log(line);
        }
    }
}

BootstrapResult bootstrap(const Alignment &aln, const Settings &settings, Devices &devices, const Tree &reference,
                          int32_t n_replicates, const LogCallback &log, const ReplicateCallback &on_replicate) {
    if (n_replicates < 1)
        throw std::invalid_argument("bootstrap: the number of replicates must be >= 1");
    const double t0 = wall_sec();
    SupportCounter counter(reference);
    bootstrap_replicates(
        aln, settings, devices, 1, n_replicates,
        [&](int32_t r, const Result &res) {
            counter.add(res.tree);
            if (on_replicate)
                on_replicate(r, res);
        },
        log);
    BootstrapResult out;
    out.n_replicates = n_replicates;
    out.tbe = counter.tbe();
    out.exact = counter.exact();
    out.mean_tbe = counter.mean_tbe();
    out.mean_exact = counter.mean_exact();
    out.time = wall_sec() - t0;
    return out;
}

std::vector<std::string> support_labels(const BootstrapResult &boot, const std::vector<int32_t> &node_map,
                                        const Tree &full_tree) {
    const int32_t n_nodes = (int32_t)full_tree.c1.size(), N = full_tree.n_leaves();
    if (boot.tbe.size() != node_map.size() || boot.exact.size() != node_map.size())
        throw std::invalid_argument("support_labels: one support and one map entry per node of the searched tree");
    std::vector<std::string> labels(n_nodes);
    for (size_t u = 0; u < node_map.size(); ++u) {
        const int32_t v = node_map[u];
        if (v < 0 || v >= n_nodes)
            throw std::invalid_argument("support_labels: the node map leaves the full tree");
        if (boot.tbe[u] != NO_SUPPORT)
            labels[v] = format_support(boot.tbe[u]) + "/" + format_support(boot.exact[u]);
    }
    const int32_t root = n_nodes - 1, a = full_tree.c1[root], b = full_tree.c2[root];
    if (labels[a].empty() && a >= N)
        labels[a] = labels[b];
    if (labels[b].empty() && b >= N)
        labels[b] = labels[a];
    return labels;
}

} // namespace p2vbme
