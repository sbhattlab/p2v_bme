// Helpers shared by the tests that check BME lengths on simulated alignments (test_inference, test_branch_lengths,
// test_device_cpu): the simulation model, its distance models, the distance matrix, Pauplin's formula and the CPU
// settings.
#ifndef P2VBME_TESTS_BME_TEST_HELPERS_HPP
#define P2VBME_TESTS_BME_TEST_HELPERS_HPP

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <queue>
#include <vector>

#include "accelerator.hpp"
#include "core.hpp"
#include "distances.hpp"
#include "eigen_dist.hpp"
#include "inference.hpp"
#include "simulations.hpp"

namespace {

// The model the alignments are simulated under: GTR rates, unequal base frequencies, gamma alpha 0.5.
inline p2vbme::SimulationModel truth() {
    p2vbme::SimulationModel m;
    const double r[p2vbme::N_GTR_RATES] = {0.5, 2.0, 0.5, 0.5, 2.0, 0.5}, f[p2vbme::N_BASES] = {0.3, 0.2, 0.2, 0.3};
    for (int k = 0; k < p2vbme::N_GTR_RATES; ++k)
        m.rates[k] = r[k];
    for (int i = 0; i < p2vbme::N_BASES; ++i)
        m.freqs[i] = f[i];
    m.alpha = 0.5;
    return m;
}

// The distance model `model`, with truth()'s GTR rates and alpha (for GTR+GC).
inline p2vbme::SubstitutionModel distance_model(p2vbme::Model model) {
    p2vbme::SubstitutionModel dm;
    dm.model = model;
    const p2vbme::SimulationModel t = truth();
    for (int k = 0; k < p2vbme::N_GTR_RATES; ++k)
        dm.rates[k] = t.rates[k];
    dm.alpha = t.alpha;
    return dm;
}

// The N x N distance matrix of an alignment under a distance model, in double precision.
inline std::vector<double> distance_matrix(const p2vbme::Alignment &aln, const p2vbme::SubstitutionModel &dm) {
    const int32_t N = aln.n_taxa, W = aln.n_words;
    std::vector<double> D((size_t)N * N, 0.0);
    for (int32_t i = 0; i < N; ++i)
        for (int32_t j = i + 1; j < N; ++j) {
            const p2vbme::PairCounts c = p2vbme::count_pair(&aln.packed[(size_t)i * p2vbme::N_PLANES * W],
                                                            &aln.packed[(size_t)j * p2vbme::N_PLANES * W], W);
            D[(size_t)i * N + j] = D[(size_t)j * N + i] =
                p2vbme::pair_distance<double>(p2vbme::distance_params<double>(dm), c.n);
        }
    return D;
}

// Pauplin's BME length of an unrooted tree given as adjacency lists on leaves 0..N-1 and internal nodes.
inline double pauplin(const std::vector<std::vector<int32_t>> &adj, int32_t N, const std::vector<double> &D) {
    double total = 0.0;
    std::vector<int32_t> dist(adj.size());
    for (int32_t i = 0; i < N; ++i) {
        std::fill(dist.begin(), dist.end(), -1);
        std::queue<int32_t> q;
        dist[i] = 0;
        q.push(i);
        while (!q.empty()) {
            const int32_t u = q.front();
            q.pop();
            for (int32_t v : adj[u])
                if (dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
        }
        for (int32_t j = i + 1; j < N; ++j)
            total += ldexp(1.0, 1 - dist[j]) * D[(size_t)i * N + j];
    }
    return total;
}

// The unrooted tree of a rooted Tree (the root's two edges joined).
inline std::vector<std::vector<int32_t>> unrooted(const p2vbme::Tree &t, int32_t N) {
    const int32_t root = 2 * N - 2;
    std::vector<std::vector<int32_t>> adj(2 * N - 1);
    for (int32_t u = N; u < root; ++u)
        for (int32_t c : {t.c1[u], t.c2[u]}) {
            adj[u].push_back(c);
            adj[c].push_back(u);
        }
    adj[t.c1[root]].push_back(t.c2[root]);
    adj[t.c2[root]].push_back(t.c1[root]);
    return adj;
}

// The settings of a CPU run under a distance model: random start, `threads` CPU workers.
inline p2vbme::Settings cpu_settings(p2vbme::Model model, p2vbme::Precision precision = p2vbme::Precision::Single,
                                     int32_t threads = 2) {
    p2vbme::Settings opt;
    opt.tree_init = p2vbme::TreeInit::Random;
    opt.accelerator = p2vbme::Accelerator::Cpu;
    opt.threads = threads;
    opt.precision = precision;
    opt.substitution_model = distance_model(model);
    return opt;
}

} // namespace

#endif // P2VBME_TESTS_BME_TEST_HELPERS_HPP
