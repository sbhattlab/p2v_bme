// Helpers shared by the tests that check BME lengths on simulated alignments (test_inference, test_branch_lengths,
// test_device_cpu, test_device_gpu): the simulation model, its distance models, missing sites, the distance matrix,
// Pauplin's formula and the CPU settings.
#ifndef VECME_TESTS_BME_TEST_HELPERS_HPP
#define VECME_TESTS_BME_TEST_HELPERS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

#include "accelerator.hpp"
#include "core.hpp"
#include "distances.hpp"
#include "inference.hpp"
#include "random.hpp"
#include "simulations.hpp"

namespace {

// The model the alignments are simulated under: GTR rates, unequal base frequencies, gamma alpha 0.5.
inline vecme::SimulationModel truth() {
    vecme::SimulationModel m;
    const double r[vecme::N_GTR_RATES] = {0.5, 2.0, 0.5, 0.5, 2.0, 0.5}, f[vecme::N_BASES] = {0.3, 0.2, 0.2, 0.3};
    for (int k = 0; k < vecme::N_GTR_RATES; ++k)
        m.rates[k] = r[k];
    for (int i = 0; i < vecme::N_BASES; ++i)
        m.freqs[i] = f[i];
    m.alpha = 0.5;
    return m;
}

// The distance model `model`, with truth()'s GTR rates (for GTR and GTR+GC) and alpha (for GTR+GC).
inline vecme::SubstitutionModel distance_model(vecme::Model model) {
    vecme::SubstitutionModel dm;
    dm.model = model;
    const vecme::SimulationModel t = truth();
    for (int k = 0; k < vecme::N_GTR_RATES; ++k)
        dm.rates[k] = t.rates[k];
    dm.alpha = t.alpha;
    return dm;
}

// The N x N distance matrix of an alignment under a distance model, in double precision.
inline std::vector<double> distance_matrix(const vecme::Alignment &aln, const vecme::SubstitutionModel &dm) {
    const int32_t N = aln.n_taxa, W = aln.n_words;
    std::vector<double> D((size_t)N * N, 0.0);
    for (int32_t i = 0; i < N; ++i)
        for (int32_t j = i + 1; j < N; ++j) {
            const vecme::PairCounts c = vecme::count_pair(&aln.packed[(size_t)i * vecme::N_PLANES * W],
                                                          &aln.packed[(size_t)j * vecme::N_PLANES * W], W);
            D[(size_t)i * N + j] = D[(size_t)j * N + i] =
                vecme::pair_distance<double>(vecme::distance_params<double>(dm), c.n);
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
            total += std::ldexp(1.0, 1 - dist[j]) * D[(size_t)i * N + j];
    }
    return total;
}

// The unrooted tree of a rooted Tree (the root's two edges joined).
inline std::vector<std::vector<int32_t>> unrooted(const vecme::Tree &t, int32_t N) {
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

// Clears each site of each sequence with probability `rate` (missing data), and makes sequences 0 and 1 share no
// valid site (0 keeps the first half, 1 the second).
inline void punch_holes(vecme::Alignment &aln, double rate, uint64_t seed) {
    uint64_t st = seed;
    const int32_t W = aln.n_words;
    for (int32_t i = 0; i < aln.n_taxa; ++i) {
        uint64_t *row = &aln.packed[(size_t)i * vecme::N_PLANES * W];
        for (int32_t s = 0; s < aln.n_sites; ++s) {
            bool clear = (double)(vecme::splitmix64(st) >> 11) * 0x1.0p-53 < rate;
            if (i == 0)
                clear = clear || s >= aln.n_sites / 2;
            if (i == 1)
                clear = clear || s < aln.n_sites / 2;
            if (clear)
                for (int p = 0; p < vecme::N_PLANES; ++p)
                    row[(size_t)p * W + s / vecme::WORD_BITS] &= ~(1ULL << (s % vecme::WORD_BITS));
        }
    }
}

// The settings of a CPU run under a distance model: random start, `threads` CPU workers.
inline vecme::Settings cpu_settings(vecme::Model model, vecme::Precision precision = vecme::Precision::Single,
                                    int32_t threads = 2) {
    vecme::Settings opt;
    opt.tree_init = vecme::TreeInit::Random;
    opt.accelerator = vecme::Accelerator::Cpu;
    opt.threads = threads;
    opt.precision = precision;
    opt.substitution_model = distance_model(model);
    return opt;
}

} // namespace

#endif // VECME_TESTS_BME_TEST_HELPERS_HPP
