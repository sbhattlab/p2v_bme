/*
 * start_tree.cpp
 *
 * Sample-and-place and random start trees. The nearest-sample step runs on the devices.
 */
#include "start_tree.hpp"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "random.hpp"

namespace p2vbme {

void make_uniform_random_tree_host(int32_t n, uint64_t seed, std::vector<int32_t> &c1, std::vector<int32_t> &c2) {
    uint64_t st = seed;
    std::vector<int32_t> perm(n), uf(n);
    for (int32_t i = 0; i < n; ++i) {
        perm[i] = i;
        uf[i] = i;
    }
    p2vbme::shuffle(perm.begin(), perm.end(), st);
    c1.assign(2 * n - 1, 0);
    c2.assign(2 * n - 1, 0);
    for (int32_t k = 0; k < n - 1; ++k) {
        int32_t i = (n - 2) - k;
        int32_t m = n + k;
        int32_t x = (int32_t)p2vbme::uniform_below(st, (uint64_t)(i + 1)); /* v_i ~ U{0..i} */
        c1[m] = uf[i + 1];
        c2[m] = uf[x];
        uf[x] = m;
    }
    for (int32_t m = n; m < 2 * n - 1; ++m) {
        if (c1[m] < n)
            c1[m] = perm[c1[m]];
        if (c2[m] < n)
            c2[m] = perm[c2[m]];
    }
}

void build_sample_init_tree(int32_t N, int32_t n0, uint64_t seed, Devices &devices, std::vector<int32_t> &out_c1,
                            std::vector<int32_t> &out_c2, SampleInitStats &stats, const SearchSettings &settings,
                            int32_t level) {
    const int n_gpus = (int)devices.size();
    const double t_begin = wall_sec();
    if (N < p2vbme::START_SAMPLE_MIN_N) /* infer() refuses this; keep the function safe on its own */
        throw std::invalid_argument("sample-and-place start: N = " + std::to_string(N) +
                                    " is too small (need N >= " + std::to_string(p2vbme::START_SAMPLE_MIN_N) + ")");
    if (n0 > N / 2)
        n0 = N / 2;
    if (n0 < p2vbme::START_SAMPLE_MIN)
        n0 = p2vbme::START_SAMPLE_MIN;
    stats.n0 = n0;

    /* 1. random subsample (partial Fisher-Yates) */
    uint64_t st = seed;
    std::vector<int32_t> order(N);
    for (int32_t i = 0; i < N; ++i)
        order[i] = i;
    p2vbme::partial_shuffle(order.begin(), order.end(), n0, st);
    std::vector<int32_t> samp(order.begin(), order.begin() + n0);
    std::vector<int32_t> qry(order.begin() + n0, order.end());
    std::sort(qry.begin(), qry.end());
    const int32_t n_q = (int32_t)qry.size();

    /* 2. optimise the subsample from a Yule-Harding random tree with the run's settings */
    for (int g = 0; g < n_gpus; ++g)
        devices[g]->push_subset(samp);
    std::vector<int32_t> sub_c1, sub_c2;
    if (n0 > p2vbme::START_RANDOM_TREE_MAX) {
        int32_t n0_inner = std::max(p2vbme::START_RANDOM_TREE_MAX, n0 / p2vbme::START_SAMPLE_DIVISOR);
        SampleInitStats inner_stats;
        /* an independent seed per level (an XOR alone would return to the top-level seed two levels down) */
        const uint64_t inner_seed = p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_LEVEL);
        build_sample_init_tree(n0, n0_inner, inner_seed, devices, sub_c1, sub_c2, inner_stats, settings, level + 1);
        /* the sample is refined by the same solver with the same settings (rounds, stop rule) as the main run */
        stats.dL_sub =
            inner_stats.dL_sub + run_multi_gpu_minibatch_bme(n0, devices, sub_c1, sub_c2, settings, level + 1);
    } else {
        /* independent stream: seed the random subsample tree with a splitmix64 output, not with a
         * value that splitmix64's own state sequence (step 0x9E3779B97F4A7C15) can reach */
        make_uniform_random_tree_host(n0, p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_RANDOM), sub_c1, sub_c2);
        stats.dL_sub = run_multi_gpu_minibatch_bme(n0, devices, sub_c1, sub_c2, settings, level + 1);
    }
    for (int g = 0; g < n_gpus; ++g)
        devices[g]->pop_subset();
    const double t_sub = wall_sec();
    stats.dt_sub = t_sub - t_begin;

    /* 3. nearest sampled leaf for every other leaf, queries split across GPUs */
    std::vector<int32_t> best_pos(n_q, 0);
    ParallelErrors errors;
#pragma omp parallel for num_threads(n_gpus) schedule(static, 1)
    for (int g = 0; g < n_gpus; ++g) {
        int32_t lo = (int32_t)((int64_t)n_q * g / n_gpus), hi = (int32_t)((int64_t)n_q * (g + 1) / n_gpus);
        int32_t nq_g = hi - lo;
        try {
            if (nq_g > 0)
                devices[g]->nearest_sampled(qry.data() + lo, nq_g, samp.data(), n0, best_pos.data() + lo);
        } catch (...) {
            errors.capture();
        }
    }
    errors.rethrow();
    const double t_nn = wall_sec();
    stats.dt_nn = t_nn - t_sub;

    /* 4. full tree: a balanced tree per bucket (members in random order, paired two at a time, level by level),
     * then the subsample tree on top. Ids are assigned children-first, so the result is post-order with root 2N-2. */
    std::vector<int32_t> bucket_off(n0 + 1, 0), bucket_mem(n_q);
    for (int32_t k = 0; k < n_q; ++k)
        bucket_off[best_pos[k] + 1]++;
    for (int32_t j = 0; j < n0; ++j) {
        stats.max_bucket = std::max(stats.max_bucket, bucket_off[j + 1]);
        bucket_off[j + 1] += bucket_off[j];
    }
    {
        std::vector<int32_t> fill(bucket_off.begin(), bucket_off.end() - 1);
        for (int32_t k = 0; k < n_q; ++k)
            bucket_mem[fill[best_pos[k]]++] = qry[k];
    }
    out_c1.assign(2 * N - 1, -1);
    out_c2.assign(2 * N - 1, -1);
    int32_t next_id = N;
    std::vector<int32_t> bucket_root(n0);
    std::vector<int32_t> q;
    for (int32_t j = 0; j < n0; ++j) {
        int32_t lo = bucket_off[j], hi = bucket_off[j + 1];
        p2vbme::shuffle(bucket_mem.begin() + lo, bucket_mem.begin() + hi, st);
        q.clear();
        q.push_back(samp[j]);
        for (int32_t a = lo; a < hi; ++a)
            q.push_back(bucket_mem[a]);
        size_t head = 0;
        while (q.size() - head > 1) {
            size_t level_end = q.size();
            while (level_end - head >= 2) {
                int32_t u1 = q[head++];
                int32_t u2 = q[head++];
                out_c1[next_id] = u1;
                out_c2[next_id] = u2;
                q.push_back(next_id++);
            }
        }
        bucket_root[j] = q[head];
    }
    const int32_t sub_root = 2 * n0 - 2;
    std::vector<int32_t> bfs;
    bfs.reserve(n0);
    bfs.push_back(sub_root);
    for (size_t i = 0; i < bfs.size(); ++i) {
        int32_t u = bfs[i];
        if (sub_c1[u] >= n0)
            bfs.push_back(sub_c1[u]);
        if (sub_c2[u] >= n0)
            bfs.push_back(sub_c2[u]);
    }
    std::vector<int32_t> map_int(2 * n0 - 1, -1);
    for (int32_t i = (int32_t)bfs.size() - 1; i >= 0; --i) {
        int32_t u = bfs[i];
        int32_t a = sub_c1[u], b = sub_c2[u];
        out_c1[next_id] = (a < n0) ? bucket_root[a] : map_int[a];
        out_c2[next_id] = (b < n0) ? bucket_root[b] : map_int[b];
        map_int[u] = next_id++;
    }
    if (next_id != 2 * N - 1 || map_int[sub_root] != 2 * N - 2) {
        throw std::logic_error("sample-and-place start: the tree has " + std::to_string(next_id) + " nodes, expected " +
                               std::to_string(2 * N - 1));
    }
    const double t_end = wall_sec();
    stats.dt_build = t_end - t_nn;
    stats.dt_total = t_end - t_begin;
    if (settings.on_event) {
        SearchEvent e;
        e.kind = SearchEvent::StartTree;
        e.level = level;
        e.n_leaves = N;
        e.init_stats = &stats;
        settings.on_event(e);
    }
}

} // namespace p2vbme
