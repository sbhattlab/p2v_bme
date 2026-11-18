/*
 * start_tree.cpp
 *
 * Sample-and-place and random start trees. The nearest-sample step runs on the devices.
 */
#include "start_tree.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "random.hpp"
#include "timing.hpp"

namespace p2vbme {

void build_random_tree(int32_t n_leaves, uint64_t seed, std::vector<int32_t> &c1, std::vector<int32_t> &c2) {
    uint64_t st = seed;
    // uf: keeps track of which node currently sits at each slot while tree is built bottom-up
    std::vector<int32_t> uf(n_leaves);
    for (int32_t i = 0; i < n_leaves; ++i) {
        uf[i] = i;
    }
    p2vbme::shuffle(uf.begin(), uf.end(), st); // random leaf labels
    c1.assign(2 * n_leaves - 1, 0);
    c2.assign(2 * n_leaves - 1, 0);
    for (int32_t k = 0; k < n_leaves - 1; ++k) {
        int32_t i = (n_leaves - 2) - k;
        int32_t m = n_leaves + k;
        // Draw an ordered phylo2vec entry, decoded into c1/c2.
        int32_t x = (int32_t)p2vbme::uniform_below(st, (uint64_t)(i + 1)); // v_i ~ U{0..i}
        c1[m] = uf[i + 1];
        c2[m] = uf[x];
        uf[x] = m;
    }
}

void build_sample_and_place_tree(int32_t n_leaves, int32_t n0_leaves, uint64_t seed, Devices &devices,
                                 std::vector<int32_t> &out_c1, std::vector<int32_t> &out_c2, SampleAndPlaceStats &stats,
                                 const SearchSettings &settings, int32_t level) {
    const int n_devices = (int)devices.size();
    const double t_begin = wall_sec();
    if (n0_leaves > n_leaves / 2)
        n0_leaves = n_leaves / 2;
    if (n0_leaves < p2vbme::START_SAMPLE_MIN)
        n0_leaves = p2vbme::START_SAMPLE_MIN;
    stats.n0 = n0_leaves;

    // 1. pick n0_leaves sequences at random (partial Fisher-Yates)
    uint64_t st = seed;
    std::vector<int32_t> order(n_leaves);
    for (int32_t i = 0; i < n_leaves; ++i)
        order[i] = i;
    p2vbme::partial_shuffle(order.begin(), order.end(), n0_leaves, st);
    // the sampled sequences, in random order: samp[j] is leaf j of t0
    std::vector<int32_t> samp(order.begin(), order.begin() + n0_leaves);
    // the other sequences (the queries of step 4), sorted
    std::vector<int32_t> qry(order.begin() + n0_leaves, order.end());
    std::sort(qry.begin(), qry.end());
    const int32_t n_q = (int32_t)qry.size();

    // 2. start tree t0 (sub_c1, sub_c2) on the sample: a random tree, or above START_RANDOM_TREE_MAX a
    // sample-and-place tree of its own
    for (int d = 0; d < n_devices; ++d)
        devices[d]->push_subset(samp);
    std::vector<int32_t> sub_c1, sub_c2;
    double dL_inner = 0.0;
    if (n0_leaves > p2vbme::START_RANDOM_TREE_MAX) {
        int32_t n0_inner = std::max(p2vbme::START_RANDOM_TREE_MAX, n0_leaves / p2vbme::START_SAMPLE_DIVISOR);
        SampleAndPlaceStats inner_stats;
        // an independent seed per level (an XOR alone would return to the top-level seed two levels down)
        const uint64_t inner_seed = p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_LEVEL);
        build_sample_and_place_tree(n0_leaves, n0_inner, inner_seed, devices, sub_c1, sub_c2, inner_stats, settings,
                                    level + 1);
        dL_inner = inner_stats.dL_sub;
    } else {
        // independent stream: seed the random subsample tree with a splitmix64 output, not with a value that
        // splitmix64's own state sequence (step 0x9E3779B97F4A7C15) can reach
        build_random_tree(n0_leaves, p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_RANDOM), sub_c1, sub_c2);
    }
    // 3. optimise t0 with the same solver and settings (rounds, stop rule) as the main run
    stats.dL_sub = dL_inner + run_minibatch_bme(n0_leaves, devices, sub_c1, sub_c2, settings, level + 1);
    for (int d = 0; d < n_devices; ++d)
        devices[d]->pop_subset();
    const double t_sub = wall_sec();
    stats.dt_sub = t_sub - t_begin;

    // 4. nearest sampled sequence for every other sequence, queries split across the devices
    std::vector<int32_t> best_pos(n_q, 0);
    ParallelErrors errors;
#pragma omp parallel for num_threads(n_devices) schedule(static, 1)
    for (int d = 0; d < n_devices; ++d) {
        int32_t lo = (int32_t)((int64_t)n_q * d / n_devices), hi = (int32_t)((int64_t)n_q * (d + 1) / n_devices);
        int32_t nq_d = hi - lo;
        try {
            if (nq_d > 0)
                devices[d]->nearest_sampled(qry.data() + lo, nq_d, samp.data(), n0_leaves, best_pos.data() + lo);
        } catch (...) {
            errors.capture();
        }
    }
    errors.rethrow();
    const double t_nn = wall_sec();
    stats.dt_nn = t_nn - t_sub;

    // 5. full tree: a balanced tree per bucket (members in random order, paired two at a time, level by level), then
    // the subsample tree on top.
    // Ids are assigned children-first, so the result is post-order with root 2 n_leaves - 2.
    std::vector<int32_t> bucket_off(n0_leaves + 1, 0), bucket_mem(n_q);
    for (int32_t k = 0; k < n_q; ++k)
        bucket_off[best_pos[k] + 1]++;
    for (int32_t j = 0; j < n0_leaves; ++j) {
        stats.max_bucket = std::max(stats.max_bucket, bucket_off[j + 1]);
        bucket_off[j + 1] += bucket_off[j];
    }
    {
        std::vector<int32_t> fill(bucket_off.begin(), bucket_off.end() - 1);
        for (int32_t k = 0; k < n_q; ++k)
            bucket_mem[fill[best_pos[k]]++] = qry[k];
    }
    out_c1.assign(2 * n_leaves - 1, -1);
    out_c2.assign(2 * n_leaves - 1, -1);
    int32_t next_id = n_leaves;
    std::vector<int32_t> bucket_root(n0_leaves);
    std::vector<int32_t> q;
    for (int32_t j = 0; j < n0_leaves; ++j) {
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
    const int32_t sub_root = 2 * n0_leaves - 2;
    std::vector<int32_t> bfs;
    bfs.reserve(n0_leaves);
    bfs.push_back(sub_root);
    for (size_t i = 0; i < bfs.size(); ++i) {
        int32_t u = bfs[i];
        if (sub_c1[u] >= n0_leaves)
            bfs.push_back(sub_c1[u]);
        if (sub_c2[u] >= n0_leaves)
            bfs.push_back(sub_c2[u]);
    }
    std::vector<int32_t> map_int(2 * n0_leaves - 1, -1);
    for (int32_t i = (int32_t)bfs.size() - 1; i >= 0; --i) {
        int32_t u = bfs[i];
        int32_t a = sub_c1[u], b = sub_c2[u];
        out_c1[next_id] = (a < n0_leaves) ? bucket_root[a] : map_int[a];
        out_c2[next_id] = (b < n0_leaves) ? bucket_root[b] : map_int[b];
        map_int[u] = next_id++;
    }
    if (next_id != 2 * n_leaves - 1 || map_int[sub_root] != 2 * n_leaves - 2) {
        throw std::logic_error("sample-and-place start: the tree has " + std::to_string(next_id) + " nodes, expected " +
                               std::to_string(2 * n_leaves - 1));
    }
    const double t_end = wall_sec();
    stats.dt_build = t_end - t_nn;
    stats.dt_total = t_end - t_begin;
    if (settings.on_event) {
        SearchEvent e;
        e.kind = SearchEvent::StartTree;
        e.level = level;
        e.n_leaves = n_leaves;
        e.init_stats = &stats;
        settings.on_event(e);
    }
}

} // namespace p2vbme
