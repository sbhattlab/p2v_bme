// O(N) BME branch lengths (branch_lengths.cpp and the devices' edge-length pass) against the exact O(N^2) branch
// lengths of Pauplin (2000) / FastME (Desper & Gascuel 2002) in double precision, with tolerances:
//   - complete profiles (cap >= N): every branch length equal to the exact one (float or double precision);
//   - capped profiles (cap 32, the default): the approximation error stays within measured bounds;
//   - Pauplin's identity: the exact lengths sum to the tree's BME length (0.5 sum_x d({x}, outside x)), and so do the
//     O(N) lengths with complete profiles (also the returned total); the root's right child gets 0.
// BranchLengths.* run on the CPU device; BranchLengthsGpu.* on GPU 0 (label gpu).
#include <gtest/gtest.h>

#include <math.h>

#include <algorithm>
#include <string>
#include <vector>

#include "branch_lengths.hpp"
#include "device.hpp"
#include "eigen_dist.hpp"
#include "inference.hpp"
#include "simulations.hpp"
#include "substitution_models.hpp"

namespace {

using namespace p2vbme;

struct Case {
    Tree tree;
    Alignment aln;
    std::vector<double> D; /* N x N, double */
};

Case make_case(int32_t N, Model model, uint64_t seed) {
    Case c;
    c.tree = random_tree(N, 0.01, 0.2, seed);
    SimulationModel sim;
    const double r[6] = {0.5, 2.0, 0.5, 0.5, 2.0, 0.5}, f[4] = {0.3, 0.2, 0.2, 0.3};
    for (int k = 0; k < 6; ++k)
        sim.rates[k] = r[k];
    for (int i = 0; i < 4; ++i)
        sim.freqs[i] = f[i];
    sim.alpha = 0.5;
    c.aln = simulate_alignment(c.tree, 1600, sim, seed + 1);
    SubstitutionModel dm;
    dm.model = model;
    const int32_t W = c.aln.n_words;
    c.D.assign((size_t)N * N, 0.0);
    for (int32_t i = 0; i < N; ++i)
        for (int32_t j = i + 1; j < N; ++j) {
            const PairCounts pc = count_pair(&c.aln.packed[(size_t)i * 5 * W], &c.aln.packed[(size_t)j * 5 * W], W);
            c.D[(size_t)i * N + j] = c.D[(size_t)j * N + i] =
                pair_distance<double>(distance_params<double>(dm, EigenSettings()), pc.n);
        }
    return c;
}

// Exact branch lengths with uncapped dyadic profiles in double (the program's reference, any distance matrix):
// edge above u = d(in u, out u) - (d(children of u) + d(sibling of u, out parent)) / 2. Also the direct Pauplin
// score 0.5 sum_x d({x}, out x).
std::vector<double> exact_lengths(const Tree &t, int32_t N, const std::vector<double> &D, double *direct) {
    const int32_t root = 2 * N - 2;
    std::vector<int32_t> parent(2 * N - 1, -1), post;
    std::vector<int32_t> st = {root};
    while (!st.empty()) {
        const int32_t u = st.back();
        st.pop_back();
        if (u >= N) {
            post.push_back(u);
            parent[t.c1[u]] = u;
            parent[t.c2[u]] = u;
            st.push_back(t.c1[u]);
            st.push_back(t.c2[u]);
        }
    }
    std::reverse(post.begin(), post.end());
    std::vector<std::vector<double>> w_in(2 * N - 2, std::vector<double>(N, 0.0)), w_out = w_in;
    for (int32_t i = 0; i < N; ++i)
        w_in[i][i] = 1.0;
    for (int32_t u : post)
        if (u != root)
            for (int32_t x = 0; x < N; ++x)
                w_in[u][x] = 0.5 * (w_in[t.c1[u]][x] + w_in[t.c2[u]][x]);
    const int32_t r_left = t.c1[root], r_right = t.c2[root];
    w_out[r_left] = w_in[r_right];
    w_out[r_right] = w_in[r_left];
    for (int32_t k = (int32_t)post.size() - 1; k >= 0; --k) {
        const int32_t u = post[k];
        if (u == root)
            continue;
        for (int32_t x = 0; x < N; ++x) {
            w_out[t.c1[u]][x] = 0.5 * (w_out[u][x] + w_in[t.c2[u]][x]);
            w_out[t.c2[u]][x] = 0.5 * (w_out[u][x] + w_in[t.c1[u]][x]);
        }
    }
    auto bdist = [&](const std::vector<double> &a, const std::vector<double> &b) {
        double s = 0.0;
        for (int32_t i = 0; i < N; ++i)
            if (a[i] != 0.0)
                for (int32_t j = 0; j < N; ++j)
                    if (b[j] != 0.0)
                        s += a[i] * b[j] * D[(size_t)i * N + j];
        return s;
    };
    *direct = 0.0;
    for (int32_t x = 0; x < N; ++x)
        *direct += 0.5 * bdist(w_in[x], w_out[x]);
    std::vector<double> bl(2 * N - 2, 0.0);
    for (int32_t u = 0; u < 2 * N - 2; ++u) {
        if (u == r_right)
            continue;
        const double cross = bdist(w_in[u], w_out[u]);
        const double div_in = u >= N ? bdist(w_in[t.c1[u]], w_in[t.c2[u]]) : 0.0;
        double div_out = 0.0;
        if (u == r_left) {
            div_out = r_right >= N ? bdist(w_in[t.c1[r_right]], w_in[t.c2[r_right]]) : 0.0;
        } else {
            const int32_t p = parent[u], sib = t.c1[p] == u ? t.c2[p] : t.c1[p];
            div_out = bdist(w_in[sib], w_out[p]);
        }
        bl[u] = cross - 0.5 * div_in - 0.5 * div_out;
    }
    return bl;
}

void run_all(Devices &devices, bool double_precision, const char *what) {
    for (Model model : {Model::JC69, Model::TN93, Model::GTR_GC})
        for (int32_t N : {20, 64, 200}) {
            Case c = make_case(N, model, 12345 + N);
            SubstitutionModel dm;
            dm.model = model;
            for (auto &dev : devices)
                dev->load(c.aln.packed.data(), N, c.aln.n_words);
            /* complete profiles: exact up to the precision */
            {
                for (auto &dev : devices)
                    dev->set_distance(dm, EigenSettings());
                double direct = 0.0;
                const std::vector<double> exact = exact_lengths(c.tree, N, c.D, &direct);
                std::vector<double> bl;
                const double total = compute_bme_branch_lengths(c.tree, devices, bl, PROFILE_CAP_MAX, nullptr, nullptr,
                                                                double_precision);
                double sum_exact = 0.0, sum_bl = 0.0;
                for (double x : exact)
                    sum_exact += x;
                for (double x : bl)
                    sum_bl += x;
                const double tol_sum = (double_precision ? 1e-10 : 1e-5) * direct;
                EXPECT_NEAR(sum_exact, direct, 1e-12 * direct) << what << " Pauplin's identity, exact lengths";
                EXPECT_NEAR(total, direct, tol_sum) << what << " " << model_name(model) << " N=" << N << " total";
                EXPECT_NEAR(sum_bl, total, 1e-9 * direct) << what << " the total is the lengths' sum";
                EXPECT_EQ(bl[c.tree.c2[2 * N - 2]], 0.0) << what << " the root's right child";
                double max_err = 0.0;
                const int32_t r_right = c.tree.c2[2 * N - 2];
                for (int32_t u = 0; u < 2 * N - 2; ++u)
                    if (u != r_right)
                        max_err = std::max(max_err, fabs(bl[u] - exact[u]));
                EXPECT_LT(max_err, double_precision ? 1e-10 : 2e-5)
                    << what << " " << model_name(model) << " N=" << N << " complete profiles";
            }
            /* capped profiles (cap 32): the measured approximation, with a margin */
            if (N > 32) {
                for (auto &dev : devices)
                    dev->set_distance(dm, EigenSettings());
                std::vector<double> bl, exact;
                double direct = 0.0;
                exact = exact_lengths(c.tree, N, c.D, &direct);
                compute_bme_branch_lengths(c.tree, devices, bl, 32, nullptr, nullptr, double_precision);
                double max_err = 0.0, mean = 0.0;
                const int32_t r_right = c.tree.c2[2 * N - 2];
                for (int32_t u = 0; u < 2 * N - 2; ++u)
                    if (u != r_right) {
                        const double err = fabs(bl[u] - exact[u]);
                        max_err = std::max(max_err, err);
                        mean += err;
                    }
                mean /= (2 * N - 3);
                printf("%s %-6s N=%3d cap 32: max |bl - exact| %.3e, mean %.3e\n", what, model_name(model), N, max_err,
                       mean);
                EXPECT_LT(max_err, 0.05) << what << " " << model_name(model) << " N=" << N;
                EXPECT_LT(mean, 0.005) << what << " " << model_name(model) << " N=" << N;
            }
        }
}

} // namespace

TEST(BranchLengths, CpuSingle) {
    Devices devices = make_cpu_devices(2, false);
    run_all(devices, false, "CPU single");
}

TEST(BranchLengths, CpuDouble) {
    Devices devices = make_cpu_devices(2, true);
    run_all(devices, true, "CPU double");
}

TEST(BranchLengthsGpu, Gpu) {
    Devices devices;
    devices.push_back(make_gpu_device(0));
    run_all(devices, false, "GPU");
}
