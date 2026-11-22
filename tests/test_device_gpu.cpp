/*
 * test_device_gpu.cpp
 *
 * The GPU device (device_gpu.cu) on GPU 0. Label gpu.
 *   - its distances under every model against the CPU formulas (distances.hpp) on the same pair counts, through the
 *     edge-length pass: single pairs and profile-weighted sums (the JC69 tiled path and the per-pair path of the
 *     other models), on an alignment with missing sites and a pair without shared sites;
 *   - its errors: a workspace beyond the GPU's memory, a GPU that does not exist.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <stdint.h>

#include <string>
#include <vector>

#include "bme_test_helpers.hpp"
#include "device.hpp"
#include "distances.hpp"
#include "fake_device_test_helpers.hpp"
#include "simulations.hpp"

namespace {

using namespace p2vbme;

const Model MODELS[] = {Model::JC69, Model::K80, Model::F81, Model::F84, Model::TN93, Model::LogDet, Model::GTR_GC};

double cpu_distance(const Alignment &aln, const SubstitutionModel &dm, int32_t a, int32_t b) {
    if (a == b)
        return 0.0;
    const int32_t W = aln.n_words;
    const PairCounts c = count_pair(&aln.packed[(size_t)a * N_PLANES * W], &aln.packed[(size_t)b * N_PLANES * W], W);
    return pair_distance<double>(distance_params<double>(dm), c.n);
}

size_t gpu_free_bytes() {
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(0);
    cudaMemGetInfo(&free_b, &total_b);
    return free_b;
}

} // namespace

// Single pairs (one leaf on each side) and weighted profiles (the JC69 tiled path; the per-pair path otherwise).
TEST(DeviceGpu, EdgeLengthPassMatchesCpu) {
    const int32_t N = 60;
    const Tree tree = random_tree(N, 0.01, 0.3, 11);
    Alignment aln = simulate_alignment(tree, 3000, truth(), 12);
    punch_holes(aln, 0.1, 13);
    auto dev = make_gpu_device(0);
    dev->load(aln.packed.data(), N, aln.n_words);

    // jobs 0..N-1: pairs (i, i + 1 mod N) and outside leaf i + 2 (job 0: leaves 0 and 1, which share no site)
    std::vector<int32_t> c1(N), c2(N), out_len(N, 1), out_leaves(N), in_len(2, 0);
    std::vector<float> out_w(N, 1.0f);
    for (int32_t i = 0; i < N; ++i) {
        c1[i] = i;
        c2[i] = (i + 1) % N;
        out_leaves[i] = (i + 2) % N;
    }
    // job N: two "internal nodes" with profiles of 40 and 30 leaves and an outside profile of 25 leaves
    const int32_t cap = 40;
    std::vector<int32_t> in_leaves(2 * cap, 0), out_leaves_p(cap, 0);
    std::vector<float> in_w(2 * cap, 0.0f), out_w_p(cap, 0.0f);
    in_len = {40, 30};
    for (int32_t k = 0; k < 40; ++k) {
        in_leaves[k] = k;
        in_w[k] = (k + 1) / 820.0f;
    }
    for (int32_t k = 0; k < 30; ++k) {
        in_leaves[cap + k] = 30 + k;
        in_w[cap + k] = 1.0f / 30.0f;
    }
    for (int32_t k = 0; k < 25; ++k) {
        out_leaves_p[k] = 2 * k + 1;
        out_w_p[k] = 1.0f / 25.0f;
    }

    for (Model model : MODELS) {
        const SubstitutionModel dm = distance_model(model);
        dev->set_distance(dm);
        std::vector<double> dAB(N), dAO(N), dBO(N);
        const double tol = model == Model::GTR_GC ? 2e-4 : 2e-5; /* float on the GPU, double here */
        // single pairs: cap 1 for the outside profiles (in_* unused)
        dev->edge_lengths(N, 1, N, c1.data(), c2.data(), 2, in_len.data(), in_leaves.data(), in_w.data(),
                          out_len.data(), out_leaves.data(), out_w.data(), dAB.data(), dAO.data(), dBO.data());
        for (int32_t i = 0; i < N; ++i) {
            const double ab = cpu_distance(aln, dm, c1[i], c2[i]), ao = cpu_distance(aln, dm, c1[i], out_leaves[i]);
            EXPECT_NEAR(dAB[i], ab, tol * ab + 1e-6) << model_name(model) << " pair " << i;
            EXPECT_NEAR(dAO[i], ao, tol * ao + 1e-6) << model_name(model) << " pair " << i;
        }
        EXPECT_EQ(dAB[0], (float)NO_OVERLAP_DISTANCE) << model_name(model) << ": leaves 0 and 1 share no site";

        // profiles
        const int32_t pc1 = N, pc2 = N + 1;
        double pAB, pAO, pBO;
        dev->edge_lengths(N, cap, 1, &pc1, &pc2, 2, in_len.data(), in_leaves.data(), in_w.data(),
                          std::vector<int32_t>{25}.data(), out_leaves_p.data(), out_w_p.data(), &pAB, &pAO, &pBO);
        double eAB = 0.0, eAO = 0.0, eBO = 0.0;
        for (int32_t p = 0; p < 40; ++p)
            for (int32_t q = 0; q < 30; ++q)
                eAB += (double)in_w[p] * in_w[cap + q] * cpu_distance(aln, dm, in_leaves[p], in_leaves[cap + q]);
        for (int32_t q = 0; q < 25; ++q) {
            for (int32_t p = 0; p < 40; ++p)
                eAO += (double)in_w[p] * out_w_p[q] * cpu_distance(aln, dm, in_leaves[p], out_leaves_p[q]);
            for (int32_t p = 0; p < 30; ++p)
                eBO += (double)in_w[cap + p] * out_w_p[q] * cpu_distance(aln, dm, in_leaves[cap + p], out_leaves_p[q]);
        }
        EXPECT_NEAR(pAB, eAB, 1e-4 * eAB) << model_name(model);
        EXPECT_NEAR(pAO, eAO, 1e-4 * eAO) << model_name(model);
        EXPECT_NEAR(pBO, eBO, 1e-4 * eBO) << model_name(model);
    }
}

// A workspace far beyond any GPU: a clean out-of-memory error naming the GPU, the size and the free memory.
// The ~1.2 GB allocated before the failing allocation is freed again (GPU-wide free memory, so a 512 MB margin
// for other processes).
TEST(DeviceGpu, OutOfMemoryIsReported) {
    make_gpu_device(0, 64); /* creates the CUDA context first */
    const size_t before = gpu_free_bytes();
    const std::string msg = thrown_message([] { make_gpu_device(0, 300000); });
    const size_t after = gpu_free_bytes();
    EXPECT_NE(msg.find("out of GPU memory on GPU 0"), std::string::npos) << msg;
    EXPECT_NE(msg.find("MiB requested"), std::string::npos) << msg;
    EXPECT_NE(msg.find("MiB free of"), std::string::npos) << msg;
    EXPECT_LT((double)before - (double)after, 512.0 * 1048576.0) << "memory left behind by the failed device";
}

TEST(DeviceGpu, CudaErrorsThrow) {
    const std::string msg = thrown_message([] { make_gpu_device(97); });
    EXPECT_NE(msg.find("GPU 97 does not exist"), std::string::npos) << msg;
}
