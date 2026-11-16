// Tests of src/random.hpp: reference values of splitmix64, and the draws equal to the original core's loops.
#include <gtest/gtest.h>

#include <stdint.h>

#include <algorithm>
#include <math.h>

#include <numeric>
#include <vector>

#include "random.hpp"

namespace {

// Vigna's splitmix64.c from state 0 (also computed independently in Python for this test).
TEST(Splitmix64, ReferenceValues) {
    uint64_t s = 0;
    EXPECT_EQ(p2vbme::splitmix64(s), 0xE220A8397B1DCDAFULL);
    EXPECT_EQ(p2vbme::splitmix64(s), 0x6E789E6AA1B965F4ULL);
    EXPECT_EQ(p2vbme::splitmix64(s), 0x06C45D188009454FULL);
    s = 1;
    EXPECT_EQ(p2vbme::splitmix64(s), 0x910A2DEC89025CC1ULL);
}

TEST(UniformBelow, IsTheOutputModuloN) {
    uint64_t a = 42, b = 42;
    for (uint64_t n : {1ULL, 2ULL, 7ULL, 1000ULL, 1ULL << 40}) {
        const uint64_t x = p2vbme::uniform_below(a, n);
        EXPECT_EQ(x, p2vbme::splitmix64(b) % n);
        EXPECT_LT(x, n);
    }
}

// The loops of the search core before random.hpp (make_uniform_random_tree_host, build_sample_init_tree).
void old_shuffle_back(std::vector<int32_t> &v, uint64_t &st) {
    for (int32_t i = (int32_t)v.size() - 1; i > 0; --i) {
        int32_t j = (int32_t)(p2vbme::splitmix64(st) % (uint64_t)(i + 1));
        std::swap(v[i], v[j]);
    }
}
void old_bucket_shuffle(std::vector<int32_t> &v, int32_t lo, int32_t hi, uint64_t &st) {
    for (int32_t a = hi - 1; a > lo; --a) {
        int32_t b = lo + (int32_t)(p2vbme::splitmix64(st) % (uint64_t)(a - lo + 1));
        std::swap(v[a], v[b]);
    }
}
void old_partial_shuffle(std::vector<int32_t> &v, int32_t n0, uint64_t &st) {
    const int32_t N = (int32_t)v.size();
    for (int32_t i = 0; i < n0; ++i) {
        int32_t j = i + (int32_t)(p2vbme::splitmix64(st) % (uint64_t)(N - i));
        std::swap(v[i], v[j]);
    }
}

std::vector<int32_t> iota_vec(int32_t n) {
    std::vector<int32_t> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

TEST(Shuffle, SameStreamAsTheOriginalCore) {
    for (uint64_t seed : {1ULL, 20261105ULL, 0xDEADBEEFULL}) {
        std::vector<int32_t> a = iota_vec(1000), b = a;
        uint64_t sa = seed, sb = seed;
        p2vbme::shuffle(a.begin(), a.end(), sa);
        old_shuffle_back(b, sb);
        EXPECT_EQ(a, b);
        EXPECT_EQ(sa, sb); // same number of draws
        std::vector<int32_t> sorted = a;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(sorted, iota_vec(1000));

        std::vector<int32_t> c = iota_vec(50), d = c;
        p2vbme::shuffle(c.begin() + 10, c.begin() + 37, sa);
        old_bucket_shuffle(d, 10, 37, sb);
        EXPECT_EQ(c, d);

        std::vector<int32_t> e = iota_vec(5000), f = e;
        p2vbme::partial_shuffle(e.begin(), e.end(), 500, sa);
        old_partial_shuffle(f, 500, sb);
        EXPECT_EQ(e, f);
        EXPECT_EQ(sa, sb);
    }
}

TEST(Shuffle, EdgeCases) {
    uint64_t s = 3;
    std::vector<int32_t> empty, one = {7};
    p2vbme::shuffle(empty.begin(), empty.end(), s);
    p2vbme::shuffle(one.begin(), one.end(), s);
    EXPECT_EQ(one[0], 7);
    EXPECT_EQ(s, 3u); // no draws for fewer than two elements
    std::vector<int32_t> v = iota_vec(10);
    p2vbme::partial_shuffle(v.begin(), v.end(), 10, s); // k = n: a full shuffle
    std::vector<int32_t> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, iota_vec(10));
}

// Each element is equally likely to be drawn first (within 4 standard errors).
TEST(PartialShuffle, FirstElementIsUniform) {
    const int n = 5, draws = 50000;
    std::vector<int> count(n, 0);
    uint64_t s = 11;
    for (int t = 0; t < draws; ++t) {
        std::vector<int32_t> v = iota_vec(n);
        p2vbme::partial_shuffle(v.begin(), v.end(), 1, s);
        ++count[v[0]];
    }
    const double expected = (double)draws / n, sd = std::sqrt(draws * (1.0 / n) * (1.0 - 1.0 / n));
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(count[i], expected, 4.0 * sd) << i;
}

TEST(DeriveSeed, SameAsTheOriginalInlineCode) {
    for (uint64_t seed : {1ULL, 20261105ULL}) {
        uint64_t s_in = seed ^ 0x9E3779B97F4A7C15ULL;
        EXPECT_EQ(p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_LEVEL), p2vbme::splitmix64(s_in));
        uint64_t tmp = seed ^ 0xA0761D6478BD642FULL;
        EXPECT_EQ(p2vbme::derive_seed(seed, p2vbme::SALT_START_TREE_RANDOM), p2vbme::splitmix64(tmp));
    }
    // two levels down does not return to the top seed (as a plain XOR would)
    const uint64_t s1 = p2vbme::derive_seed(5, p2vbme::SALT_START_TREE_LEVEL);
    EXPECT_NE(p2vbme::derive_seed(s1, p2vbme::SALT_START_TREE_LEVEL), 5u);
}

} // namespace
