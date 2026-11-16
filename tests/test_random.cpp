// Tests of src/random.hpp: splitmix64 against its reference values, and what each draw promises: its range, a
// permutation, equally likely outcomes, and the number of draws it takes from the stream.
#include <gtest/gtest.h>

#include <stdint.h>

#include <algorithm>
#include <math.h>

#include <map>
#include <numeric>
#include <vector>

#include "random.hpp"

namespace {

// 0, 1, ..., n - 1
std::vector<int32_t> iota_vec(int32_t n) {
    std::vector<int32_t> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

// The state after `draws` calls of splitmix64 from `state`: each call adds the same constant
uint64_t advanced(uint64_t state, uint64_t draws) { return state + draws * 0x9E3779B97F4A7C15ULL; }

// Every one of n_outcomes outcomes was seen, each within 4 standard errors of trials / n_outcomes
void expect_uniform(const std::map<std::vector<int32_t>, int> &count, int n_outcomes, int trials) {
    EXPECT_EQ((int)count.size(), n_outcomes);
    const double p = 1.0 / n_outcomes, sd = std::sqrt(trials * p * (1.0 - p));
    for (const auto &[outcome, c] : count)
        EXPECT_NEAR(c, trials * p, 4.0 * sd);
}

// ---- splitmix64 ----

// Outputs from states 0 and 1 equal Vigna's splitmix64.c (also computed independently in Python)
TEST(Splitmix64, ReferenceValues) {
    uint64_t s = 0;
    EXPECT_EQ(p2vbme::splitmix64(s), 0xE220A8397B1DCDAFULL);
    EXPECT_EQ(p2vbme::splitmix64(s), 0x6E789E6AA1B965F4ULL);
    EXPECT_EQ(p2vbme::splitmix64(s), 0x06C45D188009454FULL);
    s = 1;
    EXPECT_EQ(p2vbme::splitmix64(s), 0x910A2DEC89025CC1ULL);
}

// ---- uniform_below ----

// The next splitmix64 output modulo n, so always below n (n from 1 to 2^40)
TEST(UniformBelow, IsTheOutputModuloN) {
    uint64_t a = 42, b = 42;
    for (uint64_t n : {1ULL, 2ULL, 7ULL, 1000ULL, 1ULL << 40}) {
        const uint64_t x = p2vbme::uniform_below(a, n);
        EXPECT_EQ(x, p2vbme::splitmix64(b) % n);
        EXPECT_LT(x, n);
    }
}

// ---- shuffle ----

// The result is a permutation of the input, for several seeds
TEST(Shuffle, IsAPermutation) {
    for (uint64_t seed : {1ULL, 20261105ULL, 0xDEADBEEFULL}) {
        std::vector<int32_t> v = iota_vec(1000);
        uint64_t s = seed;
        p2vbme::shuffle(v.begin(), v.end(), s);
        std::sort(v.begin(), v.end());
        EXPECT_EQ(v, iota_vec(1000));
    }
}

// All 6 orders of 3 elements are equally likely (60,000 shuffles)
TEST(Shuffle, AllOrdersEquallyLikely) {
    const int trials = 60000;
    std::map<std::vector<int32_t>, int> count;
    uint64_t s = 11;
    for (int t = 0; t < trials; ++t) {
        std::vector<int32_t> v = iota_vec(3);
        p2vbme::shuffle(v.begin(), v.end(), s);
        ++count[v];
    }
    expect_uniform(count, 6, trials);
}

// n elements take n - 1 draws
TEST(Shuffle, TakesNMinusOneDraws) {
    std::vector<int32_t> v = iota_vec(1000);
    uint64_t s = 7;
    p2vbme::shuffle(v.begin(), v.end(), s);
    EXPECT_EQ(s, advanced(7, 999));
}

// On a subrange [lo, hi), as for each placed group: a permutation of the subrange, the rest left in place
TEST(Shuffle, SubrangeOnly) {
    std::vector<int32_t> v = iota_vec(50);
    uint64_t s = 7;
    p2vbme::shuffle(v.begin() + 10, v.begin() + 37, s);
    for (int32_t i = 0; i < 10; ++i)
        EXPECT_EQ(v[i], i);
    for (int32_t i = 37; i < 50; ++i)
        EXPECT_EQ(v[i], i);
    std::sort(v.begin() + 10, v.begin() + 37);
    EXPECT_EQ(v, iota_vec(50));
    EXPECT_EQ(s, advanced(7, 26));
}

// Fewer than two elements: nothing moves and no draw is made
TEST(Shuffle, EmptyAndOneElement) {
    uint64_t s = 3;
    std::vector<int32_t> empty, one = {7};
    p2vbme::shuffle(empty.begin(), empty.end(), s);
    p2vbme::shuffle(one.begin(), one.end(), s);
    EXPECT_EQ(one[0], 7);
    EXPECT_EQ(s, 3u);
}

// ---- partial_shuffle ----

// The result is a permutation of the input, after k draws
TEST(PartialShuffle, IsAPermutationAfterKDraws) {
    std::vector<int32_t> v = iota_vec(5000);
    uint64_t s = 7;
    p2vbme::partial_shuffle(v.begin(), v.end(), 500, s);
    EXPECT_EQ(s, advanced(7, 500));
    std::sort(v.begin(), v.end());
    EXPECT_EQ(v, iota_vec(5000));
}

// The first k elements are a uniform random k-subset in random order: all 12 ordered pairs from 4 elements are
// equally likely (60,000 draws)
TEST(PartialShuffle, AllOrderedSubsetsEquallyLikely) {
    const int trials = 60000;
    std::map<std::vector<int32_t>, int> count;
    uint64_t s = 11;
    for (int t = 0; t < trials; ++t) {
        std::vector<int32_t> v = iota_vec(4);
        p2vbme::partial_shuffle(v.begin(), v.end(), 2, s);
        ++count[{v[0], v[1]}];
    }
    expect_uniform(count, 12, trials);
}

// k = n is a full shuffle (a permutation); k = 0 moves nothing and makes no draw
TEST(PartialShuffle, AllOrNone) {
    uint64_t s = 3;
    std::vector<int32_t> v = iota_vec(10);
    p2vbme::partial_shuffle(v.begin(), v.end(), 10, s);
    std::sort(v.begin(), v.end());
    EXPECT_EQ(v, iota_vec(10));

    const uint64_t before = s;
    std::vector<int32_t> w = iota_vec(10);
    p2vbme::partial_shuffle(w.begin(), w.end(), 0, s);
    EXPECT_EQ(w, iota_vec(10));
    EXPECT_EQ(s, before);
}

// ---- derive_seed ----

// One splitmix64 output from seed XOR salt, for both salts
TEST(DeriveSeed, IsSplitmix64OfSeedXorSalt) {
    for (uint64_t seed : {1ULL, 20261105ULL}) {
        for (uint64_t salt : {p2vbme::SALT_START_TREE_LEVEL, p2vbme::SALT_START_TREE_RANDOM}) {
            uint64_t s = seed ^ salt;
            EXPECT_EQ(p2vbme::derive_seed(seed, salt), p2vbme::splitmix64(s));
        }
    }
}

// Two levels down with the same salt does not return to the parent seed, as a plain XOR would
TEST(DeriveSeed, TwoLevelsDownIsNotTheParentSeed) {
    const uint64_t s1 = p2vbme::derive_seed(5, p2vbme::SALT_START_TREE_LEVEL);
    EXPECT_NE(p2vbme::derive_seed(s1, p2vbme::SALT_START_TREE_LEVEL), 5u);
}

} // namespace
