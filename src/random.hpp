/*
 * random.hpp
 *
 * The package's random number generator, splitmix64 (Steele, Lea & Flood 2014; code as in Vigna
 * 2015), and the draws built on it. Every random choice reads a stream seeded from --seed. The
 * draws are our own, not std::uniform_int_distribution or std::shuffle, whose output differs
 * between standard libraries.
 *
 * References
 *   Steele, G. L., Lea, D. & Flood, C. H. (2014). Fast splittable pseudorandom number generators.
 *     OOPSLA 2014, 453-472. doi:10.1145/2660193.2660195
 *   Vigna, S. (2015). splitmix64.c (public domain). https://prng.di.unimi.it/splitmix64.c
 */
#ifndef P2VBME_RANDOM_HPP
#define P2VBME_RANDOM_HPP

#include <stddef.h>
#include <stdint.h>

#include <utility>

namespace p2vbme {

// Advances the state and returns the next 64-bit output.
inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform integer in [0, n), n >= 1, by reduction modulo n (bias below n / 2^64).
inline uint64_t uniform_below(uint64_t &state, uint64_t n) { return splitmix64(state) % n; }

// Fisher-Yates shuffle of [first, last) from the back: position a swaps with a uniform b in [0, a].
template <class RandomIt> void shuffle(RandomIt first, RandomIt last, uint64_t &state) {
    const ptrdiff_t n = last - first;
    for (ptrdiff_t a = n - 1; a > 0; --a) {
        const ptrdiff_t b = (ptrdiff_t)uniform_below(state, (uint64_t)(a + 1));
        std::swap(first[a], first[b]);
    }
}

// Partial Fisher-Yates from the front: afterwards [first, first + k) is a uniform random k-subset
// of [first, last) in random order.
template <class RandomIt> void partial_shuffle(RandomIt first, RandomIt last, ptrdiff_t k, uint64_t &state) {
    const ptrdiff_t n = last - first;
    for (ptrdiff_t i = 0; i < k; ++i) {
        const ptrdiff_t j = i + (ptrdiff_t)uniform_below(state, (uint64_t)(n - i));
        std::swap(first[i], first[j]);
    }
}

// Seed of an independent child stream: one splitmix64 output from seed XOR salt. A plain XOR is not
// enough: applied twice with the same salt it returns the parent's seed, and seeds a splitmix64
// state sequence can reach would replay the parent's stream.
inline uint64_t derive_seed(uint64_t seed, uint64_t salt) {
    uint64_t s = seed ^ salt;
    return splitmix64(s);
}

// Salt of the stream for the next level of the recursive start tree
constexpr uint64_t SALT_START_TREE_LEVEL = 0x9E3779B97F4A7C15ULL;
// Salt of the stream for the random tree of the smallest subsample
constexpr uint64_t SALT_START_TREE_RANDOM = 0xA0761D6478BD642FULL;

} // namespace p2vbme

#endif // P2VBME_RANDOM_HPP
