/*
 * config.hpp
 *
 * Every number the method depends on, in one place: the defaults of the user settings (Settings and the
 * command line both read these) and the algorithm's fixed constants. Values are the paper's (GPU suite).
 * Kernel tile sizes stay next to their kernels.
 */
#ifndef P2VBME_CONFIG_HPP
#define P2VBME_CONFIG_HPP

#include <stdint.h>

namespace p2vbme {

// ---- Defaults of the user settings ------------------------------------------------------------------

constexpr int32_t DEFAULT_MAX_ROUNDS = 8;       // outer rounds of the search, then a final cleanup pass
constexpr bool DEFAULT_STOP_WHEN_IDLE = true;   // stop once a round accepts no move (lossless; always on in p2v-bme)
constexpr double DEFAULT_EARLY_STOPPING = 1e-6; // stop once a round gains <= this x the total gain (0 = off)
constexpr double DEFAULT_TIME_LIMIT = 0.0;      // seconds after which no new round starts (0 = none)
constexpr int32_t DEFAULT_PROFILE_CAP = 64;     // leaves kept per clade profile in a window (1..PROFILE_CAP_MAX)
constexpr double DEFAULT_EXACT_CHECK_ETA = 0.0; // exact re-check of windows gaining less than eta (0 = never)
constexpr int32_t DEFAULT_SAMPLE_SIZE = 0;      // sample-and-place start: 0 = min(START_SAMPLE_MAX, N / 10)
constexpr uint64_t DEFAULT_SEED = 1;            // seeds the start tree
constexpr bool DEFAULT_BRANCH_LENGTHS = true;   // estimate BME branch lengths
constexpr int32_t DEFAULT_BL_PROFILE_CAP = 32;  // leaves kept per subtree profile for branch lengths
constexpr int32_t DEFAULT_GTR_REWEIGHT = 2;     // GTR+GC SymWLS: weight updates after the first fit (supplement)

// ---- Limits ------------------------------------------------------------------------------------------

constexpr int32_t PROFILE_CAP_MAX = 256;  // the GPU profile buffer holds PROFILE_CAP_MAX x WORKSPACE_MAX_K entries
constexpr int32_t WORKSPACE_MAX_K = 2048; // components a GPU workspace can hold (windows use K <= 1,500)

// GTR+GC: the range where single-precision distances (the GPU's) were checked against double precision.
// Inside it, 99% of simulated pairs of 1,500 sites were within 0.3% of double precision.
// Outside it, the error grows: about 1% at alpha = 0.001, and several-fold on rare pairs when one exchangeability is
// 1,000 times the others.
// infer() warns outside it in single precision; it requires only alpha > 0 and exchangeabilities > 0.
constexpr double GTR_FLOAT_ALPHA_MIN = 0.05;
constexpr double GTR_FLOAT_ALPHA_MAX = 1e3;
constexpr double GTR_FLOAT_RATE_RATIO_MAX = 100.0; // largest / smallest exchangeability

// ---- Search: windows ----------------------------------------------------------------------------------

// Sweeps of the core step per window, at most (each sweep applies a batch of moves); windows usually stop earlier, as
// soon as no move improves in either labelling.
constexpr int32_t WINDOW_MAX_SWEEPS = 200;
// Internal-vertex budget of a backbone, whole-tree or V-cycle window: 1,498 splits give at most 1,499 pieces, plus the
// outside component, so every window has K <= 1,500 components.
constexpr int32_t WINDOW_MAX_INTERNALS = 1498;
// Internal-vertex budget of a window of the leaf-covering waves.
constexpr int32_t WAVE_WINDOW_MAX_INTERNALS = 1200;
// Internal nodes per component of the leaf-covering waves, at most; also the smallest V-cycle window and the tree size
// above which a round ends with a cleanup pass.
constexpr int32_t WAVE_COMPONENT_CAP = 1050;
// Moves applied together in one sweep, at most (the batch cap B of Algorithm S1; halved after a failed batch).
constexpr int32_t MOVE_BATCH_CAP = 128;
// Exact check (--exact-check-eta): from this N on, the "complete" profiles are capped at EXACT_PROFILE_CAP.
constexpr int32_t EXACT_PROFILE_CAP_MIN_N = 1000000;
constexpr int32_t EXACT_PROFILE_CAP = 2048;

// ---- Search: rounds ------------------------------------------------------------------------------------

constexpr int32_t BACKBONE_MIN_N = 5000; // from this N on, rounds >= 1 add the whole-tree backbone windows
constexpr int32_t VCYCLE_MIN_N = 1000;   // above this N, rounds run the multi-scale V-cycle
// V-cycle window sizes (leaves): VCYCLE_BASE_SCALE x VCYCLE_SCALE_FACTOR^k up to N, then N / 4, N / 16, ... while above
// VCYCLE_FINEST_SCALE, then VCYCLE_FINEST_SCALE.
constexpr int32_t VCYCLE_BASE_SCALE = 4096;
constexpr int32_t VCYCLE_SCALE_FACTOR = 4;
constexpr int32_t VCYCLE_FINEST_SCALE = 3200;

// ---- Start tree (sample-and-place) ----------------------------------------------------------------------

constexpr int32_t START_SAMPLE_MIN_N = 16;   // fewer sequences: the random start
constexpr int32_t START_SAMPLE_MAX = 100000; // default sample: min(START_SAMPLE_MAX, N / START_SAMPLE_DIVISOR)
constexpr int32_t START_SAMPLE_DIVISOR = 10;
constexpr int32_t START_SAMPLE_MIN = 8;         // the sample is clamped to [START_SAMPLE_MIN, N / 2]
constexpr int32_t START_RANDOM_TREE_MAX = 1000; // a sample of at most this size starts from a random tree;
                                                // larger ones are themselves placed on a sample of
                                                // max(START_RANDOM_TREE_MAX, n / START_SAMPLE_DIVISOR)
// Nearest sampled sequence (p-distance for every model): a pair without shared valid sites ranks as p = 0.6485, the
// p-distance whose JC69 distance is 1.5 (what JC69 gives such pairs; the other models give them 2.0).
constexpr float NN_P_NO_OVERLAP = 0.6485f;

} // namespace p2vbme

#endif // P2VBME_CONFIG_HPP
