/*
 * config.hpp
 *
 * The numbers the method depends on: the defaults of the user settings (Settings reads them, and the command line
 * through Settings) and the algorithm's fixed constants.
 * A few stay next to their only user: kernel tile sizes, the salts of the random streams (random.hpp), the reweighting
 * passes (eigen_dist.hpp), the GTR+GC fit's (gtr_fit.hpp) and the component sizes of the waves (search.cpp).
 * Most values are the paper's.
 * Changed since: WINDOW_MIN_K 6 -> 4 (trees of 4 and 5 leaves were never searched), and JC69 now uses
 * NO_OVERLAP_DISTANCE and MAX_DISTANCE like every model (it had 1.5 and a cap at p = 0.74), so NN_P_NO_OVERLAP
 * 0.6485 -> 0.6979.
 */
#ifndef P2VBME_CONFIG_HPP
#define P2VBME_CONFIG_HPP

#include <cstdint>
#include <limits>

namespace p2vbme {

// ---- Defaults of the user settings ------------------------------------------------------------------

constexpr int32_t DEFAULT_MAX_ROUNDS = 8;      // search rounds, then a cleanup pass if N > WAVE_COMPONENT_CAP
constexpr double DEFAULT_TOL = 1e-6;           // stop once a round gains <= this x the total gain (0 = zero gain only)
constexpr double DEFAULT_TIME_LIMIT = 0.0;     // seconds after which no new round starts (0 = none)
constexpr int32_t DEFAULT_PROFILE_CAP = 64;    // leaves kept per clade profile in a window (1..PROFILE_CAP_MAX)
constexpr int32_t DEFAULT_SAMPLE_SIZE = 0;     // sample-and-place start: 0 = the default size (see "Start tree")
constexpr uint64_t DEFAULT_SEED = 1;           // seeds the start tree and the GTR+GC fit's pairs
constexpr bool DEFAULT_BRANCH_LENGTHS = true;  // estimate BME branch lengths
constexpr int32_t DEFAULT_BL_PROFILE_CAP = 32; // leaves kept per subtree profile for branch lengths
constexpr int32_t DEFAULT_GTR_FIT_PAIRS = 100000; // GTR+GC: sequence pairs drawn to fit the rates and alpha

// ---- Limits ------------------------------------------------------------------------------------------

constexpr int32_t PROFILE_CAP_MAX = 256; // the GPU profile buffer holds PROFILE_CAP_MAX x WORKSPACE_MAX_K entries

// ---- Distances -----------------------------------------------------------------------------------------

// The search needs a finite distance for every pair, but the formulas give none in two cases:
// sequences too different (a logarithm's argument reaches 0) and a pair without a shared valid site (no data).
// These constants fill both cases, for every model.
// They are the values of the paper's GTR+GC code; no reason for them is recorded.

// Distance of a pair without a shared valid site.
constexpr double NO_OVERLAP_DISTANCE = 2.0;
// Largest distance.
// Logarithms' arguments are floored at LOG_ARG_MIN = 1e-4, which bounds each term at ln(1e4) = 9.21.
// JC69, for example, reaches at most 3/4 * 9.21 = 6.91, for p >= 3/4 (the mismatch share of unrelated sequences).
constexpr double MAX_DISTANCE = 10.0;

// Smallest argument of a logarithm in the closed forms (neg_log_link), and smallest GTR+GC singular value.
constexpr double LOG_ARG_MIN = 1e-4;
// F84 and TN93 fall back to F81's form when a frequency term they divide by (pi_R, pi_Y, A, C, k1, k2, k3) is below
// this.
constexpr double FREQ_TERM_MIN = 1e-6;
// A positive floor for divisions and logs, far below anything a real pair produces.
constexpr double EPS = 1e-30;

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
// The smallest BME gains (decreases of L) that count, by level: a move, a batch of moves, a window.
// A move is a candidate only if its predicted BME change is below -MOVE_MIN_GAIN (a float, as the scores may be).
constexpr float MOVE_MIN_GAIN = 1e-6f;
// A batch of moves is kept only if it lowers the BME length by more than this; otherwise it is undone.
constexpr double BATCH_MIN_GAIN = 1e-8;
// A window's new skeleton is kept only if it lowers the BME length by more than this.
constexpr double WINDOW_MIN_GAIN = 1e-7;
// The score of "no move" (a row or regraft edge the move scores leave out): larger than any BME change.
constexpr double NO_MOVE = std::numeric_limits<double>::infinity();
// Internal-vertex budget of a window: 1,498 splits give at most 1,499 pieces, plus the outside component, so every
// window has K <= 1,500 pieces.
constexpr int32_t WINDOW_MAX_INTERNALS = 1498;
// Pieces a device workspace holds (the GPU buffers are allocated for it): the largest window above.
constexpr int32_t WORKSPACE_MAX_K = WINDOW_MAX_INTERNALS + 2;
// Smallest window: below 4 pieces there is a single unrooted tree shape, so nothing to move.
// Also the smallest tree the waves visit.
constexpr int32_t WINDOW_MIN_K = 4;
// Internal nodes per component of the leaf-covering waves, at most; also the smallest V-cycle window and the tree size
// above which rounds run the V-cycle and end with a cleanup pass.
constexpr int32_t WAVE_COMPONENT_CAP = 1050;
// A component has at most WAVE_COMPONENT_CAP + 4 internal nodes (the root's part, under 5, takes one child component),
// so a wave window always covers its whole component.
static_assert(WAVE_COMPONENT_CAP + 4 <= WINDOW_MAX_INTERNALS, "a wave component must fit one window");
// Moves applied together in one sweep, at most (the batch cap B of Algorithm S1; halved after a failed batch).
constexpr int32_t MOVE_BATCH_CAP = 128;
// Added to the batch cap after a batch that lowered L, up to MOVE_BATCH_CAP (it only acts after a halving).
constexpr int32_t MOVE_BATCH_CAP_GROWTH = 16;

// ---- Search: rounds ------------------------------------------------------------------------------------

constexpr int32_t BACKBONE_MIN_N = 5000; // from this N on, rounds >= 1 add the whole-tree backbone windows
// V-cycle window sizes (leaves): VCYCLE_BASE_SCALE x VCYCLE_SCALE_FACTOR^k up to N, then N / 4, N / 16, ... while above
// VCYCLE_FINEST_SCALE, then VCYCLE_FINEST_SCALE.
constexpr int32_t VCYCLE_BASE_SCALE = 4096;
constexpr int32_t VCYCLE_SCALE_FACTOR = 4;
constexpr int32_t VCYCLE_FINEST_SCALE = 3200;

// ---- Start tree (sample-and-place) ----------------------------------------------------------------------

constexpr int32_t START_SAMPLE_MIN_N = 16;   // fewer sequences: the random start
constexpr int32_t START_SAMPLE_MAX = 100000; // default sample: min(START_SAMPLE_MAX, N / START_SAMPLE_DIVISOR)
constexpr int32_t START_SAMPLE_DIVISOR = 10;
// The sample is clamped to [START_SAMPLE_MIN, N / 2]; half of START_SAMPLE_MIN_N, so that range is never empty.
constexpr int32_t START_SAMPLE_MIN = START_SAMPLE_MIN_N / 2;
// A sample of at most this size starts from a random tree; larger ones are themselves placed on a sample of
// max(START_RANDOM_TREE_MAX, n / START_SAMPLE_DIVISOR).
constexpr int32_t START_RANDOM_TREE_MAX = 1000;
// Placement finds each query's nearest sampled sequence by p-distance (the share of mismatches, for every model); a
// pair without a shared valid site counts as this p.
// It is NO_OVERLAP_DISTANCE seen as a p-distance: the p at which JC69 reaches that distance,
// p = 3/4 (1 - e^(-4 d / 3)) with d = NO_OVERLAP_DISTANCE = 2, so 3/4 (1 - e^(-8/3)) = 0.69789.
// Written out because std::exp is not constexpr; test_config checks it against NO_OVERLAP_DISTANCE.
constexpr float NN_P_NO_OVERLAP = 0.6979f;

} // namespace p2vbme

#endif // P2VBME_CONFIG_HPP
