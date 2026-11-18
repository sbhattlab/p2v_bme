/*
 * search.hpp
 *
 * The minibatch BME search on the host: rounds of windows (backbone, leaf-covering waves, multi-scale V-cycle),
 * each window a skeleton of K <= 1,500 components summarised by weighted leaf profiles and solved by a Device.
 * Also the Phylo2Vec encoding helpers the windows and the devices share. Internal to the package.
 */
#ifndef P2VBME_SEARCH_HPP
#define P2VBME_SEARCH_HPP

#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "config.hpp"
#include "device.hpp"

namespace p2vbme {

struct SearchSettings;

// Progress of a search, reported through SearchSettings::on_event (the core keeps no diagnostic state and prints
// nothing).
struct SearchEvent {
    enum Kind { Start, Stage, Stop } kind = Stage;
    int32_t level = 0;      // 0 = the main search; k >= 1 = a search inside the sample-and-place start, k levels down
    int32_t n_leaves = 0;   // N of this search
    int32_t round = 0;      // Stage: the outer round (the final cleanup pass has round = rounds run)
    const char *stage = ""; // Stage: backbone1, comp_waves, backbone2, scale, round_end
    int64_t detail = 0;     // Stage: waves (comp_waves), scale cap (scale), -1 (round_end), 0 (backbones)
    double total_dL = 0.0;  // Stage: BME change of this search so far
    const std::vector<int32_t> *c1 = NULL, *c2 = NULL; // Stage round_end: the current tree
    int32_t n_devices = 0;                             // Start
    const SearchSettings *settings = NULL;             // Start: the settings of this search
    int32_t rounds_planned = 0;                        // Start
    double round_gain = 0.0, round0_gain = 0.0;        // Stop: the round's gain and round 0's; `round` stopped
};

// Settings of a search.
// They apply to the main search and to every search inside the sample-and-place start; defaults in config.hpp (the
// published configuration), filled by infer() from Settings.
struct SearchSettings {
    // outer rounds before the final cleanup pass; 0 = no search (the tree is left as it is)
    int32_t rounds = p2vbme::DEFAULT_MAX_ROUNDS;
    // stop once a whole round (>= 1) accepts no move.
    // Rounds >= 1 are a deterministic function of the tree, so this stop is lossless (the same tree as running all
    // rounds).
    bool stop_zero = p2vbme::DEFAULT_STOP_WHEN_IDLE;
    // > 0: stop once a round (>= 1) improves L by <= stop_tol x (total improvement so far)
    double stop_tol = p2vbme::DEFAULT_EARLY_STOPPING;
    double deadline = 0.0;                                    // absolute wall_sec(): no new round after it; 0 = none
    double time_cap = p2vbme::DEFAULT_TIME_LIMIT;             // the time limit behind the deadline, for reporting only
    int32_t profile_cap = p2vbme::DEFAULT_PROFILE_CAP;        // leaves kept per clade profile
    double exact_check_eta = p2vbme::DEFAULT_EXACT_CHECK_ETA; // exact acceptance check, see optimize_skeleton_on_gpu
    // progress of every search (main and start tree), for printing and diagnostics; empty = none
    std::function<void(const SearchEvent &)> on_event;
};

// What a search did (the main search's values go into the reports).
struct SearchStats {
    int32_t rounds_planned = 0, rounds_run = 0, stop_round = -1;
    // why the rounds ended: max_rounds (all ran), idle (a round changed nothing), early_stopping, time_limit, or none
    // (no rounds: settings.rounds = 0)
    const char *stop_reason = "max_rounds";
};

// ---- Skeleton of a window (shared with the devices' window solve) ----

void build_skeleton_metadata(const int64_t *v_cf, int32_t K, std::vector<int32_t> &c1_cf, std::vector<int32_t> &c2_cf,
                             std::vector<int32_t> &v_map_cf, std::vector<int32_t> &m_lmin_cf,
                             std::vector<int32_t> &next_k_cf, std::vector<int32_t> &parent_u_cf,
                             std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
                             std::vector<int32_t> &adj_u_cf, int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1,
                             int32_t *out_rc2);
void apply_unrooted_spr_host(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_u, int32_t dest_v);
void relabel_from_adj_u(const int32_t *adj_u, int32_t K, int32_t root_edge_u, int32_t root_edge_v, int32_t flip_mode,
                        std::vector<int64_t> &v_cf, std::vector<int32_t> &perm);
// All-pairs topological distances between the K super-leaves of a skeleton.
void compute_skeleton_leaf_dists(const int64_t *v_cf, const int32_t *perm, int32_t K, std::vector<int32_t> &dist_out);

// ---- The window solve (Device::solve_window), shared by the devices ----

// A skeleton of K super-leaves as build_skeleton_metadata describes it: rooted children c1, c2 (2K - 1 nodes), the
// unrooted tree on 2K - 2 nodes rooted at node 2K - 3 (parent_u, sub_size, tin, adj_u) and the root's children.
struct SkeletonMeta {
    std::vector<int32_t> c1, c2, v_map, m_lmin, next_k, parent_u, sub_size, tin, adj_u;
    int32_t other_rc = -1, rc0 = -1, rc1 = -1, rc2 = -1;
};

// What a device computes for a window solve.
class WindowKernels {
  public:
    virtual ~WindowKernels() = default;
    // The K x K distances between the window's profiles (CSR), kept for the window.
    virtual void distances(int32_t K, const int32_t *prof_offsets, const int32_t *prof_leaves,
                           const float *prof_weights) = 0;
    // The skeleton's average distances between all its node pairs (perm: super-leaf of each skeleton leaf), kept for
    // best_moves; returns its BME length.
    virtual double score(const SkeletonMeta &s, const std::vector<int32_t> &perm) = 0;
    // For rows i = 1 .. K - 2 of the vector: the best regraft edge of the subtree pruned at row i and the BME change of
    // that move (best_delta[i], best_j[i]), from the last score.
    virtual void best_moves(const SkeletonMeta &s, std::vector<double> &best_delta, std::vector<int32_t> &best_j) = 0;
    // The exact check: the distances of the pairs (pair_a[k], pair_b[k]) on complete profiles (CSR).
    virtual void exact_distances(const std::vector<int32_t> &pair_a, const std::vector<int32_t> &pair_b,
                                 const int32_t *offsets, const int32_t *leaves, const float *weights,
                                 std::vector<double> &out) = 0;
    // The K x K distances of distances(), as doubles, for the exact check's uncapped pairs (a device may have
    // overwritten the pairs exact_distances refined).
    virtual void read_distances(std::vector<double> &D) = 0;
};

// The window solve of Device::solve_window, with the device's kernels: up to max_iters sweeps of batched moves
// (Algorithm S1), then the exact check when it applies.
double solve_window_host(int32_t K, int32_t max_iters, int32_t max_batch_cap, double exact_check_eta,
                         const int32_t *prof_offsets, const int32_t *prof_leaves, const float *prof_weights,
                         std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                         const ExactProfileBuilder *exact_builder, WindowKernels &kernels);

// The CPU device's window kernels on their own, one thread (tests): alignment rows read in place.
std::unique_ptr<WindowKernels> make_cpu_window_kernels(const uint64_t *rows, int32_t n_words,
                                                       const SubstitutionModel &model, const EigenSettings &eigen,
                                                       bool double_precision);

// ---- Tree summaries and windows ----

void compute_tree_summaries_blind(const int32_t *c1_r, const int32_t *c2_r, int32_t N, std::vector<int32_t> &ord_out,
                                  std::vector<int32_t> &parent_r_out, std::vector<int32_t> &leaf_cnt_out,
                                  std::vector<int32_t> &node_rep0_out);
void extract_active_skeleton(int32_t root_r, const int32_t *c1_r, const int32_t *c2_r, int32_t N,
                             int32_t condense_thresh, const std::vector<uint8_t> &node_has_active,
                             const std::vector<int32_t> &leaf_cnt, std::vector<int32_t> &super_roots,
                             std::vector<int32_t> &split_internals, std::vector<int32_t> &skel_left,
                             std::vector<int32_t> &skel_right, int32_t *out_root_skel_id,
                             int32_t max_internals = WAVE_WINDOW_MAX_INTERNALS,
                             const uint8_t *stop_at_comp_root = nullptr);
void build_component_waves(int32_t N, const std::vector<int32_t> &ord, const std::vector<int32_t> &curr_c1,
                           const std::vector<int32_t> &curr_c2, int32_t max_cap, std::vector<uint8_t> &is_comp_root,
                           std::vector<std::vector<int32_t>> &level_waves);

// ---- The search ----

// Improves the rooted tree (curr_c1, curr_c2) in place; returns the BME change.
// The devices hold the alignment of this tree's leaves.
// level: 0 = the main search, k = a search inside the sample-and-place start, k levels down.
double run_minibatch_bme(int32_t N, Devices &devices, std::vector<int32_t> &curr_c1, std::vector<int32_t> &curr_c2,
                         const SearchSettings &settings, int32_t level = 0, SearchStats *stats = nullptr);

} // namespace p2vbme

#endif // P2VBME_SEARCH_HPP
