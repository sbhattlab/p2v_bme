/*
 * search.hpp
 *
 * The minibatch BME search on the host: rounds of windows (backbone, leaf-covering waves, multi-scale V-cycle),
 * each window a skeleton of K <= 1,500 pieces summarised by weighted leaf profiles and solved by a Device.
 * Also the Phylo2Vec encoding helpers the windows and the devices share. Internal to the package.
 */
#ifndef P2VBME_SEARCH_HPP
#define P2VBME_SEARCH_HPP

#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "config.hpp"
#include "core.hpp"
#include "device.hpp"
#include "stop_reason.hpp"

namespace p2vbme {

// Progress of a search, reported through SearchSettings::on_event (the core keeps no diagnostic state and prints
// nothing).
struct SearchEvent {
    enum Kind { Start, Stage, Stop } kind = Stage;
    int32_t level = 0;       // 0 = the main search; k >= 1 = a search inside the sample-and-place start, k levels down
    int32_t n_leaves = 0;    // N of this search
    int32_t round = 0;       // Stage: the round (the final cleanup pass has round = rounds run)
    const char *stage = "";  // Stage: backbone, comp_waves, scale, round_end
    int64_t detail = 0;      // Stage: waves (comp_waves), scale cap (scale), -1 (round_end), 0 (backbones)
    double total_dL = 0.0;   // Stage: BME change of this search so far
    const Tree *tree = NULL; // Stage round_end: the current tree
    int32_t n_devices = 0;   // Start
    int32_t rounds_planned = 0;                 // Start
    double round_gain = 0.0, round0_gain = 0.0; // Stop: the round's gain and round 0's; `round` stopped
};

// Settings of a search.
// They apply to the main search and to every search inside the sample-and-place start; defaults in config.hpp (the
// published configuration), filled by infer() from Settings.
struct SearchSettings {
    // rounds before the final cleanup pass; 0 = no search (the tree is left as it is)
    int32_t rounds = p2vbme::DEFAULT_MAX_ROUNDS;
    // stop once a round (>= 1) improves L by <= stop_tol x (total improvement so far).
    // 0 = only when a round changes nothing.
    // Rounds >= 1 are a deterministic function of the tree, so that stop is lossless (the same tree as running all
    // rounds).
    double stop_tol = p2vbme::DEFAULT_TOL;
    double deadline = 0.0;                                    // absolute wall_sec(): no new round after it; 0 = none
    double time_cap = p2vbme::DEFAULT_TIME_LIMIT;             // the time limit behind the deadline, for reporting only
    int32_t profile_cap = p2vbme::DEFAULT_PROFILE_CAP;        // leaves kept per clade profile
    double exact_check_eta = p2vbme::DEFAULT_EXACT_CHECK_ETA; // exact acceptance check, see optimize_skeleton_on_gpu
    // progress of every search (main and start tree), for printing and diagnostics; empty = none
    std::function<void(const SearchEvent &)> on_event;
};

// What a search did (the main search's values go into the reports).
struct SearchStats {
    int32_t rounds_planned = 0;
    int32_t rounds_run = 0;
    StopReason stop_reason = StopReason::MaxRounds; // why the rounds ended
};

// ---- Skeleton of a window (shared with the devices' window solve) ----

void build_skeleton_metadata(const int64_t *v_cf, int32_t K, std::vector<int32_t> &c1_cf, std::vector<int32_t> &c2_cf,
                             std::vector<int32_t> &v_map_cf, std::vector<int32_t> &parent_u_cf,
                             std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
                             std::vector<int32_t> &adj_u_cf, int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1,
                             int32_t *out_rc2);
void apply_unrooted_spr_host(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_u, int32_t dest_v);
void relabel_from_adj_u(const int32_t *adj_u, int32_t K, int32_t root_edge_u, int32_t root_edge_v, int32_t flip_mode,
                        std::vector<int64_t> &v_cf, std::vector<int32_t> &perm);
// All-pairs topological distances between the K pieces of a skeleton.
void compute_skeleton_leaf_dists(const int64_t *v_cf, const int32_t *perm, int32_t K, std::vector<int32_t> &dist_out);

// ---- The window solve (Device::solve_window), shared by the devices ----

// A skeleton of K pieces as build_skeleton_metadata describes it: rooted children c1, c2 (2K - 1 nodes), the
// unrooted tree on 2K - 2 nodes rooted at node 2K - 3 (parent_u, sub_size, tin, adj_u) and the root's children.
struct SkeletonMeta {
    std::vector<int32_t> c1, c2, v_map, parent_u, sub_size, tin, adj_u;
    int32_t other_rc = -1;
    int32_t rc0 = -1;
    int32_t rc1 = -1;
    int32_t rc2 = -1;
};

// What a device computes for a window solve.
class WindowKernels {
  public:
    virtual ~WindowKernels() = default;
    // The K x K distances between the window's profiles (CSR), kept for the window.
    virtual void distances(int32_t K, const int32_t *profile_offsets, const int32_t *profile_leaves,
                           const float *profile_weights) = 0;
    // The skeleton's average distances between all its node pairs (perm: the piece of each skeleton leaf), kept for
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
                         const int32_t *profile_offsets, const int32_t *profile_leaves, const float *profile_weights,
                         std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
                         const ExactProfileBuilder *exact_builder, WindowKernels &kernels);

// The CPU device's window kernels on their own, one thread (tests): alignment rows read in place.
std::unique_ptr<WindowKernels> make_cpu_window_kernels(const uint64_t *rows, int32_t n_words,
                                                       const SubstitutionModel &model, const EigenSettings &eigen,
                                                       bool double_precision);

// ---- Tree summaries and windows ----

// What the search needs about each node of the current tree, besides its two children (c1, c2).
// Nodes as in Tree: leaves 0..N-1, internal nodes N..2N-2, root 2N-2.
struct TreeSummary {
    std::vector<int32_t> order;      // the N - 1 internal nodes, BFS from the root
    std::vector<int32_t> parent;     // parent of each node, -1 for the root
    std::vector<int32_t> leaf_count; // leaves under each node, 1 for a leaf
    std::vector<int32_t> stand_in;   // the leaf that takes the node's weight when a capped profile collapses it
};

// The summary of the tree, rebuilt in full.
// stand_in is each node's closest leaf (the c1 side on ties).
void compute_tree_summary(const Tree &tree, TreeSummary &out);
// The skeleton of a window, in the layout of Tree: tips 0..K-1 (the clade pieces left to right, then the outgroup if
// any), internal nodes K..2K-2, root 2K-2.
struct WindowSkeleton {
    std::vector<int32_t> c1, c2;      // the skeleton's children (0 for its tips)
    std::vector<int32_t> piece_roots; // the tree node at the top of each clade piece (a leaf, or a clade not opened)
    std::vector<int32_t> opened;      // the tree node of skeleton internal node K + s (children before parents)
};
// The window under window_root: it opens internal nodes from window_root, largest clades first, up to
// WINDOW_MAX_INTERNALS, never a node marked in stop_at_comp_root; the clades below them are the pieces.
// with_outgroup: one more tip (the rest of the tree), joined to window_root under a new root.
WindowSkeleton extract_skeleton(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                                const uint8_t *stop_at_comp_root, bool with_outgroup);
void build_component_waves(const Tree &tree, const TreeSummary &summary, int32_t max_cap,
                           std::vector<uint8_t> &is_comp_root, std::vector<std::vector<int32_t>> &waves);

// ---- The search ----

// Improves the topology of the rooted tree in place; returns the BME change.
// Its branch lengths are left as they are.
// The devices hold the alignment of this tree's leaves.
// level: 0 = the main search, k = a search inside the sample-and-place start, k levels down.
double run_minibatch_bme(Tree &tree, Devices &devices, const SearchSettings &settings, int32_t level = 0,
                         SearchStats *stats = nullptr);

} // namespace p2vbme

#endif // P2VBME_SEARCH_HPP
