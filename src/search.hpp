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
    double deadline = 0.0;                             // absolute wall_sec(): no new round after it; 0 = none
    double time_cap = p2vbme::DEFAULT_TIME_LIMIT;      // the time limit behind the deadline, for reporting only
    int32_t profile_cap = p2vbme::DEFAULT_PROFILE_CAP; // leaves kept per clade profile
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

// A skeleton of K pieces as to_skeleton_meta describes it: rooted children c1, c2 (2K - 1 nodes), the
// unrooted tree on 2K - 2 nodes rooted at node 2K - 3 (parent_u, sub_size, tin, adj_u) and the root's children.
struct SkeletonMeta {
    std::vector<int32_t> c1, c2, v_map, parent_u, sub_size, tin, adj_u;
    int32_t other_rc = -1;
    int32_t rc0 = -1;
    int32_t rc1 = -1;
    int32_t rc2 = -1;
};

// Decodes the skeleton's vector v (K - 1 entries) into sk.
void to_skeleton_meta(const std::vector<int64_t> &v, SkeletonMeta &sk);
void apply_unrooted_spr_host(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_u, int32_t dest_v);
// Encodes the unrooted skeleton sk.adj_u, rooted on the edge between root_c1 and root_c2, as an ordered vector v
// (queue shuffle, labelling flip_mode) and composes perm with the new tip labels.
// Reads only sk.adj_u, and K from the size of sk.v_map: after a batch of moves, the other fields of sk still describe
// the tree before it.
void from_skeleton_meta(std::vector<int64_t> &v, const SkeletonMeta &sk, int32_t root_c1, int32_t root_c2,
                        int32_t flip_mode, std::vector<int32_t> &perm);

// ---- The window solve, shared by the devices (each gives its kernels: Device::window_kernels) ----

// What a device computes for a window solve.
class WindowKernels {
  public:
    virtual ~WindowKernels() = default;
    // The K x K distances between the window's profiles (CSR), kept for the window.
    virtual void distances(int32_t K, const int32_t *profile_offsets, const int32_t *profile_leaves,
                           const float *profile_weights) = 0;
    // The skeleton's average distances between all its node pairs (perm: the piece of each skeleton leaf), kept for
    // best_moves; returns its BME length (Pauplin's formula, sum of 2^(1 - tau_ij) d_ij over the piece pairs).
    virtual double score(const SkeletonMeta &s, const std::vector<int32_t> &perm) = 0;
    // For rows i = 1 .. K - 2 of the vector: the best regraft edge of the subtree pruned at row i and the BME change of
    // that move (best_delta[i], best_j[i]), from the last score.
    virtual void best_moves(const SkeletonMeta &s, std::vector<double> &best_delta, std::vector<int32_t> &best_j) = 0;
};

// The window solve, with a device's kernels.
// A window is part of the tree cut into K pieces.
// Each piece is summarised as a weighted leaf profile (CSR: profile_offsets[K + 1], profile_leaves,
// profile_weights).
// Its skeleton, the tree connecting the pieces, is the Phylo2Vec vector v over the leaf permutation perm (phylo2vec's
// label mapping: perm[a] is the piece of tip a).
// Runs up to max_iters sweeps of batched moves on the skeleton (Algorithm S1, batch cap max_batch_cap) to shorten its
// BME length.
// Updates v and perm and returns the BME change (0 and the input tree if nothing improved).
// The caller writes the new skeleton back into the full tree.
double solve_window_host(int32_t K, int32_t max_iters, int32_t max_batch_cap, const int32_t *profile_offsets,
                         const int32_t *profile_leaves, const float *profile_weights, std::vector<int64_t> &v,
                         std::vector<int32_t> &perm, WindowKernels &kernels);

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
