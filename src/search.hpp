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

// The tables of a skeleton of K pieces, decoded from its vector by to_skeleton_tables: rooted children c1, c2 (2K - 1
// nodes; phylo2vec's ancestry, children swapped), the unrooted tree on 2K - 2 nodes rooted at node 2K - 3 (parent_u,
// sub_size, tin, adj_u) and the root's children.
// Numbered as the vector: tip a stands for piece perm[a], internal nodes in decoding order.
struct SkeletonTables {
    std::vector<int32_t> c1, c2, v_map, parent_u, sub_size, tin, adj_u;
    int32_t rc0 = -1;
    int32_t rc1 = -1;
    int32_t rc2 = -1;
};

// Decodes a skeleton's ordered vector into its tables: v -> sk.
// v has K - 1 entries.
void to_skeleton_tables(const std::vector<int64_t> &v, SkeletonTables &sk);
// Moves the subtree at prune_node, with the node it hangs from (anchor_node), onto the edge between dest_a and dest_b:
// adj_u -> adj_u (see search.cpp for a drawing).
void apply_unrooted_spr(int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_a, int32_t dest_b);
// Encodes the unrooted skeleton back into an ordered vector: (sk.adj_u, root edge, flip) -> (v, perm).
// It roots sk.adj_u on the edge between root_c1 and root_c2, queue-shuffles it (labelling flip) and composes perm
// with the new tip labels.
// Reads only sk.adj_u, and K from the size of sk.v_map: after a batch of moves, the other fields of sk still describe
// the tree before it.
// The root edge and flip change only which subtrees become rows, i.e. which moves the next sweep tries, never
// the BME length.
void from_skeleton_tables(std::vector<int64_t> &v, const SkeletonTables &sk, int32_t root_c1, int32_t root_c2,
                          bool flip, std::vector<int32_t> &perm);

// ---- The window solve, shared by the devices (each gives its kernels: Device::window_kernels) ----

// What a device computes for a window solve.
class WindowKernels {
  public:
    virtual ~WindowKernels() = default;
    // Computes the distances between the window's pieces, kept for the window: profiles (CSR) -> K x K distances.
    virtual void distances(int32_t K, const int32_t *profile_offsets, const int32_t *profile_leaves,
                           const float *profile_weights) = 0;
    // Scores a skeleton: (s, perm) -> its BME length (Pauplin's formula, sum of 2^(1 - tau_ij) d_ij over the piece
    // pairs).
    // Keeps the average distances between all its node pairs for best_moves (perm: the piece of each skeleton leaf).
    virtual double score(const SkeletonTables &s, const std::vector<int32_t> &perm) = 0;
    // Finds each row's best move, from the last score: rows i = 1 .. K - 2 -> (best_delta[i], best_j[i]).
    // best_j[i]: the best regraft edge of the subtree pruned at row i; best_delta[i]: the BME change of that move.
    virtual void best_moves(const SkeletonTables &s, std::vector<double> &best_delta, std::vector<int32_t> &best_j) = 0;
};

// Solves a window with a device's kernels: shortens its skeleton's BME length, (v, perm) -> (v, perm) and the BME
// change.
// A window is part of the tree cut into K pieces.
// Each piece is summarised as a weighted leaf profile (CSR: profile_offsets[K + 1], profile_leaves,
// profile_weights).
// Its skeleton, the tree connecting the pieces, is the Phylo2Vec vector v over the leaf permutation perm (phylo2vec's
// label mapping: perm[a] is the piece of tip a).
// Runs up to max_iters sweeps of batched moves on the skeleton (Algorithm S1, batch cap max_batch_cap) to shorten its
// BME length.
// Updates v and perm and returns the BME change (0 and the input tree if nothing improved).
// The caller writes the new skeleton back into the full tree.
double solve_window(int32_t K, int32_t max_iters, int32_t max_batch_cap, const int32_t *profile_offsets,
                    const int32_t *profile_leaves, const float *profile_weights, std::vector<int64_t> &v,
                    std::vector<int32_t> &perm, WindowKernels &kernels);

// Creates the CPU device's window kernels on their own, one thread (tests): alignment rows read in place.
std::unique_ptr<WindowKernels> make_cpu_window_kernels(const uint64_t *rows, int32_t n_words,
                                                       const SubstitutionModel &model, bool double_precision);

// ---- Tree summaries and windows ----

// What the search needs about each node of the current tree, besides its two children (c1, c2).
// Nodes as in Tree: leaves 0..N-1, internal nodes N..2N-2, root 2N-2.
struct TreeSummary {
    std::vector<int32_t> order;      // the N - 1 internal nodes, BFS from the root
    std::vector<int32_t> parent;     // parent of each node, -1 for the root
    std::vector<int32_t> leaf_count; // leaves under each node, 1 for a leaf
    std::vector<int32_t> stand_in;   // the leaf that takes the node's weight when a capped profile collapses it
};

// Rebuilds the summary of the tree in full: tree -> summary.
// stand_in is each node's closest leaf (the c1 side on ties).
void compute_tree_summary(const Tree &tree, TreeSummary &out);
// A window's skeleton: its K pieces (the clade pieces left to right, then the outgroup if any) and the tree connecting
// them, as an ordered Phylo2Vec vector.
struct Skeleton {
    std::vector<int32_t> piece_roots; // the tree node at the top of each clade piece (a leaf, or a clade not opened)
    std::vector<int32_t> opened;      // the tree nodes the window opened, children before parents (its root last)
    std::vector<int64_t> v;           // the ordered vector (K - 1 entries; empty below WINDOW_MIN_K pieces)
    std::vector<int32_t> perm;        // its label mapping: tip a of the vector stands for piece perm[a]
};
// Extracts the window under window_root: (tree, window_root) -> Skeleton.
// It opens internal nodes from window_root, largest clades first, up to WINDOW_MAX_INTERNALS, never a node marked in
// stop_at_comp_root; the clades below them are the pieces.
// with_outgroup: one more tip (the rest of the tree), joined to window_root under a new root.
// v and perm: the queue shuffle of the skeleton's tree.
Skeleton extract_skeleton(int32_t window_root, const Tree &tree, const TreeSummary &summary,
                          const uint8_t *stop_at_comp_root, bool with_outgroup);
// Cuts the tree into components and schedules them as waves of windows: tree -> (is_comp_root, waves) (see search.cpp).
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
