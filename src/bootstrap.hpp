/*
 * bootstrap.hpp
 *
 * The standard bootstrap (Felsenstein 1985): replicate alignments of the searched alignment's columns, drawn with
 * replacement; a tree inferred from each as the main tree was; and the supports of the main tree's branches over the
 * replicate trees (TBE and exact, support.hpp).
 * Replicates keep the main run's model parameters (the GTR+GC fit, the base frequencies) instead of refitting them, as
 * the paper did, and need no branch lengths.
 *
 * Seeds: replicate r (1..B) has the seed replicate_seed(--seed, r), which seeds its search; its columns come from a
 * stream of their own, derive_seed(that seed, SALT_BOOTSTRAP_COLUMNS).
 *
 * Reference
 *   Felsenstein, J. (1985). Confidence limits on phylogenies: an approach using the bootstrap. Evolution 39, 783-791.
 *     doi:10.1111/j.1558-5646.1985.tb00420.x
 */
#ifndef P2VBME_BOOTSTRAP_HPP
#define P2VBME_BOOTSTRAP_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core.hpp"
#include "device.hpp"
#include "inference.hpp"

namespace p2vbme {

// Computes the seed of bootstrap replicate r (1..B): (run seed, r) -> the replicate's seed.
uint64_t replicate_seed(uint64_t seed, int32_t replicate);

// Draws a replicate's columns: (n_sites, seed) -> n_sites column indices, each uniform in 0..n_sites-1.
std::vector<int32_t> draw_columns(int32_t n_sites, uint64_t seed);

// Builds the alignment of the given columns: (aln, columns) -> the same sequences and names on columns.size() sites,
// site j being column columns[j] of aln.
// Throws std::invalid_argument for a column outside aln.
Alignment resample_columns(const Alignment &aln, const std::vector<int32_t> &columns);

// The bootstrap's outcome.
struct BootstrapResult {
    int32_t n_replicates = 0;
    std::vector<double> tbe, exact;          // per node of the reference tree, in percent (NO_SUPPORT: no branch)
    double mean_tbe = 0.0, mean_exact = 0.0; // over the reference tree's internal branches
    double time = 0.0;                       // seconds
};

// Receives bootstrap replicate r's search result (its tree is on the alignment's sequences, without branch lengths).
using ReplicateCallback = std::function<void(int32_t replicate, const Result &result)>;

// Runs bootstrap replicate r (1..) alone: (aln, settings, devices, r) -> its search's result, as bootstrap() gets it.
// aln and settings as bootstrap()'s; the tree has no branch lengths.
// Throws std::invalid_argument if replicate < 1, and what infer() throws.
Result bootstrap_replicate(const Alignment &aln, const Settings &settings, Devices &devices, int32_t replicate,
                           const LogCallback &log = nullptr);

// Runs bootstrap replicates first..last (1-based, inclusive) one after another, as bootstrap() runs them:
// on_replicate receives each result, in order; log receives one line per replicate.
// Replicate r depends only on the settings (seed) and r, so ranges run as separate jobs give the trees of one run.
// Throws std::invalid_argument if first < 1 or last < first, and what infer() throws.
void bootstrap_replicates(const Alignment &aln, const Settings &settings, Devices &devices, int32_t first, int32_t last,
                          const ReplicateCallback &on_replicate, const LogCallback &log = nullptr);

// Runs the bootstrap: (aln, settings, devices, reference, B) -> the supports of the reference tree's branches.
// aln: the alignment the main tree was searched on (identical sequences set aside); settings: the main search's, with
// the model's parameters as it used them; reference: the main tree, on aln's sequences.
// Each replicate (bootstrap_replicate) runs infer() with settings except its seed, no branch lengths, and the
// sample-and-place start when the main search started from a given tree.
// log receives one line per replicate; on_replicate (optional) each replicate's result, e.g. to save its tree.
// Throws std::invalid_argument if n_replicates < 1, and what infer() throws.
BootstrapResult bootstrap(const Alignment &aln, const Settings &settings, Devices &devices, const Tree &reference,
                          int32_t n_replicates, const LogCallback &log = nullptr,
                          const ReplicateCallback &on_replicate = nullptr);

// Writes the supports as Newick labels of the tree restore_copies builds: (supports on the searched sequences' tree,
// its node map) -> one label per node of the full tree.
// A branch's label is "TBE/exact" (format_support); leaves and nodes without a branch get none, and the nodes that join
// identical copies "100/100".
// The root's two children are one branch: if one has no label, it takes the other's.
std::vector<std::string> support_labels(const BootstrapResult &boot, const std::vector<int32_t> &node_map,
                                        const Tree &full_tree);

} // namespace p2vbme

#endif // P2VBME_BOOTSTRAP_HPP
