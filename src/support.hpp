/*
 * support.hpp
 *
 * The supports of a reference tree's branches from replicate trees on the same leaves (the bootstrap's trees,
 * bootstrap.hpp): the exact support, the share of replicates that contain the branch's split, and the transfer
 * bootstrap expectation (TBE), which also credits splits that reappear with a few leaves moved.
 *
 * TBE (Lemoine et al. 2018): a branch whose split has p leaves on its smaller side, and a replicate tree T*, have the
 * transfer distance delta = the fewest leaves to move to obtain a split of T* (at most p - 1).
 * TBE = 1 - mean over the replicates of delta / (p - 1); branches with p < 2 have no support.
 * delta for every branch against one replicate takes O(K log^2 K) for K leaves (Truszkowski, Gascuel & Swenson): the
 * replicate keeps, for each of its subtrees C, W = |C| - 2 |C n S| in a segment tree over a heavy-path decomposition,
 * and the reference tree's leaves are added to S and removed in "DSU on tree" order, so that each leaf moves O(log K)
 * times; then delta = min(|S| + min W, (K - |S|) - max W).
 * Ported from the paper's code (its p2v_tbe), which computes the same two supports.
 *
 * References
 *   Lemoine, F., Domelevo Entfellner, J.-B., Wilkinson, E., Correia, D., Davila Felipe, M., De Oliveira, T. &
 *     Gascuel, O. (2018). Renewing Felsenstein's phylogenetic bootstrap in the era of big data. Nature 556, 452-456.
 *     doi:10.1038/s41586-018-0043-0
 *   Truszkowski, J., Gascuel, O. & Swenson, K. M. (2019). Rapidly computing the phylogenetic transfer index. bioRxiv
 *     (preprint). doi:10.1101/743948
 */
#ifndef VECME_SUPPORT_HPP
#define VECME_SUPPORT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core.hpp"

namespace vecme {

// The support of a node without a branch to support: a leaf, the root, or a split with fewer than 2 leaves on a side.
constexpr double NO_SUPPORT = -1.0;

// Counts the supports of a reference tree's branches over replicate trees added one at a time.
class SupportCounter {
  public:
    // Prepares the reference tree: reference -> counter with no replicate.
    // Needs at least 4 leaves.
    explicit SupportCounter(const Tree &reference);
    ~SupportCounter();

    // Adds one replicate tree on the same leaves: replicate -> the transfer distance of every branch to it, counted.
    void add(const Tree &replicate);

    int32_t n_replicates() const;

    // Computes the supports in percent: () -> one value per node of the reference tree, for the branch above it.
    // NO_SUPPORT where there is no branch; the root's two children are one branch of the unrooted tree and get the same
    // value.
    std::vector<double> tbe() const;
    std::vector<double> exact() const;

    // Averages the supports over the internal branches of the unrooted reference tree (each counted once).
    double mean_tbe() const;
    double mean_exact() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Formats a support in percent as the paper's code writes it: at most 2 decimals, no trailing zeros ("97.33", "100").
std::string format_support(double percent);

} // namespace vecme

#endif // VECME_SUPPORT_HPP
