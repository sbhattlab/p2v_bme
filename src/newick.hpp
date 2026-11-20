/*
 * newick.hpp
 *
 * Newick trees: start trees in, unrooted trees out.
 */
#ifndef P2VBME_NEWICK_HPP
#define P2VBME_NEWICK_HPP

#include <string>
#include <vector>

#include "core.hpp"

namespace p2vbme {

// Reads a binary Newick tree whose leaf names are exactly the alignment's names; a trifurcation at the root (unrooted
// tree) is accepted.
// Branch lengths are ignored.
// Throws std::runtime_error.
Tree read_newick(const std::string &path, const std::vector<std::string> &names);

// Writes the tree unrooted: the root's two child edges become one edge with their summed length.
// Branch lengths (when present) are the shortest decimals that read back to the same value of their type (float, or
// double with tree.double_precision_lengths); negative lengths are written as 0 unless neg_as_zero is false.
// Names that need it are single-quoted.
// Throws std::runtime_error.
void write_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree,
                           bool neg_as_zero = true);

} // namespace p2vbme

#endif // P2VBME_NEWICK_HPP
