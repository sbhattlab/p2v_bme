/*
 * newick.hpp
 *
 * Newick trees: start trees in, unrooted trees out.
 */
#ifndef VECME_NEWICK_HPP
#define VECME_NEWICK_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core.hpp"

namespace vecme {

// Reads a binary Newick tree whose leaf names are exactly the alignment's names; a trifurcation at the root (unrooted
// tree) is accepted.
// Branch lengths are ignored.
// Throws std::runtime_error.
Tree read_newick(const std::string &path, const std::vector<std::string> &names);

// Reads a binary Newick tree with its own leaf names: path -> the tree; names receives the leaf names in the order they
// appear (leaf i of the tree is names[i]).
// Branch lengths, if the tree has them, are kept as double values (double_precision_lengths).
// Throws std::runtime_error.
Tree read_newick_with_names(const std::string &path, std::vector<std::string> &names);

// Reads the binary Newick trees of a file, one after another (each ends with ';'), all on the same leaves: calls fn on
// each, in order, and returns their number; names receives the first tree's leaf names (as read_newick_with_names),
// which the other trees must have.
// Branch lengths are ignored.
// Throws std::runtime_error.
int32_t read_newick_trees(const std::string &path, std::vector<std::string> &names,
                          const std::function<void(const Tree &)> &fn);

// Writes the tree unrooted: the root's two child edges become one edge with their summed length.
// Branch lengths (when present) are the shortest decimals that read back to the same value of their type (float, or
// double with tree.double_precision_lengths); negative lengths are written as 0 unless neg_as_zero is false.
// Names that need it are single-quoted.
// labels (optional, one per node): written after the closing parenthesis of each internal node that is not empty, as
// the bootstrap's supports ("97.33/85"); the root and its child that becomes the trifurcation have no parenthesis of
// their own, so the merged root edge carries the other child's label.
// Throws std::runtime_error.
void write_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree,
                           bool neg_as_zero = true, const std::vector<std::string> &labels = {});

// Appends the tree, unrooted, to path as one more line (write_newick_unrooted without labels), e.g. a bootstrap
// replicate's tree to <prefix>.boottrees.
// Throws std::runtime_error.
void append_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree);

} // namespace vecme

#endif // VECME_NEWICK_HPP
