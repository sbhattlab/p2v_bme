/*
 * io.hpp
 *
 * Input and output of the package: aligned DNA FASTA in (packed as in the .bin bundles: 5 bitplanes), Newick
 * start trees in, unrooted Newick out.
 */
#ifndef P2VBME_IO_HPP
#define P2VBME_IO_HPP

#include <string>
#include <vector>

#include "inference.hpp"

namespace p2vbme {

/* Reads an aligned DNA FASTA file; all sequences must have the same length. Throws std::runtime_error. */
Alignment read_fasta(const std::string &path);

/* Reads a .maple alignment (MAPLE / CMAPLE): the reference sequence first ('>' header and sequence lines), then
 * each sequence as '>' name and its differences from the reference, one per line: "<character> <position>
 * [<length>]" (positions from 1; length 1 if omitted). A, C, G, T (or U) set the base; anything else (n, -,
 * ambiguity codes) marks the sites missing. The reference itself is not a sequence of the alignment. Throws
 * std::runtime_error. */
Alignment read_maple(const std::string &path);

/* read_maple for a path ending in .maple (any case), read_fasta otherwise. */
Alignment read_alignment(const std::string &path);

/* Identical sequences: copies with the same bases and the same missing sites. */
struct Deduplication {
    std::vector<int32_t> kept;    /* the sequences kept, as positions in the input, in input order */
    std::vector<int32_t> copy_of; /* per input sequence: -1 if kept, else the position in `kept` of its first copy */
    int32_t n_removed() const { return (int32_t)(copy_of.size() - kept.size()); }
};

/* Removes the third and later copies of each group of identical sequences from the alignment, in place (the first
 * two stay, as IQ-TREE keeps two, so the group still gets a branch and a support); dd records what was removed. */
void deduplicate(Alignment &aln, Deduplication &dd);

/* The tree on all input sequences from a tree on the kept ones: each removed copy is joined to the first copy of
 * its group by zero-length branches (the first copy's branch length goes above them). */
Tree reattach(const Tree &tree, const Deduplication &dd);

/* A tree on all input sequences restricted to the kept ones (a given start tree with dedup on). */
Tree prune(const Tree &tree, const Deduplication &dd);

/* Reads a binary Newick tree whose leaf names are exactly the alignment's names; a trifurcation at the
 * root (unrooted tree) is accepted. Branch lengths are ignored. Throws std::runtime_error. */
Tree read_newick(const std::string &path, const std::vector<std::string> &names);

/* Writes the tree unrooted: the root's two child edges become one edge with their summed
 * length. Branch lengths (when present) are the shortest decimals that read back to the same value of their type
 * (float, or double with tree.double_precision_lengths); negative
 * lengths are written as 0 unless neg_as_zero is false. Names that need it are single-quoted. Throws
 * std::runtime_error. */
void write_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree,
                           bool neg_as_zero = true);

} // namespace p2vbme

#endif // P2VBME_IO_HPP
