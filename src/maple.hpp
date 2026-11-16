/*
 * maple.hpp
 *
 * .maple alignment input (MAPLE / CMAPLE): a reference sequence, then each sequence's differences from it.
 */
#ifndef P2VBME_MAPLE_HPP
#define P2VBME_MAPLE_HPP

#include <string>

#include "inference.hpp"

namespace p2vbme {

/* Reads a .maple alignment (MAPLE / CMAPLE): the reference sequence first ('>' header and sequence lines), then
 * each sequence as '>' name and its differences from the reference, one per line: "<character> <position>
 * [<length>]" (positions from 1; length 1 if omitted). A, C, G, T (or U) set the base; anything else (n, -,
 * ambiguity codes) marks the sites missing. The reference itself is not a sequence of the alignment. Throws
 * std::runtime_error. */
Alignment read_maple(const std::string &path);

} // namespace p2vbme

#endif // P2VBME_MAPLE_HPP
