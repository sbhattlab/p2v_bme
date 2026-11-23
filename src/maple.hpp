/*
 * maple.hpp
 *
 * .maple alignment input, the format of MAPLE (De Maio et al. 2023) and CMAPLE (Ly-Trong et al. 2024).
 * A reference sequence, then each sequence's differences from it.
 *
 * References
 *   De Maio, N., Kalaghatgi, P., Turakhia, Y., Corbett-Detig, R., Minh, B. Q. & Goldman, N. (2023). Maximum
 *     likelihood pandemic-scale phylogenetics. Nature Genetics 55, 746-752. doi:10.1038/s41588-023-01368-0
 *   Ly-Trong, N., Bielow, C., De Maio, N. & Minh, B. Q. (2024). CMAPLE: efficient phylogenetic inference in the
 *     pandemic era. Molecular Biology and Evolution 41(7), msae134. doi:10.1093/molbev/msae134
 */
#ifndef P2VBME_MAPLE_HPP
#define P2VBME_MAPLE_HPP

#include <string>

#include "core.hpp"

namespace p2vbme {

// Reads a .maple alignment (MAPLE / CMAPLE).
// First: the reference sequence first ('>' header and sequence lines
// Then: each sequence as '>' name and its differences from the reference, one per line: "<character> <position>
// [<length>]" (positions from 1; length 1 if omitted).
// A, C, G, T (or U) set the base; anything else (n, -, * ambiguity codes) marks the sites missing.
// The reference itself is not a sequence of the alignment.
Alignment read_maple(const std::string &path);

} // namespace p2vbme

#endif // P2VBME_MAPLE_HPP
