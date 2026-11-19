/*
 * branch_lengths.hpp
 *
 * O(N) balanced minimum evolution branch lengths (Pauplin 2000; FastME, Desper & Gascuel 2002) by exact clade
 * condensation: capped dyadic inside and outside profiles on the host, the edge-length pass on the devices.
 */
#ifndef P2VBME_BRANCH_LENGTHS_HPP
#define P2VBME_BRANCH_LENGTHS_HPP

#include <stdint.h>

#include <vector>

#include "core.hpp"
#include "device.hpp"

namespace p2vbme {

// BME lengths of the 2N - 2 edges above nodes 0..2N-3 of the rooted tree (the root's right child gets 0 and its left
// child the merged root edge); profiles keep at most `cap` leaves.
// Returns their sum.
// The lengths are computed in single precision (float values) or, with double_precision (the CPU device's --precision
// double), in double.
double compute_bme_branch_lengths(const Tree &tree, Devices &devices, std::vector<double> &h_branch_lengths,
                                  int32_t cap, double *out_profile_sec = nullptr, double *out_kernel_sec = nullptr,
                                  bool double_precision = false);

} // namespace p2vbme

#endif // P2VBME_BRANCH_LENGTHS_HPP
