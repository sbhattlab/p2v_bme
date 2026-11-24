/*
 * branch_lengths.hpp
 *
 * O(N) balanced minimum evolution branch lengths (Pauplin 2000; FastME, Desper & Gascuel 2002) by exact clade
 * condensation: capped dyadic inside and outside representatives on the host, the edge-length pass on the devices.
 */
#ifndef P2VBME_BRANCH_LENGTHS_HPP
#define P2VBME_BRANCH_LENGTHS_HPP

#include <cstdint>
#include <vector>

#include "core.hpp"
#include "device.hpp"

namespace p2vbme {

// Computes the BME branch lengths of the rooted tree: tree -> branch_lengths; returns their sum.
// branch_lengths[u]: the length of the edge above node u, for u = 0..2N-3.
// The unrooted tree joins the root's two children by one edge: its length goes above the left child (c1), and the
// right child (c2) gets 0.
// Each set of representatives keeps at most `cap` leaves; with cap >= N it is complete, and the lengths exact up to
// the precision.
// The lengths are computed in single precision (float values) or, with double_precision (the CPU device's --precision
// double), in double.
// representatives_sec_out, kernel_sec_out (if not null): seconds spent on the representatives (host) and on the rest
// (devices, then host).
double compute_bme_branch_lengths(const Tree &tree, Devices &devices, std::vector<double> &branch_lengths, int32_t cap,
                                  double *representatives_sec_out = nullptr, double *kernel_sec_out = nullptr,
                                  bool double_precision = false);

} // namespace p2vbme

#endif // P2VBME_BRANCH_LENGTHS_HPP
