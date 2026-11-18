/*
 * timing.hpp
 *
 * The wall clock that times the run's steps and applies the time limit.
 */
#ifndef P2VBME_TIMING_HPP
#define P2VBME_TIMING_HPP

#include <time.h>

namespace p2vbme {

// Monotonic wall clock, seconds.
inline double wall_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

} // namespace p2vbme

#endif // P2VBME_TIMING_HPP
