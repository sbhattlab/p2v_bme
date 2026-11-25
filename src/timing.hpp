/*
 * timing.hpp
 *
 * The wall clock that times the run's steps and applies the time limit.
 */
#ifndef VECME_TIMING_HPP
#define VECME_TIMING_HPP

#include <chrono>

namespace vecme {

// Monotonic wall clock, seconds (std::chrono::steady_clock).
// Its zero is unspecified, so only differences between readings are meaningful.
inline double wall_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace vecme

#endif // VECME_TIMING_HPP
