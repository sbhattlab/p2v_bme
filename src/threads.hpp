/*
 * threads.hpp
 *
 * The number of OpenMP threads for a loop over independent items.
 */
#ifndef VECME_THREADS_HPP
#define VECME_THREADS_HPP

#include <algorithm>
#include <cstdint>

namespace vecme {

// Chooses the threads for a loop over `items` independent items: (threads, items, per_thread) -> at most `threads`,
// and at least `per_thread` items each (a small loop does not pay for a large team), but at least one.
inline int threads_for(int threads, int64_t items, int64_t per_thread) {
    return (int)std::max<int64_t>(1, std::min<int64_t>(threads, items / per_thread));
}

} // namespace vecme

#endif // VECME_THREADS_HPP
