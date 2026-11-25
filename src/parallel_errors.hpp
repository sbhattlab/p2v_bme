/*
 * parallel_errors.hpp
 *
 * Exceptions thrown inside an OpenMP parallel region, carried out of it (the device steps run one host thread per
 * device: the search's windows, the start tree's nearest-sample step, the branch lengths).
 */
#ifndef VECME_PARALLEL_ERRORS_HPP
#define VECME_PARALLEL_ERRORS_HPP

#include <atomic>
#include <exception>
#include <mutex>

namespace vecme {

// The first exception thrown by the device steps inside an OpenMP region (one host thread per device).
// It is kept and rethrown after the region, because an exception must not leave a parallel region.
// Once one is caught, the remaining work items are skipped.
class ParallelErrors {
  public:
    bool failed() const { return failed_.load(std::memory_order_relaxed); }
    void capture() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_)
            first_ = std::current_exception();
        failed_ = true;
    }
    void rethrow() const {
        if (first_)
            std::rethrow_exception(first_);
    }

  private:
    std::exception_ptr first_;
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
};

} // namespace vecme

#endif // VECME_PARALLEL_ERRORS_HPP
