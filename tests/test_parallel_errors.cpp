// The exceptions of an OpenMP region carried out of it (parallel_errors.hpp): the first one captured is rethrown, with
// its type, and the later ones are dropped; captures from every thread of a region.
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "parallel_errors.hpp"

using namespace p2vbme;

namespace {

// Captures e as a catch (...) block in a parallel region does.
template <class E> void capture(ParallelErrors &errors, const E &e) {
    try {
        throw e;
    } catch (...) {
        errors.capture();
    }
}

} // namespace

TEST(ParallelErrors, NothingCaptured) {
    const ParallelErrors errors;
    EXPECT_FALSE(errors.failed());
    EXPECT_NO_THROW(errors.rethrow());
}

// The first exception comes out as it was thrown (its type and message); the second is dropped.
TEST(ParallelErrors, RethrowsTheFirstException) {
    ParallelErrors errors;
    capture(errors, std::invalid_argument("first"));
    EXPECT_TRUE(errors.failed());
    capture(errors, std::runtime_error("second"));
    std::string msg;
    try {
        errors.rethrow();
    } catch (const std::invalid_argument &e) {
        msg = e.what();
    }
    EXPECT_EQ(msg, "first");
}

// Every work item of a region throws: one of their exceptions comes out after the region.
TEST(ParallelErrors, CapturesFromEveryThreadOfARegion) {
    const int n_items = 1000;
    ParallelErrors errors;
#pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for (int i = 0; i < n_items; ++i)
        capture(errors, std::runtime_error("item " + std::to_string(i)));
    EXPECT_TRUE(errors.failed());
    std::string msg;
    try {
        errors.rethrow();
    } catch (const std::runtime_error &e) {
        msg = e.what();
    }
    EXPECT_EQ(msg.rfind("item ", 0), 0u) << msg;
}
