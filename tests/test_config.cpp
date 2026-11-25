// Tests of src/config.hpp: the constants written out by hand from others.
//   - NN_P_NO_OVERLAP is NO_OVERLAP_DISTANCE seen as a JC69 p-distance.
#include <gtest/gtest.h>

#include <cmath>

#include "config.hpp"

// NN_P_NO_OVERLAP is the p at which JC69 reaches NO_OVERLAP_DISTANCE, p = 3/4 (1 - e^(-4 d / 3)), to its 4 decimals
TEST(Config, NnPNoOverlapIsTheJc69PAtNoOverlapDistance) {
    const double p = 0.75 * (1.0 - std::exp(-4.0 / 3.0 * vecme::NO_OVERLAP_DISTANCE));
    EXPECT_NEAR(vecme::NN_P_NO_OVERLAP, p, 5e-5);
}
