/*
 * eigen_dist.cpp
 *
 * Names of the eigenvalue GTR+GC estimator settings (host).
 */
#include "eigen_dist.hpp"

#include <stdexcept>

namespace p2vbme {

const char *estimator_name(GtrEstimator e) {
    switch (e) {
    case GtrEstimator::Wls:
        return "wls";
    case GtrEstimator::Svd:
        return "svd";
    case GtrEstimator::Orig:
        return "orig";
    }
    return "unknown";
}

const char *saturation_name(Saturation s) { return s == Saturation::Pair ? "pair" : "direction"; }

GtrEstimator parse_estimator(const std::string &name) {
    if (name == "wls")
        return GtrEstimator::Wls;
    if (name == "svd")
        return GtrEstimator::Svd;
    if (name == "orig")
        return GtrEstimator::Orig;
    throw std::runtime_error("GTR estimator must be wls, svd or orig, not '" + name + "'");
}

Saturation parse_saturation(const std::string &name) {
    if (name == "direction")
        return Saturation::Direction;
    if (name == "pair")
        return Saturation::Pair;
    throw std::runtime_error("saturation must be direction or pair, not '" + name + "'");
}

} // namespace p2vbme
