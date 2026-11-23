/*
 * stop_reason.hpp
 *
 * Why the rounds of a search ended (SearchStats, Result), with its text form (the report).
 */
#ifndef P2VBME_STOP_REASON_HPP
#define P2VBME_STOP_REASON_HPP

namespace p2vbme {

enum class StopReason {
    MaxRounds, // all the planned rounds ran
    Converged, // a round (>= 1) improved L by <= tol x (total improvement so far)
    TimeLimit, // the time limit passed
    None,      // no rounds (max rounds = 0)
};

// Text forms: max_rounds, converged, time_limit, none.
inline const char *stop_reason_name(StopReason reason) {
    switch (reason) {
    case StopReason::Converged:
        return "converged";
    case StopReason::TimeLimit:
        return "time_limit";
    case StopReason::None:
        return "none";
    default:
        return "max_rounds";
    }
}

} // namespace p2vbme

#endif // P2VBME_STOP_REASON_HPP
