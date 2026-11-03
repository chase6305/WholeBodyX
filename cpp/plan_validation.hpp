#pragma once

#include "wholebodyx/control.hpp"
#include <algorithm>
#include <cmath>

namespace wholebodyx::detail {

inline double joint_plan_violation(const JointPlan& plan, const JointLimits& limits,
                                   const RobotState& initial) {
  if (!plan.positions.allFinite() || !plan.velocities.allFinite()) return INFINITY;
  double violation = 0.;
  const bool check_slew = limits.acceleration.size() != 0;
  const auto interval_count = plan.velocities.rows();
  // Column views follow Eigen's storage order and let reductions operate
  // without allocating position/velocity vectors for every interval.
  for (Eigen::Index joint = 0; joint < plan.velocities.cols(); ++joint) {
    const auto positions = plan.positions.col(joint).tail(interval_count);
    const auto velocities = plan.velocities.col(joint);
    violation = std::max({violation, limits.lower[joint] - positions.minCoeff(),
                          positions.maxCoeff() - limits.upper[joint],
                          velocities.cwiseAbs().maxCoeff() - limits.velocity[joint]});
    if (check_slew) {
      double maximum_change = std::abs(velocities[0] - initial.v[joint]);
      if (interval_count > 1) {
        maximum_change = std::max(
            maximum_change,
            (velocities.tail(interval_count - 1) - velocities.head(interval_count - 1))
                .cwiseAbs().maxCoeff());
      }
      violation = std::max(violation, maximum_change - plan.dt * limits.acceleration[joint]);
    }
  }
  return violation;
}

}  // namespace wholebodyx::detail
