#pragma once

#include "wholebodyx/control.hpp"
#include <algorithm>
#include <cmath>

namespace wholebodyx::detail {

inline double joint_plan_violation(const JointPlan& plan, const JointLimits& limits, const RobotState& initial) {
  if (!plan.positions.allFinite() || !plan.velocities.allFinite()) return INFINITY;
  double violation = 0.;
  Vector previous = initial.v;
  for (Eigen::Index k = 0; k < plan.velocities.rows(); ++k) {
    const Vector position = plan.positions.row(k + 1).transpose();
    const Vector velocity = plan.velocities.row(k).transpose();
    violation = std::max({violation, (limits.lower - position).maxCoeff(),
                          (position - limits.upper).maxCoeff(),
                          (velocity.cwiseAbs() - limits.velocity).maxCoeff()});
    if (limits.acceleration.size())
      violation = std::max(violation, ((velocity - previous).cwiseAbs() - plan.dt * limits.acceleration).maxCoeff());
    previous = velocity;
  }
  return violation;
}

}  // namespace wholebodyx::detail
