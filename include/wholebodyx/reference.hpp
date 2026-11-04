#pragma once

#include "wholebodyx/control.hpp"

namespace wholebodyx {

struct ReferenceUpdate {
  std::optional<Vector> position, velocity;
  std::string reason;
  std::optional<QPResult> solver;
  std::optional<double> solve_ms, tracking_error;
  bool success() const { return position.has_value() && velocity.has_value(); }
  bool replanned() const { return success() && solve_ms.has_value(); }
};

// Synchronous, state-clock scheduling and acceptance of joint-integrator plans.
// The caller owns the Planner for this manager's lifetime and serializes calls.
// A failed refresh invalidates the previous plan. Clock rollback is latched
// until both the planner and manager have successfully reset.
class MPCReferenceManager {
 public:
  MPCReferenceManager(Planner&, JointLimits, double replan_period = .1,
                      double tracking_tolerance = .03);
  void reset();
  void invalidate(std::string reason = "forced");
  void discard_outdated(double stamp, const Vector& goal);
  ReferenceUpdate update(const RobotState&, const Vector& goal, double control_dt);
  const std::optional<JointPlan>& plan() const { return plan_; }

 private:
  void clear();
  JointPlan accept(const JointPlan&, const RobotState&, double control_dt) const;
  Planner& planner_;
  JointLimits limits_;
  double replan_period_, tracking_tolerance_;
  std::optional<JointPlan> plan_;
  Vector goal_;
  std::optional<double> next_replan_, last_stamp_;
  bool clock_invalid_ = false;
  std::string pending_reason_ = "initial";
};

}  // namespace wholebodyx
