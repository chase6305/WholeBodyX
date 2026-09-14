#pragma once

#include "wholebodyx/control.hpp"

namespace wholebodyx {

// A bijection over the same fixed-base scalar joints. No unit/sign/frame changes.
class JointOrder {
 public:
  JointOrder(std::vector<std::string> model_names, std::vector<std::string> backend_names);
  const std::vector<std::string>& model_names() const { return model_names_; }
  const std::vector<std::string>& backend_names() const { return backend_names_; }
  Vector to_backend(const Vector&) const;
  Vector to_model(const Vector&) const;
  Matrix columns_to_backend(const Matrix&) const;
  Matrix columns_to_model(const Matrix&) const;
  RobotState state_to_backend(const RobotState&) const;
  JointLimits limits_to_backend(const JointLimits&) const;
  LinearTask task_to_backend(const LinearTask&) const;
  VelocityCommand command_to_model(const VelocityCommand&) const;
  JointPlan plan_to_model(const JointPlan&) const;

 private:
  std::vector<std::string> model_names_, backend_names_;
  std::vector<Eigen::Index> forward_, inverse_;
};

// The wrapped instance must already use backend-order limits and settings.
// QPResult.x keeps the underlying solver's decision-variable order.
class ReorderedController final : public Controller {
 public:
  ReorderedController(std::unique_ptr<Controller>, JointOrder);
  void reset() override;
  ControlResult step(const RobotState&, const std::vector<LinearTask>&, double dt) override;

 private:
  std::unique_ptr<Controller> controller_;
  JointOrder order_;
};

class ReorderedPlanner final : public Planner {
 public:
  ReorderedPlanner(std::unique_ptr<Planner>, JointOrder);
  void reset() override;
  PlanResult solve(const RobotState&, const Vector& goal) override;

 private:
  std::unique_ptr<Planner> planner_;
  JointOrder order_;
};
}  // namespace wholebodyx
