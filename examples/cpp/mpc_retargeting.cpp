#include "wholebodyx/control.hpp"
#include <algorithm>
#include <iostream>

int main() {
  using namespace wholebodyx;
  constexpr double dt = 0.02;
  JointLimits limits{Vector::Constant(2, -2), Vector::Constant(2, 2), Vector::Ones(2), Vector::Constant(2, 4)};
  RobotState state{Vector::Zero(2), Vector::Zero(2), 0};
  Vector goal = Vector::Constant(2, 0.5);
  JointMPC planner(limits, 15, 0.1);
  KinematicWBC controller(limits);
  std::optional<JointPlan> plan;
  int replans = 0, goal_changes = 0;
  double max_violation = 0;
  for (int k = 0; k < 400; ++k) {
    // Off the periodic replan grid: changes must trigger an immediate solve.
    const bool changed = k == 103 || k == 207;
    if (changed) { goal.setConstant(k == 103 ? -0.4 : 0.25); ++goal_changes; }
    if (k % 5 == 0 || changed) {
      const auto result = planner.solve(state, goal);
      if (!result.plan) { std::cerr << result.solver.detail << '\n'; return 1; }
      plan = result.plan;
      ++replans;
    }
    const auto reference = plan->sample(state.stamp);
    LinearTask task{Matrix::Identity(2, 2), reference.second + 4 * (reference.first - state.q), Vector::Ones(2)};
    const auto result = controller.step(state, {task}, dt);
    if (!result.command) { std::cerr << result.solver.detail << '\n'; return 1; }
    const auto bounds = limits.velocity_bounds(state, dt);
    const Vector velocity = result.command->velocity;
    if ((velocity.array() < bounds.first.array() - 1e-6).any() ||
        (velocity.array() > bounds.second.array() + 1e-6).any()) return 1;
    max_violation = std::max(max_violation, result.solver.violation);
    state.v = velocity;
    state.q += dt * velocity;
    state.stamp += dt;
  }
  const double error = (state.q - goal).norm();
  std::cout << "goal_changes=" << goal_changes << " replans=" << replans
            << " final_joint_error=" << error << " max_violation=" << max_violation << '\n';
  return error < 1e-3 && goal_changes == 2 && replans == 82 ? 0 : 1;
}
