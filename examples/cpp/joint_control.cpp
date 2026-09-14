#include "wholebodyx/control.hpp"
#include <iostream>

int main() {
  using namespace wholebodyx;
  JointLimits limits{Vector::Constant(6, -2.8), Vector::Constant(6, 2.8),
                     Vector::Ones(6), Vector::Constant(6, 4.0)};
  RobotState state{Vector::Zero(6), Vector::Zero(6), 0};
  const Vector goal = Vector::Constant(6, 0.5);
  JointMPC planner(limits);
  KinematicWBC controller(limits);
  std::optional<JointPlan> plan;
  for (int k = 0; k < 250; ++k) {
    if (k % 5 == 0) {
      auto result = planner.solve(state, goal);
      if (!result.plan) { std::cerr << result.solver.detail << '\n'; return 1; }
      plan = result.plan;
    }
    const auto reference = plan->sample(state.stamp);
    LinearTask task{Matrix::Identity(6, 6), reference.second + 4 * (reference.first - state.q), Vector::Ones(6)};
    auto result = controller.step(state, {task}, 0.02);
    if (!result.command) { std::cerr << result.solver.detail << '\n'; return 1; }
    state.v = result.command->velocity;
    state.q += 0.02 * state.v;
    state.stamp += 0.02;
  }
  const double error = (state.q - goal).norm();
  std::cout << "final_joint_error_rad=" << error << '\n';
  return error < 1e-3 ? 0 : 1;
}
