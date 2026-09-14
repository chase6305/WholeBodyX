#include "wholebodyx/control.hpp"
#include <cmath>
#include <iostream>

int main() {
  using namespace wholebodyx;
  JointLimits limits{Vector::Constant(1, -1), Vector::Ones(1), Vector::Constant(1, 0.5), Vector()};
  KinematicWBC controller(limits);
  RobotState state{Vector::Constant(1, 0.9), Vector::Zero(1), 0};
  for (int k = 0; k < 50; ++k) {
    LinearTask task{Matrix::Identity(1, 1), 4 * (Vector::Constant(1, 2) - state.q), Vector::Ones(1)};
    const auto result = controller.step(state, {task}, 0.02);
    if (!result.command) { std::cerr << result.solver.detail << '\n'; return 1; }
    state.v = result.command->velocity;
    state.q += 0.02 * state.v;
    state.stamp += 0.02;
    if (std::abs(state.q[0]) > 1 + 1e-6 || std::abs(state.v[0]) > 0.5 + 1e-6) return 1;
  }
  // Inject an inconsistent measurement separately; do not execute a failed result.
  const auto fault = controller.step({Vector::Constant(1, 2), Vector::Zero(1), state.stamp}, {}, 0.02);
  if (fault.command || fault.solver.status != "infeasible") return 1;
  controller.reset();
  LinearTask return_home{Matrix::Identity(1, 1), -state.q, Vector::Ones(1)};
  const auto recovered = controller.step(state, {return_home}, 0.02);
  if (!recovered.command || recovered.command->velocity[0] >= 0) return 1;
  std::cout << "bounded_position=" << state.q[0] << " fault=" << fault.solver.status
            << " recovered_velocity=" << recovered.command->velocity[0] << '\n';
  return std::abs(state.q[0] - 1) < 1e-6 ? 0 : 1;
}
