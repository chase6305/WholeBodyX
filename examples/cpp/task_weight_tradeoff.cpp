#include "wholebodyx/control.hpp"
#include <cmath>
#include <iostream>

int main() {
  using namespace wholebodyx;
  JointLimits limits{Vector::Constant(1, -2), Vector::Constant(1, 2), Vector::Constant(1, 2), Vector()};
  KinematicWBC controller(limits, 0.1);
  RobotState state{Vector::Zero(1), Vector::Zero(1), 0};
  for (double weight : {0.1, 1.0, 10.0, 100.0}) {
    // Two conflicting objectives for the same joint; finite weights trade off.
    LinearTask positive{Matrix::Identity(1, 1), Vector::Ones(1), Vector::Constant(1, weight)};
    LinearTask negative{Matrix::Identity(1, 1), -Vector::Ones(1), Vector::Ones(1)};
    const auto result = controller.step(state, {positive, negative}, 0.02);
    if (!result.command) { std::cerr << result.solver.detail << '\n'; return 1; }
    const double expected = (weight - 1) / (weight + 1 + 0.1);
    const double velocity = result.command->velocity[0];
    std::cout << "positive_weight=" << weight << " velocity=" << velocity << " analytic=" << expected << '\n';
    if (std::abs(velocity - expected) > 1e-6) return 1;
  }
  return 0;
}
