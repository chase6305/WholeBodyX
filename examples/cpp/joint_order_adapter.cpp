#include "wholebodyx/joint_order.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace wholebodyx;

int main() {
  try {
    const JointOrder wbc_order({"shoulder", "elbow", "wrist"}, {"wrist", "shoulder", "elbow"});
    const JointOrder mpc_order(wbc_order.model_names(), {"elbow", "wrist", "shoulder"});
    JointLimits limits{(Vector(3) << -1.2, -0.7, -2.1).finished(),
                       (Vector(3) << 1.0, 1.4, 0.8).finished(),
                       (Vector(3) << 0.3, 0.6, 0.9).finished(),
                       (Vector(3) << 2.0, 3.0, 4.0).finished()};
    KinematicWBC baseline_wbc(limits);
    JointMPC baseline_mpc(limits);
    ReorderedController wbc(std::make_unique<KinematicWBC>(wbc_order.limits_to_backend(limits)), wbc_order);
    ReorderedPlanner mpc(std::make_unique<JointMPC>(mpc_order.limits_to_backend(limits)), mpc_order);
    RobotState baseline{(Vector(3) << .1, -.2, .3).finished(), Vector::Zero(3), 0};
    RobotState mapped = baseline;
    const Vector goal = (Vector(3) << .8, .5, -.6).finished();
    double difference = 0;
    for (int k = 0; k < 250; ++k) {
      // Replan every step here to exercise complete trajectory conversion.
      auto reference = baseline_mpc.solve(baseline, goal);
      auto converted = mpc.solve(mapped, goal);
      if (!reference.plan || !converted.plan) throw std::runtime_error("MPC failed");
      difference = std::max(difference, (reference.plan->positions - converted.plan->positions).cwiseAbs().maxCoeff());
      auto a = reference.plan->sample(baseline.stamp);
      auto b = converted.plan->sample(mapped.stamp);
      auto ca = baseline_wbc.step(baseline, {{Matrix::Identity(3, 3), a.second, Vector::Ones(3)}}, .02);
      auto cb = wbc.step(mapped, {{Matrix::Identity(3, 3), b.second, Vector::Ones(3)}}, .02);
      if (!ca.command || !cb.command) throw std::runtime_error("WBC failed");
      difference = std::max(difference, (ca.command->velocity - cb.command->velocity).cwiseAbs().maxCoeff());
      baseline.v = ca.command->velocity;
      mapped.v = cb.command->velocity;
      baseline.q += .02 * baseline.v;
      mapped.q += .02 * mapped.v;
      baseline.stamp = mapped.stamp = (k + 1) * .02;
    }
    wbc.reset();
    mpc.reset();
    if (difference > 2e-6 || (mapped.q - goal).norm() > 1e-3)
      throw std::runtime_error("reordered MPC/WBC differs from model-order baseline");
    std::cout << "250 steps, distinct WBC/MPC orders, max trajectory/command difference: " << difference << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
