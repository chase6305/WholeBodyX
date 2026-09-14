#include <wholebodyx/control.hpp>
#include <wholebodyx/joint_order.hpp>

int main() {
  using namespace wholebodyx;
  JointLimits limits{Vector::Constant(1, -1), Vector::Constant(1, 1), Vector::Ones(1), {}};
  KinematicWBC controller(limits);
  const auto result = controller.step({Vector::Zero(1), Vector::Zero(1), 0},
                                      {{Matrix::Identity(1, 1), Vector::Constant(1, 0.5), Vector::Ones(1)}}, 0.01);
  JointOrder order({"a", "b", "c"}, {"c", "a", "b"});
  JointLimits three{Vector::Constant(3, -1), Vector::Ones(3), (Vector(3) << .2, .4, .6).finished(), {}};
  ReorderedController wrapped(std::make_unique<KinematicWBC>(order.limits_to_backend(three)), order);
  auto converted = wrapped.step({Vector::Zero(3), Vector::Zero(3), 0},
                                {{Matrix::Identity(3, 3), Vector::Ones(3), Vector::Ones(3)}}, .02);
  return result.solver.success() && converted.command &&
         (converted.command->velocity - three.velocity).norm() < 1e-6 ? 0 : 1;
}
