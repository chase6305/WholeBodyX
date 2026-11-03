#include <wholebodyx/control.hpp>
#include <wholebodyx/joint_order.hpp>
#include <wholebodyx/runtime.hpp>
#include <wholebodyx/reference.hpp>
#include <wholebodyx/sparse_mpc.hpp>

struct Backend : wholebodyx::VelocityRobotBackend {
  int writes = 0, stops = 0;
  wholebodyx::RobotState read() override {
    return {wholebodyx::Vector::Zero(1), wholebodyx::Vector::Zero(1), 0};
  }
  void write(const wholebodyx::VelocityCommand&) override { ++writes; }
  void stop() override { ++stops; }
};

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
  Backend backend;
  VelocityControlLoop loop(backend, controller, limits, 1., .5);
  const auto cycle = loop.step([](const RobotState&) {
    return std::vector<LinearTask>{{Matrix::Identity(1, 1), Vector::Constant(1, .5), Vector::Ones(1)}};
  });
  JointMPC planner(limits);
  MPCReferenceManager references(planner, limits);
  const auto reference = references.update({Vector::Zero(1), Vector::Zero(1), 0}, Vector::Constant(1, .3), .02);
  SparseJointMPC sparse(limits);
  MPCReferenceManager sparse_references(sparse, limits);
  const auto sparse_reference = sparse_references.update({Vector::Zero(1), Vector::Zero(1), 0}, Vector::Constant(1, .3), .02);
  return result.solver.success() && converted.command &&
         (converted.command->velocity - three.velocity).norm() < 1e-6 &&
         cycle.control.command && cycle.cycle_ms >= cycle.control_ms &&
         backend.writes == 1 && backend.stops == 0 && reference.success() && reference.replanned() &&
         sparse_reference.success() && sparse_reference.replanned() ? 0 : 1;
}
