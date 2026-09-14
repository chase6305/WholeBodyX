#include "wholebodyx/joint_order.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace wholebodyx;
namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejected(Function function) {
  bool invalid = false;
  try { function(); } catch (const std::invalid_argument&) { invalid = true; }
  check(invalid, "invalid adapter input was accepted");
}
class StubController : public Controller {
 public:
  ControlResult result;
  bool reset_called = false;
  void reset() override { reset_called = true; }
  ControlResult step(const RobotState&, const std::vector<LinearTask>&, double) override { return result; }
};
class StubPlanner : public Planner {
 public:
  PlanResult result;
  bool reset_called = false;
  void reset() override { reset_called = true; }
  PlanResult solve(const RobotState&, const Vector&) override { return result; }
};
}  // namespace

int main() {
  try {
    rejected([] { JointOrder({}, {}); });
    rejected([] { JointOrder({"a", "a"}, {"a", "b"}); });
    rejected([] { JointOrder({"a", "b"}, {"b", "b"}); });
    rejected([] { JointOrder({"a", "b"}, {"a", "c"}); });
    rejected([] { JointOrder({"a", "b"}, {"a"}); });
    rejected([] { JointOrder({"a "}, {"a "}); });
    JointOrder order({"a", "b", "c"}, {"c", "a", "b"});
    Vector q = (Vector(3) << .1, .3, -.7).finished();
    check((order.to_backend(q) - (Vector(3) << -.7, .1, .3).finished()).norm() == 0, "forward order");
    check((order.to_model(order.to_backend(q)) - q).norm() == 0, "inverse order");
    rejected([&] { order.to_backend(Vector::Zero(2)); });
    rejected([&] { order.columns_to_backend(Matrix::Zero(3, 2)); });
    rejected([&] { order.to_backend(Vector::Constant(3, std::numeric_limits<double>::quiet_NaN())); });

    Matrix j(2, 3); j << 1, 3, -2, .2, -.1, .7;
    LinearTask task{j, (Vector(2) << .2, -.3).finished(), (Vector(2) << 2, 7).finished()};
    const auto mapped_task = order.task_to_backend(task);
    check((mapped_task.jacobian * order.to_backend(q) - j * q).norm() < 1e-12, "task residual changed");
    check((mapped_task.weight - task.weight).norm() == 0, "task weights changed");
    check((mapped_task.target_velocity - task.target_velocity).norm() == 0, "task row target changed");

    JointLimits limits{(Vector(3) << -1.2, -.7, -2.1).finished(), (Vector(3) << 1, 1.4, .8).finished(),
                       (Vector(3) << .15, .6, .9).finished(), {}};
    RobotState state{(Vector(3) << .2, .25, -.35).finished(), (Vector(3) << .05, -.12, .08).finished(), 2.7};
    for (bool acceleration : {false, true}) {
      if (acceleration) limits.acceleration = (Vector(3) << 2, 3, 4).finished();
      auto mapped_limits = order.limits_to_backend(limits);
      auto a = limits.velocity_bounds(state, .02);
      auto b = mapped_limits.velocity_bounds(order.state_to_backend(state), .02);
      check((order.to_backend(a.first) - b.first).norm() < 1e-12, "lower bound order");
      check((order.to_backend(a.second) - b.second).norm() < 1e-12, "upper bound order");
      KinematicWBC baseline(limits, .01, .03);
      ReorderedController converted(std::make_unique<KinematicWBC>(mapped_limits, .01, .03), order);
      auto original = baseline.step(state, {task}, .02);
      auto result = converted.step(state, {task}, .02);
      check(original.command && result.command, "WBC solve failed");
      check((original.command->velocity - result.command->velocity).norm() < 2e-7, "WBC permutation changed solution");
    }
    JointMPC baseline(limits, 8);
    ReorderedPlanner converted(std::make_unique<JointMPC>(order.limits_to_backend(limits), 8), order);
    Vector goal = (Vector(3) << .8, -.4, .6).finished();
    auto original = baseline.solve(state, goal);
    auto result = converted.solve(state, goal);
    check(original.plan && result.plan, "MPC solve failed");
    check((original.plan->positions - result.plan->positions).cwiseAbs().maxCoeff() < 2e-7, "MPC position order");
    check((original.plan->velocities - result.plan->velocities).cwiseAbs().maxCoeff() < 2e-7, "MPC velocity order");
    for (double stamp : {2.7, 2.74, 2.8, 3.49}) {
      auto a = original.plan->sample(stamp), b = result.plan->sample(stamp);
      check((a.first - b.first).norm() < 2e-7 && (a.second - b.second).norm() < 2e-7, "plan sampling changed");
    }

    auto controller = std::make_unique<StubController>();
    auto* source = controller.get();
    ReorderedController adapter(std::move(controller), order);
    source->result.solver.status = "infeasible";
    check(!adapter.step(state, {}, .02).command, "failure emitted a command");
    source->result.command = VelocityCommand{Vector::Zero(3), state.stamp, state.stamp + .02};
    rejected([&] { adapter.step(state, {}, .02); });
    source->result.solver = {"solved", (Vector(4) << 7, 8, 9, 10).finished(), 12, 0, "external"};
    check((adapter.step(state, {}, .02).solver.x - source->result.solver.x).norm() == 0, "QP decision layout changed");
    source->result.command->velocity = Vector::Zero(2);
    rejected([&] { adapter.step(state, {}, .02); });
    source->result.command.reset();
    rejected([&] { adapter.step(state, {}, .02); });
    adapter.reset();
    check(source->reset_called, "controller reset not forwarded");
    auto planner = std::make_unique<StubPlanner>();
    auto* planner_source = planner.get();
    ReorderedPlanner planner_adapter(std::move(planner), order);
    planner_source->result.solver.status = "infeasible";
    check(!planner_adapter.solve(state, goal).plan, "failure emitted a plan");
    planner_source->result.plan = original.plan;
    rejected([&] { planner_adapter.solve(state, goal); });
    planner_source->result.solver = {"solved", Vector::Ones(4), 12, 0, "external"};
    planner_source->result.plan.reset();
    rejected([&] { planner_adapter.solve(state, goal); });
    planner_adapter.reset();
    check(planner_source->reset_called, "planner reset not forwarded");
    std::cout << "Joint permutations, task residuals, asymmetric constraints, trajectories and failures passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
