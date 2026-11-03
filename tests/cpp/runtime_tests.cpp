#include "wholebodyx/runtime.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <thread>

using namespace wholebodyx;
namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejected(Function function) {
  bool invalid = false;
  try { function(); } catch (const std::exception&) { invalid = true; }
  check(invalid, "expected operation to fail");
}
class MockBackend : public VelocityRobotBackend {
 public:
  explicit MockBackend(const RobotState& initial) : state(initial) {}
  RobotState state;
  double dt = .02;
  bool fail_write = false;
  int writes = 0, stops = 0, reads = 0;
  VelocityCommand last_command{Vector(), 0, 1};
  RobotState read() override { ++reads; return state; }
  void write(const VelocityCommand& command) override {
    ++writes;
    if (fail_write) throw std::runtime_error("transport disconnected");
    last_command = command;
    state = {state.q + dt * command.velocity, command.velocity, state.stamp + dt};
  }
  void stop() override {
    ++stops;
    state.v.setZero();
  }
};
class FixedController : public Controller {
 public:
  ControlResult result;
  int calls = 0, resets = 0;
  int reset_failure = 0;
  void reset() override {
    ++resets;
    if (reset_failure == 1) throw std::runtime_error("controller reset failed");
    if (reset_failure == 2) throw 7;
    if (reset_failure == 3) throw std::runtime_error("");
  }
  ControlResult step(const RobotState&, const std::vector<LinearTask>&, double) override { ++calls; return result; }
};
ControlResult solved(Vector velocity, double stamp, double until) {
  return {{"solved", velocity, 4, 0, "mock"}, VelocityCommand{velocity, stamp, until}};
}
}  // namespace

int main() {
  try {
    JointLimits limits{Vector::Constant(2, -1), Vector::Constant(2, 1),
                       (Vector(2) << .4, .8).finished(), (Vector(2) << 2, 3).finished()};
    const auto tasks = [](const RobotState&) {
      return std::vector<LinearTask>{{Matrix::Identity(2, 2), (Vector(2) << .3, -.4).finished(), Vector::Ones(2)}};
    };
    MockBackend backend{{Vector::Zero(2), Vector::Zero(2), 0}};
    KinematicWBC controller(limits);
    VelocityControlLoop loop(backend, controller, limits, .02);
    for (int i = 0; i < 10; ++i) {
      const auto result = loop.step(tasks);
      check(result.control.command && result.control.solver.success(), "valid cycle did not write command");
      check(std::isfinite(result.control_ms), "invalid timing diagnostic");
      check(result.cycle_ms >= result.control_ms, "cycle timing must include controller timing");
    }
    check(backend.writes == 10 && backend.stops == 0 && std::abs(backend.state.stamp - .2) < 1e-12, "normal loop accounting");
    check(std::abs(backend.last_command.stamp - .18) < 1e-12 && backend.last_command.valid_until >= .2 - 1e-12,
          "command timing changed");

    CommandGate gate(limits, .02);
    RobotState state{Vector::Zero(2), Vector::Zero(2), 1};
    gate.observe(state);
    check(gate.validate(state, {Vector::Zero(2), 1, 1.02}).velocity.norm() == 0, "valid gate command rejected");
    RobotState changed = state;
    changed.q[0] = .1;
    rejected([&] { gate.validate(changed, {Vector::Zero(2), 1, 1.02}); });
    changed = state;
    changed.v[0] = .1;
    rejected([&] { gate.validate(changed, {Vector::Zero(2), 1, 1.02}); });
    rejected([&] { gate.validate(state, {Vector::Zero(2), .9, 1.02}); });
    rejected([&] { gate.validate(state, {Vector::Zero(2), 1, 1.01}); });
    rejected([&] { gate.validate(state, {Vector::Ones(2), 1, 1.02}); });
    rejected([&] { gate.observe(state); });
    gate.reset();
    check(gate.observe(state).stamp == 1, "gate reset did not clear clock");

    FixedController fixed;
    MockBackend faulty{{Vector::Zero(2), Vector::Zero(2), 3}};
    fixed.result = solved(Vector::Zero(2), 0, 3.02);  // stale source timestamp
    VelocityControlLoop failing_loop(faulty, fixed, limits, .02);
    rejected([&] { failing_loop.step(tasks); });
    check(faulty.writes == 0 && faulty.stops == 1 && !failing_loop.fault().empty(), "stale command was not stopped");
    rejected([&] { failing_loop.step(tasks); });
    check(faulty.stops == 1 && fixed.calls == 1, "faulted loop did more I/O");
    fixed.result = solved(Vector::Zero(2), 3, 3.02);
    failing_loop.reset();
    check(failing_loop.step(tasks).control.command.has_value() && faulty.writes == 1 && fixed.resets == 1, "reset did not recover loop");

    for (int failure : {1, 2, 3}) {
      FixedController reset_failure;
      MockBackend reset_backend{{Vector::Zero(2), Vector::Zero(2), 3}};
      reset_failure.result = solved(Vector::Zero(2), 3, 3.02);
      VelocityControlLoop reset_loop(reset_backend, reset_failure, limits, .02);
      reset_loop.step(tasks);
      reset_failure.reset_failure = failure;
      rejected([&] { reset_loop.reset(); });
      check(reset_backend.stops == 1 && !reset_loop.fault().empty(), "failed reset did not stop/latch");
      rejected([&] { reset_loop.step(tasks); });
      check(reset_backend.reads == 1 && reset_backend.writes == 1 && reset_backend.stops == 1,
            "failed reset allowed more I/O");
      reset_failure.reset_failure = 0;
      reset_failure.result = solved(Vector::Zero(2), 3.02, 3.04);
      reset_loop.reset();
      check(reset_loop.step(tasks).control.command.has_value() && reset_backend.writes == 2,
            "successful reset did not recover after reset failure");
    }

    FixedController infeasible;
    infeasible.result = {{"infeasible", Vector(), 0, std::numeric_limits<double>::infinity(), "bounds"}, std::nullopt};
    MockBackend rejected_backend{{Vector::Zero(2), Vector::Zero(2), 4}};
    VelocityControlLoop rejected_loop(rejected_backend, infeasible, limits, .02);
    rejected([&] { rejected_loop.step(tasks); });
    check(rejected_backend.writes == 0 && rejected_backend.stops == 1, "solver failure wrote to driver");

    FixedController writer;
    writer.result = solved(Vector::Zero(2), 5, 5.02);
    MockBackend disconnected{{Vector::Zero(2), Vector::Zero(2), 5}};
    disconnected.fail_write = true;
    VelocityControlLoop write_loop(disconnected, writer, limits, .02);
    rejected([&] { write_loop.step(tasks); });
    check(disconnected.writes == 1 && disconnected.stops == 1 && !write_loop.fault().empty(), "write exception was not latched");

    MockBackend outside{{(Vector(2) << 2, 0).finished(), Vector::Zero(2), 6}};
    KinematicWBC unused(limits);
    VelocityControlLoop state_loop(outside, unused, limits, .02);
    rejected([&] { state_loop.step(tasks); });
    check(outside.writes == 0 && outside.stops == 1, "invalid state wrote to driver");
    MockBackend delayed{{Vector::Zero(2), Vector::Zero(2), 7}};
    FixedController quick;
    quick.result = solved(Vector::Zero(2), 7, 7.02);
    VelocityControlLoop deadline(delayed, quick, limits, .02, .001);
    rejected([&] { deadline.step([&](const RobotState& measured) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      return tasks(measured);
    }); });
    check(delayed.writes == 0 && delayed.stops == 1 && !deadline.fault().empty(), "deadline wrote a late command");
    rejected([&] { VelocityControlLoop invalid(delayed, quick, limits, .02, 0); });
    MockBackend interrupted{{Vector::Zero(2), Vector::Zero(2), 8}};
    VelocityControlLoop interrupted_loop(interrupted, quick, limits, .02);
    rejected([&] { interrupted_loop.step([](const RobotState&) -> std::vector<LinearTask> { throw 7; }); });
    check(interrupted.writes == 0 && interrupted.stops == 1 && !interrupted_loop.fault().empty(),
          "non-standard exception bypassed stop");
    MockBackend empty_error{{Vector::Zero(2), Vector::Zero(2), 9}};
    VelocityControlLoop empty_error_loop(empty_error, quick, limits, .02);
    rejected([&] { empty_error_loop.step([](const RobotState&) -> std::vector<LinearTask> {
      throw std::runtime_error("");
    }); });
    rejected([&] { empty_error_loop.step(tasks); });
    check(empty_error.reads == 1 && empty_error.stops == 1 && !empty_error_loop.fault().empty(),
          "empty exception detail did not latch the fault");
    std::cout << "Runtime gate, fault latch, safe stop and recovery checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
