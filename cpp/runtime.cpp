#include "wholebodyx/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace wholebodyx {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
double time_tolerance(double a, double b) {
  return 4 * std::max(std::nextafter(a, std::numeric_limits<double>::infinity()) - a,
                      std::nextafter(b, std::numeric_limits<double>::infinity()) - b);
}
bool same_time(double a, double b) { return std::abs(a - b) <= time_tolerance(a, b); }
void validate_control(const ControlResult& control) {
  require(control.solver.status != "solved" || control.solver.x.size() > 0,
          "solved controller result must contain a decision vector");
  if (!control.solver.success()) throw std::runtime_error("WBC rejected command: " + control.solver.status + " — " + control.solver.detail);
  require(control.command.has_value(), "solved controller must return a command");
  require(std::isfinite(control.solver.violation) && control.solver.violation >= 0 && control.solver.violation <= 1e-6,
          "controller reported an invalid constraint violation");
}
}  // namespace

CommandGate::CommandGate(JointLimits limits, double control_dt) : limits_(std::move(limits)), control_dt_(control_dt) {
  limits_.validate();
  require(std::isfinite(control_dt_) && control_dt_ > 0, "control dt must be positive and finite");
}
void CommandGate::reset() {
  last_stamp_.reset();
  observed_stamp_.reset();
  fault_.clear();
}
void CommandGate::latch_fault(std::string detail) {
  if (fault_.empty()) fault_ = std::move(detail);
}
RobotState CommandGate::observe(const RobotState& state) {
  if (!fault_.empty()) throw std::runtime_error("command gate is faulted: " + fault_ + "; reset before continuing");
  state.validate();
  require(state.q.size() == limits_.lower.size(), "state and limits dimensions differ");
  constexpr double tolerance = 1e-6;
  require(!((state.q.array() < limits_.lower.array() - tolerance).any() ||
            (state.q.array() > limits_.upper.array() + tolerance).any()),
          "measured configuration is outside configured joint limits");
  require(!(state.v.array().abs() > limits_.velocity.array() + tolerance).any(),
          "measured velocity exceeds configured limits");
  require(!last_stamp_ || state.stamp > *last_stamp_, "robot state clock did not advance");
  last_stamp_ = observed_stamp_ = state.stamp;
  return state;
}
VelocityCommand CommandGate::validate(const RobotState& state, const VelocityCommand& command) const {
  if (!fault_.empty()) throw std::runtime_error("command gate is faulted: " + fault_ + "; reset before continuing");
  require(observed_stamp_ && same_time(state.stamp, *observed_stamp_),
          "command was not validated against the latest observed state");
  require(command.velocity.size() == state.q.size() && command.velocity.allFinite(),
          "command and measured state dimensions differ");
  require(std::isfinite(command.stamp) && std::isfinite(command.valid_until) && command.stamp >= 0 &&
          command.valid_until > command.stamp, "invalid command validity interval");
  require(same_time(command.stamp, state.stamp), "command timestamp does not match measured state");
  const double end = state.stamp + control_dt_;
  require(std::isfinite(end) && end > state.stamp, "control interval cannot be represented");
  require(command.valid_until + time_tolerance(command.valid_until, end) >= end,
          "command does not cover the complete control interval");
  const auto bounds = limits_.velocity_bounds(state, control_dt_);
  constexpr double tolerance = 1e-6;
  require(!((command.velocity.array() < bounds.first.array() - tolerance).any() ||
            (command.velocity.array() > bounds.second.array() + tolerance).any()),
          "command violates configured position, velocity, or slew limits");
  return command;
}

VelocityControlLoop::VelocityControlLoop(VelocityRobotBackend& backend, Controller& controller,
                                         JointLimits limits, double control_dt)
    : backend_(backend), controller_(controller), gate_(std::move(limits), control_dt), control_dt_(control_dt) {}
void VelocityControlLoop::reset() {
  controller_.reset();
  gate_.reset();
}
[[noreturn]] void VelocityControlLoop::fail(const std::exception& exc) {
  gate_.latch_fault(exc.what());
  std::string detail = exc.what();
  try {
    backend_.stop();
  } catch (const std::exception& stop_error) {
    detail += "; backend stop failed: ";
    detail += stop_error.what();
  } catch (...) {
    detail += "; backend stop failed with unknown exception";
  }
  throw std::runtime_error("velocity control cycle fault: " + detail);
}
CycleResult VelocityControlLoop::step(const TaskFactory& task_factory) {
  if (!gate_.fault().empty()) throw std::runtime_error("velocity control loop is faulted: " + gate_.fault() + "; reset before continuing");
  require(static_cast<bool>(task_factory), "task factory must be callable");
  try {
    const RobotState state = gate_.observe(backend_.read());
    const auto tasks = task_factory(state);
    for (const auto& task : tasks) task.validate(state.q.size());
    const auto started = std::chrono::steady_clock::now();
    ControlResult control = controller_.step(state, tasks, control_dt_);
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    validate_control(control);
    const VelocityCommand command = gate_.validate(state, *control.command);
    backend_.write(command);
    control.command = command;
    return {state, std::move(control), elapsed};
  } catch (const std::exception& exc) {
    fail(exc);
  }
}
}  // namespace wholebodyx
