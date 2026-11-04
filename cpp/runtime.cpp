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
double time_tolerance(double a, double b, double dt) {
  return std::min(4 * std::max(std::nextafter(a, std::numeric_limits<double>::infinity()) - a,
                               std::nextafter(b, std::numeric_limits<double>::infinity()) - b),
                  1e-6 * dt);
}
bool same_time(double a, double b, double dt) {
  return std::abs(a - b) <= time_tolerance(a, b, dt);
}
void validate_control(const ControlResult& control) {
  require(control.solver.status != "solved" || control.solver.x.size() > 0,
          "solved controller result must contain a decision vector");
  if (!control.solver.success())
    throw std::runtime_error("WBC rejected command: " + control.solver.status + " — " +
                             control.solver.detail);
  require(control.command.has_value(), "solved controller must return a command");
  require(std::isfinite(control.solver.violation) && control.solver.violation >= 0 &&
              control.solver.violation <= 1e-6,
          "controller reported an invalid constraint violation");
}
}  // namespace

CommandGate::CommandGate(JointLimits limits, double control_dt)
    : limits_(std::move(limits)), control_dt_(control_dt) {
  limits_.validate();
  require(std::isfinite(control_dt_) && control_dt_ > 0, "control dt must be positive and finite");
}
void CommandGate::reset() {
  last_stamp_.reset();
  observed_state_.reset();
  fault_.clear();
}
void CommandGate::latch_fault(std::string detail) {
  if (fault_.empty()) fault_ = detail.empty() ? "unspecified control fault" : std::move(detail);
}
RobotState CommandGate::observe(const RobotState& state) {
  if (!fault_.empty())
    throw std::runtime_error("command gate is faulted: " + fault_ + "; reset before continuing");
  state.validate();
  require(state.q.size() == limits_.lower.size(), "state and limits dimensions differ");
  constexpr double tolerance = 1e-6;
  require(!((state.q.array() < limits_.lower.array() - tolerance).any() ||
            (state.q.array() > limits_.upper.array() + tolerance).any()),
          "measured configuration is outside configured joint limits");
  require(!(state.v.array().abs() > limits_.velocity.array() + tolerance).any(),
          "measured velocity exceeds configured limits");
  require(!last_stamp_ || state.stamp > *last_stamp_, "robot state clock did not advance");
  last_stamp_ = state.stamp;
  observed_state_ = state;
  return state;
}
VelocityCommand CommandGate::validate(const RobotState& state,
                                      const VelocityCommand& command) const {
  if (!fault_.empty())
    throw std::runtime_error("command gate is faulted: " + fault_ + "; reset before continuing");
  require(observed_state_ && same_time(state.stamp, observed_state_->stamp, control_dt_),
          "command was not validated against the latest observed state");
  require(state.q.size() == observed_state_->q.size() &&
              state.v.size() == observed_state_->v.size() &&
              (state.q.array() == observed_state_->q.array()).all() &&
              (state.v.array() == observed_state_->v.array()).all(),
          "command state differs from the latest observed state");
  require(command.velocity.size() == state.q.size() && command.velocity.allFinite(),
          "command and measured state dimensions differ");
  require(std::isfinite(command.stamp) && std::isfinite(command.valid_until) &&
              command.stamp >= 0 && command.valid_until > command.stamp,
          "invalid command validity interval");
  require(same_time(command.stamp, state.stamp, control_dt_),
          "command timestamp does not match measured state");
  const double end = state.stamp + control_dt_;
  require(std::isfinite(end) && end > state.stamp, "control interval cannot be represented");
  require(end - command.valid_until <= time_tolerance(command.valid_until, end, control_dt_),
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
    : backend_(backend),
      controller_(controller),
      gate_(std::move(limits), control_dt),
      control_dt_(control_dt) {}
VelocityControlLoop::VelocityControlLoop(VelocityRobotBackend& backend, Controller& controller,
                                         JointLimits limits, double control_dt,
                                         double max_cycle_time)
    : VelocityControlLoop(backend, controller, std::move(limits), control_dt) {
  require(std::isfinite(max_cycle_time) && max_cycle_time > 0,
          "max cycle time must be positive and finite");
  max_cycle_time_ = max_cycle_time;
}
void VelocityControlLoop::reset() {
  try {
    controller_.reset();
    gate_.reset();
  } catch (const std::exception& exc) {
    fail(exc, "reset");
  } catch (...) {
    fail(std::runtime_error("non-standard exception during controller reset"), "reset");
  }
}
[[noreturn]] void VelocityControlLoop::fail(const std::exception& exc, const char* operation) {
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
  throw std::runtime_error(std::string("velocity control ") + operation + " fault: " + detail);
}
CycleResult VelocityControlLoop::step(const TaskFactory& task_factory) {
  if (!gate_.fault().empty())
    throw std::runtime_error("velocity control loop is faulted: " + gate_.fault() +
                             "; reset before continuing");
  require(static_cast<bool>(task_factory), "task factory must be callable");
  const auto cycle_started = std::chrono::steady_clock::now();
  const auto check_deadline = [&](const VelocityCommand* command = nullptr) {
    const double elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cycle_started).count();
    if (max_cycle_time_) {
      const double budget = command
                                ? std::min(*max_cycle_time_, command->valid_until - command->stamp)
                                : *max_cycle_time_;
      if (elapsed_seconds >= budget)
        throw std::runtime_error("velocity control cycle exceeded its execution deadline");
    }
    return elapsed_seconds * 1000;
  };
  try {
    RobotState state = gate_.observe(backend_.read());
    check_deadline();
    const auto tasks = task_factory(state);
    for (const auto& task : tasks) task.validate(state.q.size());
    check_deadline();
    const auto control_started = std::chrono::steady_clock::now();
    ControlResult control = controller_.step(state, tasks, control_dt_);
    const double control_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - control_started)
            .count();
    validate_control(control);
    const VelocityCommand command = gate_.validate(state, *control.command);
    check_deadline(&command);
    backend_.write(command);
    const double cycle_ms = check_deadline(&command);
    control.command = command;
    return {std::move(state), std::move(control), control_ms, cycle_ms};
  } catch (const std::exception& exc) {
    fail(exc);
  } catch (...) {
    fail(std::runtime_error("non-standard exception in velocity control cycle"));
  }
}
}  // namespace wholebodyx
