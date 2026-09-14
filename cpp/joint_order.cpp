#include "wholebodyx/joint_order.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace wholebodyx {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void validate_names(const std::vector<std::string>& names) {
  require(!names.empty(), "joint names must not be empty");
  std::unordered_set<std::string> seen;
  for (const auto& name : names) {
    require(!name.empty(), "joint names must not be empty");
    require(!std::isspace(static_cast<unsigned char>(name.front())) &&
            !std::isspace(static_cast<unsigned char>(name.back())), "joint name has surrounding whitespace");
    require(seen.insert(name).second, "duplicate joint name");
  }
}
Vector permute(const Vector& value, const std::vector<Eigen::Index>& indices) {
  require(value.size() == static_cast<Eigen::Index>(indices.size()) && value.allFinite(),
          "joint vector must be finite and match joint names");
  Vector result(value.size());
  for (Eigen::Index i = 0; i < value.size(); ++i) result[i] = value[indices[i]];
  return result;
}
Matrix permute_columns(const Matrix& value, const std::vector<Eigen::Index>& indices) {
  require(value.cols() == static_cast<Eigen::Index>(indices.size()) && value.allFinite(),
          "joint matrix must be finite and have one column per joint");
  Matrix result(value.rows(), value.cols());
  for (Eigen::Index i = 0; i < value.cols(); ++i) result.col(i) = value.col(indices[i]);
  return result;
}
void validate_solver(const QPResult& result) {
  require(result.x.allFinite(), "solver decision vector must be finite");
  require(result.status != "solved" || result.x.size() > 0, "solved result must contain a decision vector");
}
}  // namespace

JointOrder::JointOrder(std::vector<std::string> model_names, std::vector<std::string> backend_names)
    : model_names_(std::move(model_names)), backend_names_(std::move(backend_names)) {
  validate_names(model_names_);
  validate_names(backend_names_);
  require(model_names_.size() == backend_names_.size(), "joint name sets differ");
  std::unordered_map<std::string, Eigen::Index> positions;
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(model_names_.size()); ++i) positions.emplace(model_names_[i], i);
  inverse_.resize(model_names_.size());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(backend_names_.size()); ++i) {
    const auto found = positions.find(backend_names_[i]);
    require(found != positions.end(), "joint name sets differ");
    forward_.push_back(found->second);
    inverse_[found->second] = i;
  }
}
Vector JointOrder::to_backend(const Vector& value) const { return permute(value, forward_); }
Vector JointOrder::to_model(const Vector& value) const { return permute(value, inverse_); }
Matrix JointOrder::columns_to_backend(const Matrix& value) const { return permute_columns(value, forward_); }
Matrix JointOrder::columns_to_model(const Matrix& value) const { return permute_columns(value, inverse_); }
RobotState JointOrder::state_to_backend(const RobotState& state) const {
  state.validate();
  return {to_backend(state.q), to_backend(state.v), state.stamp};
}
JointLimits JointOrder::limits_to_backend(const JointLimits& limits) const {
  limits.validate();
  return {to_backend(limits.lower), to_backend(limits.upper), to_backend(limits.velocity),
          limits.acceleration.size() ? to_backend(limits.acceleration) : Vector()};
}
LinearTask JointOrder::task_to_backend(const LinearTask& task) const {
  task.validate(static_cast<Eigen::Index>(forward_.size()));
  return {columns_to_backend(task.jacobian), task.target_velocity, task.weight};
}
VelocityCommand JointOrder::command_to_model(const VelocityCommand& command) const {
  require(std::isfinite(command.stamp) && command.stamp >= 0 && std::isfinite(command.valid_until)
          && command.valid_until > command.stamp, "invalid command validity interval");
  return {to_model(command.velocity), command.stamp, command.valid_until};
}
JointPlan JointOrder::plan_to_model(const JointPlan& plan) const {
  plan.validate();
  return {columns_to_model(plan.positions), columns_to_model(plan.velocities), plan.stamp, plan.dt};
}

ReorderedController::ReorderedController(std::unique_ptr<Controller> controller, JointOrder order)
    : controller_(std::move(controller)), order_(std::move(order)) {
  require(controller_ != nullptr, "controller must not be null");
}
void ReorderedController::reset() { controller_->reset(); }
ControlResult ReorderedController::step(const RobotState& state, const std::vector<LinearTask>& tasks, double dt) {
  std::vector<LinearTask> converted;
  converted.reserve(tasks.size());
  for (const auto& task : tasks) converted.push_back(order_.task_to_backend(task));
  auto result = controller_->step(order_.state_to_backend(state), converted, dt);
  validate_solver(result.solver);
  if (!result.solver.success()) {
    require(!result.command, "failed controller returned a command");
    return result;
  }
  require(result.command.has_value(), "solved controller must return a command");
  result.command = order_.command_to_model(*result.command);
  return result;
}

ReorderedPlanner::ReorderedPlanner(std::unique_ptr<Planner> planner, JointOrder order)
    : planner_(std::move(planner)), order_(std::move(order)) {
  require(planner_ != nullptr, "planner must not be null");
}
void ReorderedPlanner::reset() { planner_->reset(); }
PlanResult ReorderedPlanner::solve(const RobotState& state, const Vector& goal) {
  auto result = planner_->solve(order_.state_to_backend(state), order_.to_backend(goal));
  validate_solver(result.solver);
  if (!result.solver.success()) {
    require(!result.plan, "failed planner returned a plan");
    return result;
  }
  require(result.plan.has_value(), "solved planner must return a plan");
  result.plan = order_.plan_to_model(*result.plan);
  return result;
}
}  // namespace wholebodyx
