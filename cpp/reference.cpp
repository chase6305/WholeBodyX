#include "wholebodyx/reference.hpp"

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
double ulp(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity()) - value;
}
bool deadline_reached(double stamp, double deadline, double origin, double dt) {
  const double ticks = std::max(1., std::ceil(std::max(0., stamp - origin) / dt));
  const double tolerance = std::min((4 + ticks) * std::max(ulp(stamp), ulp(deadline)), .25 * dt);
  return stamp >= deadline - tolerance;
}
}  // namespace

MPCReferenceManager::MPCReferenceManager(Planner& planner, JointLimits limits,
                                         double replan_period, double tracking_tolerance)
    : planner_(planner), limits_(std::move(limits)), replan_period_(replan_period),
      tracking_tolerance_(tracking_tolerance) {
  limits_.validate();
  require(std::isfinite(replan_period_) && replan_period_ > 0, "replan period must be positive and finite");
  require(std::isfinite(tracking_tolerance_) && tracking_tolerance_ > 0,
          "tracking tolerance must be positive and finite");
}
void MPCReferenceManager::clear() {
  plan_.reset();
  goal_.resize(0);
  next_replan_.reset();
  last_stamp_.reset();
  clock_invalid_ = false;
  pending_reason_ = "initial";
}
void MPCReferenceManager::invalidate(std::string reason) {
  plan_.reset();
  next_replan_.reset();
  pending_reason_ = std::move(reason);
}
void MPCReferenceManager::reset() {
  invalidate();
  planner_.reset();
  clear();
}
void MPCReferenceManager::discard_outdated(double stamp, const Vector& goal) {
  require(std::isfinite(stamp) && stamp >= 0, "invalid reference query time");
  require(goal.size() == limits_.lower.size() && goal.allFinite(), "invalid reference goal");
  if (!plan_) return;
  if (goal_.size() != goal.size() || !(goal_.array() == goal.array()).all())
    invalidate("goal_changed");
  else if (stamp < plan_->stamp || stamp >= plan_->valid_until())
    invalidate("expired");
}
JointPlan MPCReferenceManager::accept(const JointPlan& proposed, const RobotState& state, double control_dt) const {
  JointPlan plan = proposed;  // Own all arrays, independently of planner storage.
  plan.validate();
  require(plan.stamp == state.stamp, "planner returned a stale or future-dated plan");
  require(plan.positions.cols() == state.q.size() &&
          (plan.positions.row(0).transpose() - state.q).cwiseAbs().maxCoeff() <= 1e-6,
          "plan does not start at the supplied measured state");
  require(plan.valid_until() >= state.stamp + control_dt, "plan does not cover the complete control interval");
  constexpr double tolerance = 1e-6;
  for (Eigen::Index k = 0; k < plan.velocities.rows(); ++k) {
    require(!((plan.positions.row(k + 1).transpose().array() < limits_.lower.array() - tolerance).any() ||
              (plan.positions.row(k + 1).transpose().array() > limits_.upper.array() + tolerance).any() ||
              (plan.velocities.row(k).transpose().array().abs() > limits_.velocity.array() + tolerance).any()),
            "plan violates position or velocity limits");
    if (limits_.acceleration.size()) {
      Vector previous = k == 0 ? state.v : Vector(plan.velocities.row(k - 1).transpose());
      require(!((plan.velocities.row(k).transpose() - previous).array().abs() >
                 limits_.acceleration.array() * plan.dt + tolerance).any(),
              "plan violates velocity-slew limits");
    }
  }
  return plan;
}
ReferenceUpdate MPCReferenceManager::update(const RobotState& state, const Vector& goal, double control_dt) {
  state.validate();
  require(state.q.size() == limits_.lower.size(), "state and limits dimensions differ");
  require(goal.size() == state.q.size() && goal.allFinite(), "goal and state dimensions differ");
  require(std::isfinite(control_dt) && control_dt > 0, "control dt must be positive and finite");
  const double end = state.stamp + control_dt;
  require(std::isfinite(end) && end > state.stamp, "control interval cannot be represented");
  ReferenceUpdate update;
  if (clock_invalid_ || (last_stamp_ && state.stamp < *last_stamp_)) {
    clock_invalid_ = true;
    invalidate("clock_rollback");
    update.reason = "clock_rollback";
    update.solver = QPResult{"error", {}, 0, INFINITY, "clock moved backward; reset the runtime"};
    return update;
  }
  last_stamp_ = state.stamp;
  discard_outdated(state.stamp, goal);
  if (!plan_) update.reason = pending_reason_;
  else if (end > plan_->valid_until()) update.reason = "coverage";
  else {
    update.tracking_error = (state.q - plan_->sample(state.stamp).first).cwiseAbs().maxCoeff();
    if (*update.tracking_error > tracking_tolerance_) update.reason = "tracking_error";
    else if (deadline_reached(state.stamp, *next_replan_, plan_->stamp, control_dt)) update.reason = "periodic";
    else update.reason = "reused";
  }
  if (update.reason != "reused") {
    invalidate(update.reason);
    const auto started = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    };
    const auto failure = [&](const std::exception& exc) {
      update.solver = QPResult{"error", {}, 0, INFINITY, exc.what()};
      update.solve_ms = elapsed();
      return update;
    };
    try {
      const auto outcome = planner_.solve(state, goal);
      update.solver = outcome.solver;
      if (!outcome.solver.success() || !outcome.plan) {
        if (outcome.solver.status == "solved")
          update.solver = QPResult{"error", {}, 0, INFINITY, "solved result has no valid solution or plan"};
        update.solve_ms = elapsed();
        return update;
      }
      JointPlan accepted = accept(*outcome.plan, state, control_dt);
      const double next = state.stamp + replan_period_;
      require(std::isfinite(next) && next > state.stamp, "replan deadline cannot be represented");
      plan_ = std::move(accepted);
      goal_ = goal;
      next_replan_ = next;
    } catch (const std::invalid_argument& exc) { return failure(exc); }
      catch (const std::runtime_error& exc) { return failure(exc); }
    update.solve_ms = elapsed();
  }
  auto sampled = plan_->sample(state.stamp);
  update.position = std::move(sampled.first);
  update.velocity = std::move(sampled.second);
  return update;
}

}  // namespace wholebodyx
