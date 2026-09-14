#include "wholebodyx/control.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace wholebodyx {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void positive(double value, const char* name) { require(std::isfinite(value) && value > 0, name); }
double ulp(double value) {
  const double next = std::nextafter(value, INFINITY);
  return std::isfinite(next) ? next - value : value - std::nextafter(value, -INFINITY);
}
double time_window(double stamp, double dt, Eigen::Index count) {
  const double end = stamp + dt * count;
  require(std::isfinite(stamp) && stamp >= 0 && std::isfinite(end) && end > stamp &&
          dt > 8 * std::max(ulp(stamp), ulp(end)), "plan time window must be finite and resolve each time step");
  return end;
}
Eigen::Index interval(double time, double start, double dt, Eigen::Index count) {
  if (time >= start + count * dt) return count;
  const double elapsed = (time - start) / dt;
  const auto nearest = static_cast<Eigen::Index>(std::round(elapsed));
  const double knot = start + nearest * dt;
  if (std::abs(time - knot) <= 4 * std::max(ulp(time), ulp(knot))) return std::min(nearest, count - 1);
  return std::min(static_cast<Eigen::Index>(elapsed), count - 1);
}
}  // namespace

void RobotState::validate() const {
  require(q.size() > 0 && q.size() == v.size() && q.allFinite() && v.allFinite(), "invalid fixed-base state");
  require(std::isfinite(stamp) && stamp >= 0, "invalid timestamp");
}
void JointLimits::validate() const {
  const auto n = lower.size();
  require(n > 0 && upper.size() == n && velocity.size() == n, "invalid joint limit dimensions");
  require(lower.allFinite() && upper.allFinite() && velocity.allFinite(), "nonfinite joint limits");
  require((lower.array() < upper.array()).all() && (velocity.array() > 0).all(), "invalid joint limits");
  require(acceleration.size() == 0 || (acceleration.size() == n && acceleration.allFinite() &&
          (acceleration.array() > 0).all()), "invalid acceleration limits");
}
std::pair<Vector, Vector> JointLimits::velocity_bounds(const RobotState& state, double dt) const {
  validate();
  state.validate();
  positive(dt, "dt must be positive and finite");
  require(state.q.size() == lower.size(), "state and limits dimensions differ");
  Vector lo = (-velocity).cwiseMax((lower - state.q) / dt);
  Vector hi = velocity.cwiseMin((upper - state.q) / dt);
  if (acceleration.size()) {
    lo = lo.cwiseMax(state.v - acceleration * dt);
    hi = hi.cwiseMin(state.v + acceleration * dt);
  }
  return {lo, hi};
}
void LinearTask::validate(Eigen::Index joints) const {
  require(jacobian.cols() == joints && jacobian.rows() > 0 && target_velocity.size() == jacobian.rows() &&
          weight.size() == jacobian.rows(), "invalid task dimensions");
  require(jacobian.allFinite() && target_velocity.allFinite() && weight.allFinite() &&
          (weight.array() >= 0).all(), "invalid task values");
}
void QuadraticProgram::validate() const {
  validate_against(nullptr);
}
void QuadraticProgram::validate_against(const Matrix* previous) const {
  const auto n = gradient.size(), m = lower.size();
  require(n > 0 && hessian.rows() == n && hessian.cols() == n && constraint.rows() == m &&
          constraint.cols() == n && upper.size() == m, "invalid QP dimensions");
  require(hessian.allFinite() && gradient.allFinite() && constraint.allFinite() &&
          !lower.array().isNaN().any() && !upper.array().isNaN().any(), "invalid QP values");
  require((hessian - hessian.transpose()).cwiseAbs().maxCoeff() <= 1e-10, "nonsymmetric Hessian");
  if (previous && previous->rows() == n && previous->cols() == n && (previous->array() == hessian.array()).all())
    return;  // Exact values only: caller mutation always triggers convexity validation.
  Eigen::SelfAdjointEigenSolver<Matrix> eig(hessian, Eigen::EigenvaluesOnly);
  require(eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() >= -1e-10, "nonconvex QP");
}

KinematicWBC::KinematicWBC(JointLimits limits, double regularization, double smoothing,
                         std::shared_ptr<QPSolver> solver)
    : limits_(std::move(limits)), regularization_(regularization), smoothing_(smoothing),
      solver_(solver ? std::move(solver) : std::make_shared<OSQPSolver>()) {
  limits_.validate();
  positive(regularization_, "invalid regularization");
  require(std::isfinite(smoothing_) && smoothing_ >= 0, "invalid smoothing");
}
void KinematicWBC::reset() { solver_->reset(); }
QuadraticProgram KinematicWBC::formulate(const RobotState& state, const std::vector<LinearTask>& tasks, double dt) const {
  auto bounds = limits_.velocity_bounds(state, dt);
  const auto n = state.q.size();
  Matrix h = (regularization_ + smoothing_) * Matrix::Identity(n, n);
  Vector g = -smoothing_ * state.v;
  for (const auto& task : tasks) {
    task.validate(n);
    h.noalias() += task.jacobian.transpose() * task.weight.asDiagonal() * task.jacobian;
    g.noalias() -= task.jacobian.transpose() * task.weight.asDiagonal() * task.target_velocity;
  }
  return {h, g, Matrix::Identity(n, n), bounds.first, bounds.second};
}
ControlResult KinematicWBC::step(const RobotState& state, const std::vector<LinearTask>& tasks, double dt) {
  auto result = solver_->solve(formulate(state, tasks, dt));
  if (!result.success()) return {result, std::nullopt};
  return {result, VelocityCommand{result.x, state.stamp, state.stamp + dt}};
}

double JointPlan::valid_until() const { return stamp + dt * velocities.rows(); }
void JointPlan::validate() const {
  positive(dt, "invalid plan time step");
  require(velocities.rows() > 0 && velocities.cols() > 0 && positions.rows() == velocities.rows() + 1 &&
          positions.cols() == velocities.cols() && positions.allFinite() && velocities.allFinite(), "invalid plan shape");
  time_window(stamp, dt, velocities.rows());
  Matrix expected = dt * velocities;
  Matrix difference = positions.bottomRows(velocities.rows()) - positions.topRows(velocities.rows());
  require(((difference - expected).array().abs() <= 1e-8 + 1e-7 * expected.array().abs()).all(),
          "plan positions and velocities must satisfy integrator dynamics");
}
std::pair<Vector, Vector> JointPlan::sample(double time) const {
  validate();
  require(std::isfinite(time) && time >= stamp && time < valid_until(), "reference is future-dated or expired");
  const auto k = interval(time, stamp, dt, velocities.rows());
  Vector v = velocities.row(k).transpose();
  return {positions.row(k).transpose() + std::max(0.0, time - stamp - k * dt) * v, v};
}

JointMPC::JointMPC(JointLimits limits, int horizon, double dt, double tracking_weight,
                   double control_weight, double terminal_weight, std::shared_ptr<QPSolver> solver)
    : limits_(std::move(limits)), horizon_(horizon), dt_(dt), tracking_weight_(tracking_weight),
      control_weight_(control_weight), terminal_weight_(terminal_weight),
      solver_(solver ? std::move(solver) : std::make_shared<OSQPSolver>()) {
  limits_.validate();
  require(horizon_ > 0, "invalid horizon");
  positive(dt_, "invalid MPC dt");
  positive(tracking_weight_, "invalid tracking weight");
  positive(control_weight_, "invalid control weight");
  positive(terminal_weight_, "invalid terminal weight");
  time_window(0, dt_, horizon_);
  const auto n = limits_.lower.size();
  const auto size = n * horizon_;
  Matrix prediction = Matrix::Zero(size, size);
  Vector weights = Vector::Constant(size, tracking_weight_);
  weights.tail(n).setConstant(terminal_weight_);
  Matrix repeated = Matrix::Zero(size, n);
  const bool slew = limits_.acceleration.size() != 0;
  Matrix a = Matrix::Zero((slew ? 3 : 2) * size, size);
  a.topRows(size).setIdentity();
  Vector lo(a.rows()), hi(a.rows());
  for (int k = 0; k < horizon_; ++k) {
    for (int j = 0; j <= k; ++j) prediction.block(k * n, j * n, n, n).diagonal().setConstant(dt_);
    repeated.block(k * n, 0, n, n).setIdentity();
    lo.segment(k * n, n) = -limits_.velocity;
    hi.segment(k * n, n) = limits_.velocity;
    lo.segment(size + k * n, n) = limits_.lower;
    hi.segment(size + k * n, n) = limits_.upper;
    if (slew) {
      const auto row = 2 * size + k * n;
      a.block(row, k * n, n, n).setIdentity();
      if (k > 0) a.block(row, (k - 1) * n, n, n).diagonal().setConstant(-1);
      lo.segment(row, n) = -dt_ * limits_.acceleration;
      hi.segment(row, n) = dt_ * limits_.acceleration;
    }
  }
  a.middleRows(size, size) = prediction;
  hessian_ = prediction.transpose() * weights.asDiagonal() * prediction;
  hessian_.diagonal().array() += control_weight_;
  gradient_map_ = prediction.transpose() * weights.asDiagonal() * repeated;
  constraint_ = std::move(a);
  lower_ = std::move(lo);
  upper_ = std::move(hi);
}
void JointMPC::reset() { solver_->reset(); last_plan_.reset(); }
QuadraticProgram JointMPC::formulate(const RobotState& state, const Vector& goal) const {
  state.validate();
  time_window(state.stamp, dt_, horizon_);
  const auto n = state.q.size(), size = n * horizon_;
  require(n == limits_.lower.size() && goal.size() == n && goal.allFinite(), "invalid MPC dimensions or goal");
  Vector lo = lower_, hi = upper_;
  for (int k = 0; k < horizon_; ++k) {
    lo.segment(size + k * n, n) -= state.q;
    hi.segment(size + k * n, n) -= state.q;
  }
  if (limits_.acceleration.size()) {
    lo.segment(2 * size, n) += state.v;
    hi.segment(2 * size, n) += state.v;
  }
  Vector gradient = gradient_map_ * (state.q - goal);
  require(gradient.allFinite() && lo.allFinite() && hi.allFinite(), "nonfinite MPC vectors");
  return {hessian_, gradient, constraint_, lo, hi};
}
PlanResult JointMPC::solve(const RobotState& state, const Vector& goal) {
  auto problem = formulate(state, goal);
  const auto n = state.q.size();
  Vector warm;
  if (last_plan_ && state.stamp >= last_plan_->stamp) {
    const int offset = static_cast<int>(interval(state.stamp, last_plan_->stamp, dt_, horizon_));
    warm.resize(n * horizon_);
    for (int k = 0; k < horizon_; ++k)
      warm.segment(k * n, n) = last_plan_->velocities.row(std::min(k + offset, horizon_ - 1)).transpose();
  }
  auto result = solver_->solve(problem, warm);
  if (!result.success()) return {result, std::nullopt};
  JointPlan plan{Matrix(horizon_ + 1, n), Matrix(horizon_, n), state.stamp, dt_};
  plan.positions.row(0) = state.q.transpose();
  for (int k = 0; k < horizon_; ++k) {
    plan.velocities.row(k) = result.x.segment(k * n, n).transpose();
    plan.positions.row(k + 1) = plan.positions.row(k) + dt_ * plan.velocities.row(k);
  }
  last_plan_ = plan;
  return {result, plan};
}
}  // namespace wholebodyx
