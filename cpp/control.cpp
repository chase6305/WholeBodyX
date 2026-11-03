#include "wholebodyx/control.hpp"
#include "time_utils.hpp"
#include "fixed_qp.hpp"
#include "plan_validation.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
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
double braking_speed(double distance, double acceleration, double dt) {
  if (acceleration * dt == 0) return 0;
  // Evaluate in extended precision so finite double inputs do not overflow
  // while finding the number of positive, slew-limited braking intervals.
  const long double travel = std::max(0.0, distance);
  const long double change = static_cast<long double>(acceleration) * dt;
  const long double root = std::sqrt(2.L * travel / acceleration) / dt;
  const long double count = std::max(1.L, std::ceil(std::hypot(.5L, root) - .5L));
  const long double speed = (travel / count) / dt + .5L * (count - 1) * change;
  return static_cast<double>(std::min(speed, travel / dt));
}
using detail::interval;
using detail::time_window;
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
    for (Eigen::Index i = 0; i < lo.size(); ++i) {
      lo[i] = std::max(lo[i], -braking_speed(state.q[i] - lower[i], acceleration[i], dt));
      hi[i] = std::min(hi[i], braking_speed(upper[i] - state.q[i], acceleration[i], dt));
      const double scale = std::max({std::abs(lower[i]), std::abs(upper[i]), std::abs(state.q[i])});
      if (std::isfinite(lo[i]) && std::isfinite(hi[i]) && lo[i] > hi[i] &&
          lo[i] - hi[i] <= 16 * std::numeric_limits<double>::epsilon() * scale / dt)
        lo[i] = hi[i] = .5 * lo[i] + .5 * hi[i];
    }
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
  Eigen::LLT<Matrix> cholesky(hessian);
  if (cholesky.info() == Eigen::Success) return;
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
  const auto problem = formulate(state, tasks, dt);
  auto result = solver_->solve(problem);
  if (!result.success()) return {result, std::nullopt};
  if (result.x.size() != state.q.size() || !result.x.allFinite())
    return {{"error", {}, result.iterations, INFINITY, "invalid WBC solution"}, std::nullopt};
  const Vector velocity = result.x.cwiseMax(problem.lower).cwiseMin(problem.upper);
  const double correction = (velocity - result.x).cwiseAbs().maxCoeff();
  if (correction > 1e-6)
    return {{"error", {}, result.iterations, correction, "WBC solution exceeds velocity bounds"}, std::nullopt};
  if (correction > 0) { result.x = velocity; result.violation = 0; }
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
  require(n <= std::numeric_limits<int>::max() / (horizon_ * (horizon_ + 4LL)), "MPC dimensions exceed sparse index capacity");
  const auto size = n * horizon_;
  std::vector<Eigen::Triplet<double>> h, a;
  h.reserve(n * horizon_ * horizon_);
  a.reserve(n * (horizon_ * (horizon_ + 1LL) / 2 + 3LL * horizon_));
  gradient_weights_.resize(horizon_);
  const bool slew = limits_.acceleration.size() != 0;
  const auto rows = (slew ? 3 : 2) * size;
  Vector lo(rows), hi(rows);
  for (int k = 0; k < horizon_; ++k) {
    for (int j = 0; j < horizon_; ++j) {
      const double remaining_weight = terminal_weight_ + (horizon_ - 1 - std::max(k, j)) * tracking_weight_;
      const double value = dt_ * dt_ * remaining_weight + (k == j ? control_weight_ : 0.);
      for (Eigen::Index joint = 0; joint < n; ++joint) {
        h.emplace_back(k * n + joint, j * n + joint, value);
        if (j <= k) a.emplace_back(size + k * n + joint, j * n + joint, dt_);
      }
    }
    const double remaining_weight = terminal_weight_ + (horizon_ - 1 - k) * tracking_weight_;
    gradient_weights_[k] = dt_ * remaining_weight;
    for (Eigen::Index joint = 0; joint < n; ++joint) a.emplace_back(k * n + joint, k * n + joint, 1.);
    lo.segment(k * n, n) = -limits_.velocity;
    hi.segment(k * n, n) = limits_.velocity;
    lo.segment(size + k * n, n) = limits_.lower;
    hi.segment(size + k * n, n) = limits_.upper;
    if (slew) {
      const auto row = 2 * size + k * n;
      for (Eigen::Index joint = 0; joint < n; ++joint) {
        a.emplace_back(row + joint, k * n + joint, 1.);
        if (k > 0) a.emplace_back(row + joint, (k - 1) * n + joint, -1.);
      }
      lo.segment(row, n) = -dt_ * limits_.acceleration;
      hi.segment(row, n) = dt_ * limits_.acceleration;
    }
  }
  require(gradient_weights_.allFinite() && lo.allFinite() && hi.allFinite(), "nonfinite MPC matrices or bounds");
  Eigen::SparseMatrix<double> hessian(size, size), constraint(rows, size);
  hessian.setFromTriplets(h.begin(), h.end()); constraint.setFromTriplets(a.begin(), a.end());
  fixed_ = std::shared_ptr<const detail::FixedQP>(new detail::FixedQP(std::move(hessian), std::move(constraint)));
  lower_ = std::move(lo);
  upper_ = std::move(hi);
}
void JointMPC::reset() { solver_->reset(); last_plan_.reset(); }
std::size_t JointMPC::fixed_matrix_bytes() const {
  return detail::sparse_matrix_bytes(fixed_->hessian) + detail::sparse_matrix_bytes(fixed_->constraint);
}
QuadraticProgram JointMPC::formulate(const RobotState& state, const Vector& goal) const {
  Vector gradient, lo, hi;
  vectors(state, goal, gradient, lo, hi);
  return {Matrix(fixed_->hessian), std::move(gradient), Matrix(fixed_->constraint), std::move(lo), std::move(hi)};
}
void JointMPC::vectors(const RobotState& state, const Vector& goal, Vector& gradient, Vector& lo, Vector& hi) const {
  state.validate();
  time_window(state.stamp, dt_, horizon_);
  const auto n = state.q.size(), size = n * horizon_;
  require(n == limits_.lower.size() && goal.size() == n && goal.allFinite(), "invalid MPC dimensions or goal");
  lo = lower_; hi = upper_;
  for (int k = 0; k < horizon_; ++k) {
    lo.segment(size + k * n, n) -= state.q;
    hi.segment(size + k * n, n) -= state.q;
  }
  if (limits_.acceleration.size()) {
    lo.segment(2 * size, n) += state.v;
    hi.segment(2 * size, n) += state.v;
  }
  gradient.resize(size);
  const Vector delta = state.q - goal;
  for (int k = 0; k < horizon_; ++k) gradient.segment(k * n, n) = gradient_weights_[k] * delta;
  require(gradient.allFinite() && lo.allFinite() && hi.allFinite(), "nonfinite MPC vectors");
}
PlanResult JointMPC::solve(const RobotState& state, const Vector& goal) {
  Vector gradient, lo, hi;
  vectors(state, goal, gradient, lo, hi);
  const auto n = state.q.size();
  Vector warm;
  if (last_plan_ && state.stamp >= last_plan_->stamp) {
    const int offset = static_cast<int>(interval(state.stamp, last_plan_->stamp, dt_, horizon_));
    warm.resize(n * horizon_);
    for (int k = 0; k < horizon_; ++k)
      warm.segment(k * n, n) = last_plan_->velocities.row(std::min(k + offset, horizon_ - 1)).transpose();
  }
  auto* osqp = dynamic_cast<OSQPSolver*>(solver_.get());
  auto result = osqp ? osqp->solve_fixed(fixed_, gradient, lo, hi, warm)
                     : solver_->solve({Matrix(fixed_->hessian), gradient, Matrix(fixed_->constraint), lo, hi}, warm);
  if (!result.success()) return {result, std::nullopt};
  if (result.x.size() != n * horizon_ || !result.x.allFinite())
    return {{"error", {}, result.iterations, INFINITY, "invalid MPC solution"}, std::nullopt};
  JointPlan plan{Matrix(horizon_ + 1, n), Matrix(horizon_, n), state.stamp, dt_};
  plan.positions.row(0) = state.q.transpose();
  for (int k = 0; k < horizon_; ++k) {
    plan.velocities.row(k) = result.x.segment(k * n, n).transpose();
    plan.positions.row(k + 1) = plan.positions.row(k) + dt_ * plan.velocities.row(k);
  }
  const double violation = detail::joint_plan_violation(plan, limits_, state);
  if (violation > 1e-6)
    return {{"error", {}, result.iterations, violation, "reconstructed MPC plan violates joint bounds"}, std::nullopt};
  last_plan_ = plan;
  return {result, plan};
}
}  // namespace wholebodyx
