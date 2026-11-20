#include "wholebodyx/control.hpp"
#include "wholebodyx/sparse_mpc.hpp"
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
  require(q.size() > 0 && q.size() == v.size() && q.allFinite() && v.allFinite(),
          "invalid fixed-base state");
  require(std::isfinite(stamp) && stamp >= 0, "invalid timestamp");
}
void JointLimits::validate() const {
  const auto n = lower.size();
  require(n > 0 && upper.size() == n && velocity.size() == n, "invalid joint limit dimensions");
  require(lower.allFinite() && upper.allFinite() && velocity.allFinite(), "nonfinite joint limits");
  require((lower.array() < upper.array()).all() && (velocity.array() > 0).all(),
          "invalid joint limits");
  require(acceleration.size() == 0 || (acceleration.size() == n && acceleration.allFinite() &&
                                       (acceleration.array() > 0).all()),
          "invalid acceleration limits");
}
std::pair<Vector, Vector> JointLimits::velocity_bounds(const RobotState& state, double dt) const {
  validate();
  state.validate();
  positive(dt, "dt must be positive and finite");
  require(state.q.size() == lower.size(), "state and limits dimensions differ");
  Vector velocity_lower = (-velocity).cwiseMax((lower - state.q) / dt);
  Vector velocity_upper = velocity.cwiseMin((upper - state.q) / dt);
  if (acceleration.size()) {
    velocity_lower = velocity_lower.cwiseMax(state.v - acceleration * dt);
    velocity_upper = velocity_upper.cwiseMin(state.v + acceleration * dt);
    for (Eigen::Index i = 0; i < velocity_lower.size(); ++i) {
      // A conservative stopping distance proves that the exact discrete
      // envelope cannot tighten a joint comfortably inside its limits.
      // Extended precision keeps this shortcut valid at extreme scales.
      const long double speed = velocity[i];
      const long double travel = speed * (speed / acceleration[i] + dt);
      const double below = state.q[i] - lower[i], above = upper[i] - state.q[i];
      const bool zero_slew = acceleration[i] * dt == 0.;
      if (velocity_lower[i] < 0 && (below < travel || zero_slew))
        velocity_lower[i] = std::max(
            velocity_lower[i], -braking_speed(below, acceleration[i], dt));
      if (velocity_upper[i] > 0 && (above < travel || zero_slew))
        velocity_upper[i] = std::min(
            velocity_upper[i], braking_speed(above, acceleration[i], dt));
      const double scale = std::max({std::abs(lower[i]), std::abs(upper[i]), std::abs(state.q[i])});
      if (std::isfinite(velocity_lower[i]) && std::isfinite(velocity_upper[i]) &&
          velocity_lower[i] > velocity_upper[i] &&
          velocity_lower[i] - velocity_upper[i] <=
              16 * std::numeric_limits<double>::epsilon() * scale / dt)
        velocity_lower[i] = velocity_upper[i] = .5 * velocity_lower[i] + .5 * velocity_upper[i];
    }
  }
  return {std::move(velocity_lower), std::move(velocity_upper)};
}
void LinearTask::validate(Eigen::Index joints) const {
  require(jacobian.cols() == joints && jacobian.rows() > 0 &&
              target_velocity.size() == jacobian.rows() && weight.size() == jacobian.rows(),
          "invalid task dimensions");
  require(jacobian.allFinite() && target_velocity.allFinite() && weight.allFinite() &&
              (weight.array() >= 0).all(),
          "invalid task values");
}
void QuadraticProgram::validate() const { validate_against(nullptr); }
void QuadraticProgram::validate_against(const Matrix* previous) const {
  const auto n = gradient.size(), m = lower.size();
  require(n > 0 && hessian.rows() == n && hessian.cols() == n && constraint.rows() == m &&
              constraint.cols() == n && upper.size() == m,
          "invalid QP dimensions");
  require(hessian.allFinite() && gradient.allFinite() && constraint.allFinite() &&
              !lower.array().isNaN().any() && !upper.array().isNaN().any(),
          "invalid QP values");
  require((hessian - hessian.transpose()).cwiseAbs().maxCoeff() <= 1e-10, "nonsymmetric Hessian");
  if (previous && previous->rows() == n && previous->cols() == n &&
      (previous->array() == hessian.array()).all())
    return;  // Only an exactly equal Hessian reuses the previous convexity check.
  Eigen::LLT<Matrix> cholesky(hessian);
  if (cholesky.info() == Eigen::Success) return;
  Eigen::SelfAdjointEigenSolver<Matrix> eig(hessian, Eigen::EigenvaluesOnly);
  require(eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() >= -1e-10, "nonconvex QP");
}

KinematicWBC::KinematicWBC(JointLimits limits, double regularization, double smoothing,
                           std::shared_ptr<QPSolver> solver)
    : limits_(std::move(limits)),
      regularization_(regularization),
      smoothing_(smoothing),
      solver_(solver ? std::move(solver) : std::make_shared<OSQPSolver>()) {
  limits_.validate();
  positive(regularization_, "invalid regularization");
  require(std::isfinite(smoothing_) && smoothing_ >= 0, "invalid smoothing");
}
void KinematicWBC::reset() { solver_->reset(); }
QuadraticProgram KinematicWBC::formulate(const RobotState& state,
                                         const std::vector<LinearTask>& tasks, double dt) const {
  auto bounds = limits_.velocity_bounds(state, dt);
  const auto joint_count = state.q.size();
  Matrix hessian = (regularization_ + smoothing_) * Matrix::Identity(joint_count, joint_count);
  Vector gradient = -smoothing_ * state.v;
  for (const auto& task : tasks) {
    task.validate(joint_count);
    hessian.noalias() += task.jacobian.transpose() * task.weight.asDiagonal() * task.jacobian;
    gradient.noalias() -=
        task.jacobian.transpose() * task.weight.asDiagonal() * task.target_velocity;
  }
  return {std::move(hessian), std::move(gradient), Matrix::Identity(joint_count, joint_count),
          std::move(bounds.first), std::move(bounds.second)};
}
ControlResult KinematicWBC::step(const RobotState& state, const std::vector<LinearTask>& tasks,
                                 double dt) {
  const auto problem = formulate(state, tasks, dt);
  auto result = solver_->solve(problem);
  if (!result.success()) return {std::move(result), std::nullopt};
  if (result.x.size() != state.q.size() || !result.x.allFinite())
    return {{"error", {}, result.iterations, INFINITY, "invalid WBC solution"}, std::nullopt};
  if ((problem.lower.array() > problem.upper.array()).any())
    return {{"error", {}, result.iterations, INFINITY, "inconsistent WBC velocity bounds"},
            std::nullopt};
  // Normalize tiny residuals so active braking bounds remain exact on the next cycle.
  const Vector velocity = result.x.cwiseMax(problem.lower).cwiseMin(problem.upper);
  const double correction = (velocity - result.x).cwiseAbs().maxCoeff();
  if (correction > 1e-6)
    return {{"error", {}, result.iterations, correction, "WBC solution exceeds velocity bounds"},
            std::nullopt};
  if (correction > 0) {
    result.x = velocity;
    result.violation = 0;
  }
  VelocityCommand command{result.x, state.stamp, state.stamp + dt};
  return {std::move(result), std::move(command)};
}

double JointPlan::valid_until() const { return stamp + dt * velocities.rows(); }
void JointPlan::validate() const {
  positive(dt, "invalid plan time step");
  require(velocities.rows() > 0 && velocities.cols() > 0 &&
              positions.rows() == velocities.rows() + 1 && positions.cols() == velocities.cols() &&
              positions.allFinite() && velocities.allFinite(),
          "invalid plan shape");
  time_window(stamp, dt, velocities.rows());
  const auto expected = dt * velocities;
  const auto difference =
      positions.bottomRows(velocities.rows()) - positions.topRows(velocities.rows());
  // Finite velocities can overflow dt*v; inf <= inf cannot certify dynamics.
  require((expected.array().isFinite() &&
           ((difference - expected).array().abs() <= 1e-8 + 1e-7 * expected.array().abs())).all(),
          "plan positions and velocities must satisfy integrator dynamics");
}
std::pair<Vector, Vector> JointPlan::sample(double time) const {
  validate();
  return detail::sample_validated_joint_plan(*this, time);
}

JointMPC::JointMPC(JointLimits limits, int horizon, double dt, double tracking_weight,
                   double control_weight, double terminal_weight, std::shared_ptr<QPSolver> solver)
    : limits_(std::move(limits)),
      horizon_(horizon),
      dt_(dt),
      tracking_weight_(tracking_weight),
      control_weight_(control_weight),
      terminal_weight_(terminal_weight),
      solver_(solver ? solver : std::make_shared<OSQPSolver>()) {
  limits_.validate();
  require(horizon_ > 0, "invalid horizon");
  positive(dt_, "invalid MPC dt");
  positive(tracking_weight_, "invalid tracking weight");
  positive(control_weight_, "invalid control weight");
  positive(terminal_weight_, "invalid terminal weight");
  time_window(0, dt_, horizon_);
  const auto joint_count = limits_.lower.size();
  require(joint_count <= std::numeric_limits<int>::max() / (horizon_ * (horizon_ + 4LL)),
          "MPC dimensions exceed sparse index capacity");
  const auto trajectory_size = joint_count * horizon_;
  std::vector<Eigen::Triplet<double>> hessian_entries, constraint_entries;
  hessian_entries.reserve(joint_count * horizon_ * horizon_);
  constraint_entries.reserve(joint_count * (horizon_ * (horizon_ + 1LL) / 2 + 3LL * horizon_));
  gradient_weights_.resize(horizon_);
  Vector position_cost(horizon_);
  for (int k = 0; k < horizon_; ++k) {
    const double weight = terminal_weight_ + (horizon_ - 1 - k) * tracking_weight_;
    gradient_weights_[k] = dt_ * weight;
    position_cost[k] = dt_ * dt_ * weight;
  }
  const bool slew = limits_.acceleration.size() != 0;
  // Constraint rows: velocity boxes, predicted position boxes, then optional slew.
  const auto rows = (slew ? 3 : 2) * trajectory_size;
  Vector lower_bounds(rows), upper_bounds(rows);
  for (int k = 0; k < horizon_; ++k) {
    for (int j = 0; j < horizon_; ++j) {
      const double value = position_cost[std::max(k, j)] + (k == j ? control_weight_ : 0.);
      for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
        hessian_entries.emplace_back(k * joint_count + joint, j * joint_count + joint, value);
        if (j <= k)
          constraint_entries.emplace_back(trajectory_size + k * joint_count + joint,
                                          j * joint_count + joint, dt_);
      }
    }
    for (Eigen::Index joint = 0; joint < joint_count; ++joint)
      constraint_entries.emplace_back(k * joint_count + joint, k * joint_count + joint, 1.);
    lower_bounds.segment(k * joint_count, joint_count) = -limits_.velocity;
    upper_bounds.segment(k * joint_count, joint_count) = limits_.velocity;
    lower_bounds.segment(trajectory_size + k * joint_count, joint_count) = limits_.lower;
    upper_bounds.segment(trajectory_size + k * joint_count, joint_count) = limits_.upper;
    if (slew) {
      const auto row = 2 * trajectory_size + k * joint_count;
      for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
        constraint_entries.emplace_back(row + joint, k * joint_count + joint, 1.);
        if (k > 0) constraint_entries.emplace_back(row + joint, (k - 1) * joint_count + joint, -1.);
      }
      lower_bounds.segment(row, joint_count) = -dt_ * limits_.acceleration;
      upper_bounds.segment(row, joint_count) = dt_ * limits_.acceleration;
    }
  }
  require(gradient_weights_.allFinite() && lower_bounds.allFinite() && upper_bounds.allFinite(),
          "nonfinite MPC matrices or bounds");
  Eigen::SparseMatrix<double> hessian(trajectory_size, trajectory_size),
      constraint(rows, trajectory_size);
  hessian.setFromTriplets(hessian_entries.begin(), hessian_entries.end());
  constraint.setFromTriplets(constraint_entries.begin(), constraint_entries.end());
  fixed_ = std::shared_ptr<const detail::FixedQP>(
      new detail::FixedQP(std::move(hessian), std::move(constraint)));
  if (!solver && gradient_weights_[0] > 0.) {
    const Matrix temporal = Matrix::NullaryExpr(
        horizon_, horizon_, [&](Eigen::Index k, Eigen::Index j) {
          return position_cost[std::max(k, j)] + (k == j ? control_weight_ : 0.);
        });
    const double rounding = 8. * horizon_ * std::numeric_limits<double>::epsilon() *
                            temporal.rowwise().sum().maxCoeff();
    // Use the same per-joint Hessian as OSQP. When roundoff can dominate the
    // regularization, leave the solve to the configured adapter.
    if (control_weight_ > std::max(std::numeric_limits<double>::min(), rounding)) {
      Eigen::LLT<Matrix> factor(temporal);
      if (factor.info() == Eigen::Success) {
        Vector gain = factor.solve(-gradient_weights_ / gradient_weights_[0]);
        if (gain.allFinite()) unconstrained_gain_ = std::move(gain);
      }
    }
  }
  lower_ = std::move(lower_bounds);
  upper_ = std::move(upper_bounds);
  prefer_sparse_ = !solver && detail::prefer_slew_variables(limits_, dt_, horizon_);
}
void JointMPC::reset() {
  solver_->reset();
  last_plan_.reset();
  if (short_step_planner_) short_step_planner_->reset();
}
std::size_t JointMPC::fixed_matrix_bytes() const {
  const auto bytes = detail::sparse_matrix_bytes(fixed_->hessian) +
                     detail::sparse_matrix_bytes(fixed_->constraint);
  return bytes + (short_step_planner_ ? short_step_planner_->fixed_matrix_bytes() : 0);
}
QuadraticProgram JointMPC::formulate(const RobotState& state, const Vector& goal) const {
  Vector gradient, lower_bounds, upper_bounds;
  vectors(state, goal, gradient, lower_bounds, upper_bounds);
  return {Matrix(fixed_->hessian), std::move(gradient), Matrix(fixed_->constraint),
          std::move(lower_bounds), std::move(upper_bounds)};
}
void JointMPC::vectors(const RobotState& state, const Vector& goal, Vector& gradient,
                       Vector& lower_bounds, Vector& upper_bounds) const {
  state.validate();
  time_window(state.stamp, dt_, horizon_);
  const auto joint_count = state.q.size(), trajectory_size = joint_count * horizon_;
  require(joint_count == limits_.lower.size() && goal.size() == joint_count && goal.allFinite(),
          "invalid MPC dimensions or goal");
  lower_bounds = lower_;
  upper_bounds = upper_;
  for (int k = 0; k < horizon_; ++k) {
    lower_bounds.segment(trajectory_size + k * joint_count, joint_count) -= state.q;
    upper_bounds.segment(trajectory_size + k * joint_count, joint_count) -= state.q;
  }
  if (limits_.acceleration.size()) {
    lower_bounds.segment(2 * trajectory_size, joint_count) += state.v;
    upper_bounds.segment(2 * trajectory_size, joint_count) += state.v;
  }
  gradient.resize(trajectory_size);
  const Vector delta = state.q - goal;
  for (int k = 0; k < horizon_; ++k)
    gradient.segment(k * joint_count, joint_count) = gradient_weights_[k] * delta;
  require(gradient.allFinite() && lower_bounds.allFinite() && upper_bounds.allFinite(),
          "nonfinite MPC vectors");
}
Vector JointMPC::unconstrained_solution(const Vector& gradient, const Vector& lower,
                                       const Vector& upper) const {
  if (!unconstrained_gain_.size()) return {};
  const auto joints = limits_.lower.size();
  const auto first = unconstrained_gain_[0] * gradient.head(joints);
  if (limits_.acceleration.size()) {
    const auto slew = 2 * gradient.size();
    if (!(first.array() >= lower.segment(slew, joints).array()).all() ||
        !(first.array() <= upper.segment(slew, joints).array()).all())
      return {};
  }
  if (!(first.array() >= lower.head(joints).array()).all() ||
      !(first.array() <= upper.head(joints).array()).all())
    return {};
  Vector candidate(gradient.size());
  for (int k = 0; k < horizon_; ++k)
    candidate.segment(k * joints, joints) = unconstrained_gain_[k] * gradient.head(joints);
  const Vector values = fixed_->constraint * candidate;
  if (!values.allFinite() || !(values.array() >= lower.array()).all() ||
      !(values.array() <= upper.array()).all())
    return {};
  return candidate;
}
PlanResult JointMPC::solve_with_sparse_model(const RobotState& state, const Vector& goal,
                                             const Vector& lower, const Vector& upper) {
  PlanResult outcome;
  try {
    if (!short_step_planner_)
      short_step_planner_ = std::make_shared<SparseJointMPC>(
          limits_, horizon_, dt_, tracking_weight_, control_weight_, terminal_weight_);
    outcome = short_step_planner_->solve(state, goal);
  } catch (const std::invalid_argument& error) {
    return {{"error", {}, 0, INFINITY, error.what()}, std::nullopt};
  }
  if (!outcome.plan) return outcome;
  auto& result = outcome.solver;
  const auto size = state.q.size() * horizon_;
  Vector velocities = result.x.tail(size);
  const Vector values = fixed_->constraint * velocities;
  if (!values.allFinite())
    return {{"error", {}, result.iterations, INFINITY, "nonfinite constraint evaluation"},
            std::nullopt};
  result.violation = std::max({0., (lower - values).maxCoeff(), (values - upper).maxCoeff()});
  if (result.violation > 1e-6)
    return {{"error", {}, result.iterations, result.violation,
             "solution exceeds absolute constraint tolerance"}, std::nullopt};
  result.x = std::move(velocities);
  return reconstruct_plan(state, std::move(result));
}

PlanResult JointMPC::solve(const RobotState& state, const Vector& goal) {
  Vector gradient, lower_bounds, upper_bounds;
  vectors(state, goal, gradient, lower_bounds, upper_bounds);
  Vector unconstrained = unconstrained_solution(gradient, lower_bounds, upper_bounds);
  if (unconstrained.size()) {
    auto direct = reconstruct_plan(
        state, {"solved", std::move(unconstrained), 0, 0., "unconstrained optimum"});
    if (direct.plan) {
      try {
        direct.plan->validate();
      } catch (const std::invalid_argument&) {
        direct.plan.reset();
      }
      if (direct.plan) {
        last_plan_ = *direct.plan;
        return direct;
      }
    }
  }
  std::optional<QPResult> sparse_failure;
  if (prefer_sparse_) {
    auto attempt = solve_with_sparse_model(state, goal, lower_bounds, upper_bounds);
    if (attempt.plan || attempt.solver.status == "infeasible") {
      if (attempt.plan) last_plan_ = *attempt.plan;
      return attempt;
    }
    sparse_failure = std::move(attempt.solver);
  }
  const auto joint_count = state.q.size();
  Vector warm;
  if (last_plan_ && state.stamp >= last_plan_->stamp) {
    const int offset = static_cast<int>(interval(state.stamp, last_plan_->stamp, dt_, horizon_));
    warm.resize(joint_count * horizon_);
    for (int k = 0; k < horizon_; ++k)
      warm.segment(k * joint_count, joint_count) =
          last_plan_->velocities.row(std::min(k + offset, horizon_ - 1)).transpose();
  }
  auto* osqp = dynamic_cast<OSQPSolver*>(solver_.get());
  auto outcome = reconstruct_plan(
      state, osqp ? osqp->solve_fixed(fixed_, gradient, lower_bounds, upper_bounds, warm)
                  : solver_->solve({Matrix(fixed_->hessian), gradient,
                                    Matrix(fixed_->constraint), lower_bounds, upper_bounds},
                                   warm));
  if (sparse_failure) {
    outcome.solver.iterations += sparse_failure->iterations;
    outcome.solver.detail = "condensed retry after sparse solve: " + outcome.solver.detail;
  }
  if (outcome.plan) last_plan_ = *outcome.plan;
  return outcome;
}
PlanResult JointMPC::reconstruct_plan(const RobotState& state, QPResult result) const {
  const auto joint_count = state.q.size();
  if (!result.success()) return {std::move(result), std::nullopt};
  if (result.x.size() != joint_count * horizon_ || !result.x.allFinite())
    return {{"error", {}, result.iterations, INFINITY, "invalid MPC solution"}, std::nullopt};
  JointPlan plan{Matrix(horizon_ + 1, joint_count), Matrix(horizon_, joint_count), state.stamp,
                 dt_};
  plan.positions.row(0) = state.q.transpose();
  for (int k = 0; k < horizon_; ++k) {
    plan.velocities.row(k) = result.x.segment(k * joint_count, joint_count).transpose();
    plan.positions.row(k + 1) = plan.positions.row(k) + dt_ * plan.velocities.row(k);
  }
  const double violation = detail::joint_plan_violation(plan, limits_, state);
  if (violation > 1e-6)
    return {
        {"error", {}, result.iterations, violation, "reconstructed MPC plan violates joint bounds"},
        std::nullopt};
  return {std::move(result), std::move(plan)};
}
}  // namespace wholebodyx
