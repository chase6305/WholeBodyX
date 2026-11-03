#include "wholebodyx/sparse_mpc.hpp"
#include "time_utils.hpp"
#include "plan_validation.hpp"
#include "osqp_execution.hpp"

#include <osqp/osqp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace wholebodyx {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

struct CscStorage {
  std::vector<c_int> columns, rows;
  std::vector<c_float> values;
  csc matrix{};
  explicit CscStorage(const Eigen::SparseMatrix<double>& input)
      : columns(input.outerIndexPtr(), input.outerIndexPtr() + input.cols() + 1),
        rows(input.innerIndexPtr(), input.innerIndexPtr() + input.nonZeros()),
        values(input.valuePtr(), input.valuePtr() + input.nonZeros()) {
    matrix.nzmax = values.size(); matrix.m = input.rows(); matrix.n = input.cols();
    matrix.p = columns.data(); matrix.i = rows.data(); matrix.x = values.data(); matrix.nz = -1;
  }
};
}  // namespace

struct SparseJointMPC::Impl {
  JointLimits limits;
  int horizon;
  double dt, tracking_weight, control_weight, terminal_weight, position_scale;
  Eigen::Index n, size;
  Eigen::SparseMatrix<double> hessian, constraint;
  Eigen::SparseMatrix<double> scaled_hessian, scaled_constraint;
  Vector lower, upper, position_weights;
  std::optional<JointPlan> last_plan;
  OSQPWorkspace* work = nullptr;

  Impl(JointLimits bounds, int count, double step, double tracking, double control, double terminal)
      : limits(std::move(bounds)), horizon(count), dt(step), tracking_weight(tracking),
        control_weight(control), terminal_weight(terminal), n(limits.lower.size()) {
    limits.validate();
    require(horizon > 0, "invalid horizon");
    for (double value : {dt, tracking_weight, control_weight, terminal_weight})
      require(std::isfinite(value) && value > 0, "MPC time step and weights must be positive and finite");
    detail::time_window(0, dt, horizon);
    position_scale = std::clamp(dt, .01, 1.);
    // Eigen's default sparse storage indices are int. Check before any product
    // or allocation, including the maximum number of constraint nonzeros.
    require(n <= std::numeric_limits<int>::max() / (8LL * horizon), "sparse MPC dimensions exceed index capacity");
    size = n * horizon;
    const bool slew = limits.acceleration.size() != 0;
    const auto rows = (slew ? 4 : 3) * size;
    std::vector<Eigen::Triplet<double>> h, a;
    h.reserve(2 * size); a.reserve((slew ? 7 : 5) * size);
    lower = Vector::Zero(rows); upper = Vector::Zero(rows);
    position_weights = Vector::Constant(size, tracking_weight);
    position_weights.tail(n).setConstant(terminal_weight);
    for (Eigen::Index i = 0; i < size; ++i) {
      const auto joint = i % n;
      h.emplace_back(i, i, position_weights[i]);
      h.emplace_back(size + i, size + i, control_weight);
      a.emplace_back(i, i, 1.);
      if (i >= n) a.emplace_back(i, i - n, -1.);
      a.emplace_back(i, size + i, -dt);
      a.emplace_back(size + i, i, 1.);
      a.emplace_back(2 * size + i, size + i, 1.);
      lower[size + i] = limits.lower[joint]; upper[size + i] = limits.upper[joint];
      lower[2 * size + i] = -limits.velocity[joint]; upper[2 * size + i] = limits.velocity[joint];
      if (slew) {
        a.emplace_back(3 * size + i, size + i, 1.);
        if (i >= n) a.emplace_back(3 * size + i, size + i - n, -1.);
        upper[3 * size + i] = dt * limits.acceleration[joint];
        lower[3 * size + i] = -upper[3 * size + i];
      }
    }
    require(lower.allFinite() && upper.allFinite(), "nonfinite MPC bounds");
    hessian.resize(2 * size, 2 * size); hessian.setFromTriplets(h.begin(), h.end());
    constraint.resize(rows, 2 * size); constraint.setFromTriplets(a.begin(), a.end());
    // Internal x = D*z conditioning; public diagnostics and results retain x.
    scaled_hessian = hessian; scaled_constraint = constraint;
    for (Eigen::Index j = 0; j < size; ++j) {
      scaled_hessian.coeffRef(j, j) *= position_scale * position_scale;
      for (Eigen::SparseMatrix<double>::InnerIterator it(scaled_constraint, j); it; ++it)
        it.valueRef() *= position_scale;
    }
  }

  ~Impl() { if (work) osqp_cleanup(work); }

  void reset() {
    if (work) osqp_cleanup(work);
    work = nullptr;
    last_plan.reset();
  }

  void vectors(const RobotState& state, const Vector& goal, Vector& g, Vector& lo, Vector& hi) const {
    state.validate();
    detail::time_window(state.stamp, dt, horizon);
    require(state.q.size() == n && goal.size() == n && goal.allFinite(), "invalid MPC dimensions or goal");
    lo = lower; hi = upper;
    lo.head(n) = hi.head(n) = state.q;
    g = Vector::Zero(2 * size);
    for (int k = 0; k < horizon; ++k)
      g.segment(k * n, n) = -position_weights.segment(k * n, n).cwiseProduct(goal);
    if (limits.acceleration.size()) {
      lo.segment(3 * size, n) += state.v;
      hi.segment(3 * size, n) += state.v;
    }
    require(g.allFinite() && lo.allFinite() && hi.allFinite(), "nonfinite MPC vectors");
  }

  QPResult solve_qp(Vector& gradient, const Vector& lower_bound, const Vector& upper_bound, const Vector& warm, bool* retryable = nullptr) {
    if (retryable) *retryable = false;
    if (warm.size() && !warm.allFinite())
      return {"error", {}, 0, INFINITY, "nonfinite sparse MPC warm start"};
    for (Eigen::Index i = 0; i < lower_bound.size(); ++i)
      if (lower_bound[i] > upper_bound[i])
        return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};
    Vector lo = lower_bound.cwiseMax(-OSQP_INFTY), hi = upper_bound.cwiseMin(OSQP_INFTY);
    Vector q = gradient, scaled_warm = warm;
    q.head(size) *= position_scale;
    if (scaled_warm.size()) scaled_warm.head(size) /= position_scale;
    if (!q.allFinite() || !scaled_warm.allFinite())
      return {"error", {}, 0, INFINITY, "nonfinite scaled MPC vectors"};
    c_int error = 0;
    if (!work) {
      CscStorage h(scaled_hessian), a(scaled_constraint);
      OSQPData data{};
      data.n = gradient.size(); data.m = lo.size(); data.P = &h.matrix; data.A = &a.matrix;
      data.q = q.data(); data.l = lo.data(); data.u = hi.data();
      auto settings = detail::osqp_settings();
      settings.adaptive_rho_interval = 100;
      error = osqp_setup(&work, &data, &settings);
    } else {
      error = osqp_update_lin_cost(work, q.data());
      if (!error) error = osqp_update_bounds(work, lo.data(), hi.data());
    }
    if (error) { reset(); return {"error", {}, 0, INFINITY, "OSQP setup/update failed"}; }
    if (scaled_warm.size() && osqp_warm_start_x(work, scaled_warm.data())) {
      reset(); return {"error", {}, 0, INFINITY, "OSQP warm start failed"};
    }
    error = detail::execute_osqp(work);
    QPResult result;
    result.iterations = static_cast<int>(work->info->iter);
    result.detail = work->info->status;
    const auto status = work->info->status_val;
    if (error || status != OSQP_SOLVED) {
      if (retryable && !error)
        *retryable = status == OSQP_SOLVED_INACCURATE || status == OSQP_MAX_ITER_REACHED;
      result.status = detail::rejected_osqp_status(status);
      return result;
    }
    Vector x = Eigen::Map<const Vector>(work->solution->x, 2 * size);
    x.head(size) *= position_scale;
    if (!x.allFinite()) { result.detail = "nonfinite solution"; return result; }
    Vector ax = constraint * x;
    if (!ax.allFinite()) { result.detail = "nonfinite constraint evaluation"; return result; }
    result.violation = 0;
    for (Eigen::Index i = 0; i < ax.size(); ++i)
      result.violation = std::max({result.violation, lower_bound[i] - ax[i], ax[i] - upper_bound[i]});
    if (result.violation > 1e-6) { result.detail = "solution exceeds absolute constraint tolerance"; return result; }
    result.status = "solved";
    result.x = std::move(x);
    return result;
  }
};

SparseJointMPC::SparseJointMPC(JointLimits limits, int horizon, double dt, double tracking_weight,
                               double control_weight, double terminal_weight)
    : impl_(std::make_unique<Impl>(std::move(limits), horizon, dt, tracking_weight, control_weight, terminal_weight)) {}
SparseJointMPC::~SparseJointMPC() = default;
void SparseJointMPC::reset() { impl_->reset(); }

SparseMPCProblem SparseJointMPC::formulate(const RobotState& state, const Vector& goal) const {
  Vector gradient, lo, hi;
  impl_->vectors(state, goal, gradient, lo, hi);
  return {impl_->hessian, impl_->constraint, std::move(gradient), std::move(lo), std::move(hi)};
}

PlanResult SparseJointMPC::solve(const RobotState& state, const Vector& goal) {
  auto& p = *impl_;
  Vector gradient, lo, hi;
  p.vectors(state, goal, gradient, lo, hi);
  Vector warm;
  if (p.last_plan && state.stamp >= p.last_plan->stamp) {
    const auto offset = detail::interval(state.stamp, p.last_plan->stamp, p.dt, p.horizon);
    warm.resize(2 * p.size);
    Vector position = state.q;
    for (int k = 0; k < p.horizon; ++k) {
      Vector velocity = p.last_plan->velocities.row(std::min<Eigen::Index>(k + offset, p.horizon - 1)).transpose();
      position += p.dt * velocity;
      warm.segment(k * p.n, p.n) = position;
      warm.segment(p.size + k * p.n, p.n) = velocity;
    }
  }
  const bool reused_workspace = p.work != nullptr;
  bool retryable = false;
  auto result = p.solve_qp(gradient, lo, hi, warm, &retryable);
  if (reused_workspace && retryable) {
    // A badly conditioned warm workspace may retain unsuitable dual/rho state.
    // Retry once from cold; inaccurate candidates never become executable plans.
    const auto first_iterations = result.iterations;
    const auto first_status = result.detail;
    p.reset();
    result = p.solve_qp(gradient, lo, hi, Vector());
    result.iterations += first_iterations;
    result.detail = "cold retry after " + first_status + "; " + result.detail;
  }
  if (!result.success()) return {std::move(result), std::nullopt};
  JointPlan plan{Matrix(p.horizon + 1, p.n), Matrix(p.horizon, p.n), state.stamp, p.dt};
  plan.positions.row(0) = state.q.transpose();
  for (int k = 0; k < p.horizon; ++k) {
    const Vector velocity = result.x.segment(p.size + k * p.n, p.n);
    plan.velocities.row(k) = velocity.transpose();
    plan.positions.row(k + 1) = plan.positions.row(k) + p.dt * velocity.transpose();
  }
  const double violation = detail::joint_plan_violation(plan, p.limits, state);
  if (violation > 1e-6)
    return {{"error", {}, result.iterations, violation, "reconstructed sparse MPC plan violates joint bounds"}, std::nullopt};
  p.last_plan = plan;
  return {std::move(result), std::move(plan)};
}

}  // namespace wholebodyx
