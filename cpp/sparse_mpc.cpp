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
  std::vector<c_int> column_offsets, row_indices;
  std::vector<c_float> values;
  csc matrix{};

  // The C view points into these buffers; a memberwise copy would leave it stale.
  CscStorage(const CscStorage&) = delete;
  CscStorage& operator=(const CscStorage&) = delete;

  explicit CscStorage(const Eigen::SparseMatrix<double>& input)
      : column_offsets(input.outerIndexPtr(), input.outerIndexPtr() + input.cols() + 1),
        row_indices(input.innerIndexPtr(), input.innerIndexPtr() + input.nonZeros()),
        values(input.valuePtr(), input.valuePtr() + input.nonZeros()) {
    matrix.nzmax = values.size();
    matrix.m = input.rows();
    matrix.n = input.cols();
    matrix.p = column_offsets.data();
    matrix.i = row_indices.data();
    matrix.x = values.data();
    matrix.nz = -1;
  }
};
}  // namespace

struct SparseJointMPC::Impl {
  JointLimits limits;
  int horizon;
  double dt, tracking_weight, control_weight, terminal_weight, position_scale;
  Eigen::Index joint_count, trajectory_size;
  Eigen::SparseMatrix<double> hessian, constraint;
  Eigen::SparseMatrix<double> scaled_hessian, scaled_constraint;
  Vector lower, upper, position_weights;
  std::optional<JointPlan> last_plan;
  OSQPWorkspace* workspace = nullptr;

  Impl(JointLimits bounds, int count, double step, double tracking, double control, double terminal)
      : limits(std::move(bounds)),
        horizon(count),
        dt(step),
        tracking_weight(tracking),
        control_weight(control),
        terminal_weight(terminal),
        joint_count(limits.lower.size()) {
    limits.validate();
    require(horizon > 0, "invalid horizon");
    for (double value : {dt, tracking_weight, control_weight, terminal_weight})
      require(std::isfinite(value) && value > 0,
              "MPC time step and weights must be positive and finite");
    detail::time_window(0, dt, horizon);
    position_scale = std::clamp(dt, .01, 1.);
    // Eigen's default sparse storage indices are int. Check before any product
    // or allocation, including the maximum number of constraint nonzeros.
    require(joint_count <= std::numeric_limits<int>::max() / (8LL * horizon),
            "sparse MPC dimensions exceed index capacity");
    trajectory_size = joint_count * horizon;
    const bool slew = limits.acceleration.size() != 0;
    const auto rows = (slew ? 4 : 3) * trajectory_size;
    std::vector<Eigen::Triplet<double>> hessian_entries, constraint_entries;
    hessian_entries.reserve(2 * trajectory_size);
    constraint_entries.reserve((slew ? 7 : 5) * trajectory_size);
    lower = Vector::Zero(rows);
    upper = Vector::Zero(rows);
    position_weights = Vector::Constant(trajectory_size, tracking_weight);
    position_weights.tail(joint_count).setConstant(terminal_weight);
    for (Eigen::Index i = 0; i < trajectory_size; ++i) {
      const auto joint = i % joint_count;
      hessian_entries.emplace_back(i, i, position_weights[i]);
      hessian_entries.emplace_back(trajectory_size + i, trajectory_size + i, control_weight);
      constraint_entries.emplace_back(i, i, 1.);
      if (i >= joint_count) constraint_entries.emplace_back(i, i - joint_count, -1.);
      constraint_entries.emplace_back(i, trajectory_size + i, -dt);
      constraint_entries.emplace_back(trajectory_size + i, i, 1.);
      constraint_entries.emplace_back(2 * trajectory_size + i, trajectory_size + i, 1.);
      lower[trajectory_size + i] = limits.lower[joint];
      upper[trajectory_size + i] = limits.upper[joint];
      lower[2 * trajectory_size + i] = -limits.velocity[joint];
      upper[2 * trajectory_size + i] = limits.velocity[joint];
      if (slew) {
        constraint_entries.emplace_back(3 * trajectory_size + i, trajectory_size + i, 1.);
        if (i >= joint_count)
          constraint_entries.emplace_back(3 * trajectory_size + i,
                                          trajectory_size + i - joint_count, -1.);
        upper[3 * trajectory_size + i] = dt * limits.acceleration[joint];
        lower[3 * trajectory_size + i] = -upper[3 * trajectory_size + i];
      }
    }
    require(lower.allFinite() && upper.allFinite(), "nonfinite MPC bounds");
    hessian.resize(2 * trajectory_size, 2 * trajectory_size);
    hessian.setFromTriplets(hessian_entries.begin(), hessian_entries.end());
    constraint.resize(rows, 2 * trajectory_size);
    constraint.setFromTriplets(constraint_entries.begin(), constraint_entries.end());
    // Internal x = D*z conditioning; public diagnostics and results retain x.
    scaled_hessian = hessian;
    scaled_constraint = constraint;
    for (Eigen::Index j = 0; j < trajectory_size; ++j) {
      scaled_hessian.coeffRef(j, j) *= position_scale * position_scale;
      for (Eigen::SparseMatrix<double>::InnerIterator it(scaled_constraint, j); it; ++it)
        it.valueRef() *= position_scale;
    }
  }

  ~Impl() {
    if (workspace) osqp_cleanup(workspace);
  }

  void reset() {
    if (workspace) osqp_cleanup(workspace);
    workspace = nullptr;
    last_plan.reset();
  }

  void vectors(const RobotState& state, const Vector& goal, Vector& gradient, Vector& lower_bounds,
               Vector& upper_bounds) const {
    state.validate();
    detail::time_window(state.stamp, dt, horizon);
    require(state.q.size() == joint_count && goal.size() == joint_count && goal.allFinite(),
            "invalid MPC dimensions or goal");
    lower_bounds = lower;
    upper_bounds = upper;
    lower_bounds.head(joint_count) = upper_bounds.head(joint_count) = state.q;
    gradient = Vector::Zero(2 * trajectory_size);
    for (int k = 0; k < horizon; ++k)
      gradient.segment(k * joint_count, joint_count) =
          -position_weights.segment(k * joint_count, joint_count).cwiseProduct(goal);
    if (limits.acceleration.size()) {
      lower_bounds.segment(3 * trajectory_size, joint_count) += state.v;
      upper_bounds.segment(3 * trajectory_size, joint_count) += state.v;
    }
    require(gradient.allFinite() && lower_bounds.allFinite() && upper_bounds.allFinite(),
            "nonfinite MPC vectors");
  }

  QPResult solve_qp(const Vector& gradient, const Vector& lower_bound, const Vector& upper_bound,
                    const Vector& warm, bool* retryable = nullptr) {
    if (retryable) *retryable = false;
    if (warm.size() && !warm.allFinite())
      return {"error", {}, 0, INFINITY, "nonfinite sparse MPC warm start"};
    for (Eigen::Index i = 0; i < lower_bound.size(); ++i)
      if (lower_bound[i] > upper_bound[i])
        return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};
    Vector lower_bounds = lower_bound.cwiseMax(-OSQP_INFTY),
           upper_bounds = upper_bound.cwiseMin(OSQP_INFTY);
    Vector scaled_gradient = gradient, scaled_warm = warm;
    scaled_gradient.head(trajectory_size) *= position_scale;
    if (scaled_warm.size()) scaled_warm.head(trajectory_size) /= position_scale;
    if (!scaled_gradient.allFinite() || !scaled_warm.allFinite())
      return {"error", {}, 0, INFINITY, "nonfinite scaled MPC vectors"};
    c_int error = 0;
    if (!workspace) {
      CscStorage hessian_csc(scaled_hessian), constraint_csc(scaled_constraint);
      OSQPData data{};
      data.n = gradient.size();
      data.m = lower_bounds.size();
      data.P = &hessian_csc.matrix;
      data.A = &constraint_csc.matrix;
      data.q = scaled_gradient.data();
      data.l = lower_bounds.data();
      data.u = upper_bounds.data();
      auto settings = detail::osqp_settings();
      settings.adaptive_rho_interval = 100;
      error = osqp_setup(&workspace, &data, &settings);
    } else {
      error = osqp_update_lin_cost(workspace, scaled_gradient.data());
      if (!error) error = osqp_update_bounds(workspace, lower_bounds.data(), upper_bounds.data());
    }
    if (error) {
      reset();
      return {"error", {}, 0, INFINITY, "OSQP setup/update failed"};
    }
    if (scaled_warm.size() && osqp_warm_start_x(workspace, scaled_warm.data())) {
      reset();
      return {"error", {}, 0, INFINITY, "OSQP warm start failed"};
    }
    error = detail::execute_osqp(workspace);
    QPResult result;
    result.iterations = static_cast<int>(workspace->info->iter);
    result.detail = workspace->info->status;
    const auto status = workspace->info->status_val;
    if (error || status != OSQP_SOLVED) {
      if (retryable && !error)
        *retryable = status == OSQP_SOLVED_INACCURATE || status == OSQP_MAX_ITER_REACHED;
      result.status = detail::rejected_osqp_status(status);
      return result;
    }
    Vector solution = Eigen::Map<const Vector>(workspace->solution->x, 2 * trajectory_size);
    solution.head(trajectory_size) *= position_scale;
    if (!solution.allFinite()) {
      result.detail = "nonfinite solution";
      return result;
    }
    Vector constraint_values = constraint * solution;
    if (!constraint_values.allFinite()) {
      result.detail = "nonfinite constraint evaluation";
      return result;
    }
    result.violation = 0;
    for (Eigen::Index i = 0; i < constraint_values.size(); ++i)
      result.violation = std::max({result.violation, lower_bound[i] - constraint_values[i],
                                   constraint_values[i] - upper_bound[i]});
    if (result.violation > 1e-6) {
      result.detail = "solution exceeds absolute constraint tolerance";
      return result;
    }
    result.status = "solved";
    result.x = std::move(solution);
    return result;
  }
};

SparseJointMPC::SparseJointMPC(JointLimits limits, int horizon, double dt, double tracking_weight,
                               double control_weight, double terminal_weight)
    : impl_(std::make_unique<Impl>(std::move(limits), horizon, dt, tracking_weight, control_weight,
                                   terminal_weight)) {}
SparseJointMPC::~SparseJointMPC() = default;
void SparseJointMPC::reset() { impl_->reset(); }

SparseMPCProblem SparseJointMPC::formulate(const RobotState& state, const Vector& goal) const {
  Vector gradient, lower_bounds, upper_bounds;
  impl_->vectors(state, goal, gradient, lower_bounds, upper_bounds);
  return {impl_->hessian, impl_->constraint, std::move(gradient), std::move(lower_bounds),
          std::move(upper_bounds)};
}

PlanResult SparseJointMPC::solve(const RobotState& state, const Vector& goal) {
  auto& planner = *impl_;
  Vector gradient, lower_bounds, upper_bounds;
  planner.vectors(state, goal, gradient, lower_bounds, upper_bounds);
  Vector warm;
  if (planner.last_plan && state.stamp >= planner.last_plan->stamp) {
    const auto offset =
        detail::interval(state.stamp, planner.last_plan->stamp, planner.dt, planner.horizon);
    warm.resize(2 * planner.trajectory_size);
    Vector position = state.q;
    for (int k = 0; k < planner.horizon; ++k) {
      const auto velocity =
          planner.last_plan->velocities.row(std::min<Eigen::Index>(k + offset, planner.horizon - 1))
              .transpose();
      position += planner.dt * velocity;
      warm.segment(k * planner.joint_count, planner.joint_count) = position;
      warm.segment(planner.trajectory_size + k * planner.joint_count, planner.joint_count) =
          velocity;
    }
  }
  const bool reused_workspace = planner.workspace != nullptr;
  bool retryable = false;
  auto result = planner.solve_qp(gradient, lower_bounds, upper_bounds, warm, &retryable);
  if (reused_workspace && retryable) {
    // A badly conditioned warm workspace may retain unsuitable dual/rho state.
    // Retry once from cold; inaccurate candidates never become executable plans.
    const auto first_iterations = result.iterations;
    const auto first_status = result.detail;
    planner.reset();
    result = planner.solve_qp(gradient, lower_bounds, upper_bounds, Vector());
    result.iterations += first_iterations;
    result.detail = "cold retry after " + first_status + "; " + result.detail;
  }
  if (!result.success()) return {std::move(result), std::nullopt};
  JointPlan plan{Matrix(planner.horizon + 1, planner.joint_count),
                 Matrix(planner.horizon, planner.joint_count), state.stamp, planner.dt};
  plan.positions.row(0) = state.q.transpose();
  for (int k = 0; k < planner.horizon; ++k) {
    const auto velocity =
        result.x.segment(planner.trajectory_size + k * planner.joint_count, planner.joint_count);
    plan.velocities.row(k) = velocity.transpose();
    plan.positions.row(k + 1) = plan.positions.row(k) + planner.dt * velocity.transpose();
  }
  const double violation = detail::joint_plan_violation(plan, planner.limits, state);
  if (violation > 1e-6)
    return {{"error",
             {},
             result.iterations,
             violation,
             "reconstructed sparse MPC plan violates joint bounds"},
            std::nullopt};
  planner.last_plan = plan;
  return {std::move(result), std::move(plan)};
}

}  // namespace wholebodyx
