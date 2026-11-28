#include "wholebodyx/sparse_mpc.hpp"
#include "fixed_qp.hpp"
#include "time_utils.hpp"
#include "plan_validation.hpp"
#include "osqp_execution.hpp"

#include <osqp/osqp.h>
#include <algorithm>
#include <array>
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
  struct PrefixTrial {
    int horizon = 0;
    double terminal_weight = 0.;
    Vector travel;
    std::unique_ptr<SparseJointMPC> planner;
  };

  JointLimits limits;
  int horizon;
  double dt, tracking_weight, control_weight, terminal_weight, position_scale;
  Eigen::Index joint_count, trajectory_size;
  Eigen::SparseMatrix<double> hessian, constraint;
  Eigen::SparseMatrix<double> scaled_hessian, scaled_constraint;
  Eigen::SparseMatrix<double> slew_hessian, slew_constraint;
  Vector lower, upper, position_weights, position_origin;
  Vector tail_feedback;
  std::array<PrefixTrial, 3> prefixes;
  std::optional<JointPlan> last_plan;
  OSQPWorkspace* workspace = nullptr;
  bool workspace_has_slew_variables = false;
  static constexpr double slew_scale = .1;
  static constexpr int minimum_suffix_horizon = 96;

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
    position_origin = Vector::Zero(joint_count);
    bool center_positions = false;
    for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
      const double half_lower = .5 * limits.lower[joint], half_upper = .5 * limits.upper[joint];
      const double midpoint = half_lower + half_upper;
      // A distant, narrow range can spend OSQP's relative tolerance on the
      // coordinate offset. Move the origin before the internal solve.
      if (.125 * std::abs(midpoint) > half_upper - half_lower) {
        position_origin[joint] = midpoint;
        center_positions = center_positions || std::abs(midpoint) > 100.;
      }
    }
    if (!center_positions) position_origin.resize(0);
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
    if (trajectory_size <= std::numeric_limits<int>::max() / 10 &&
        detail::prefer_slew_variables(limits, dt, horizon)) {
      // Keep the public position/velocity problem. Internally add one
      // decision per velocity difference and separate its box from dynamics.
      slew_hessian = scaled_hessian;
      slew_hessian.conservativeResize(3 * trajectory_size, 3 * trajectory_size);
      slew_constraint = scaled_constraint;
      slew_constraint.conservativeResize(5 * trajectory_size, 3 * trajectory_size);
      for (Eigen::Index i = 0; i < trajectory_size; ++i) {
        slew_constraint.insert(3 * trajectory_size + i, 2 * trajectory_size + i) = -slew_scale;
        slew_constraint.insert(4 * trajectory_size + i, 2 * trajectory_size + i) = slew_scale;
      }
      slew_constraint.makeCompressed();
    }
    prepare_tail_feedback();
  }

  void prepare_tail_feedback() {
    if (horizon < minimum_suffix_horizon || position_origin.size()) return;
    const int shortest = 32;
    std::array<PrefixTrial, 3> candidates;
    candidates[0].horizon = shortest;
    candidates[1].horizon = std::clamp(horizon / 5, 40, 64);
    // Keep subsidiary solves below the threshold that creates their own prefix.
    candidates[2].horizon = std::min(horizon / 2, minimum_suffix_horizon - 1);
    Vector feedback(horizon - shortest);
    const long double step = dt, effort = control_weight;
    long double value = terminal_weight;
    // Eliminating one unconstrained interval leaves a scalar quadratic
    // value in the position error at its origin.
    for (Eigen::Index i = feedback.size(); i-- > 0;) {
      const long double denominator = effort + step * step * value;
      const long double gain = step * value / denominator;
      if (!std::isfinite(denominator) || !(gain > 0.) ||
          gain > std::numeric_limits<double>::max()) return;
      feedback[i] = static_cast<double>(gain);
      if (!(feedback[i] > 0. && dt * feedback[i] <= 1.)) return;
      value = tracking_weight + value * (effort / denominator);
      if (!std::isfinite(value) || value > std::numeric_limits<double>::max()) return;
      for (auto& trial : candidates)
        if (i + shortest == trial.horizon) trial.terminal_weight = static_cast<double>(value);
    }
    for (auto& trial : candidates) trial.travel = travel_estimate(trial.horizon);
    prefixes = std::move(candidates);
    tail_feedback = std::move(feedback);
  }

  ~Impl() {
    if (workspace) osqp_cleanup(workspace);
  }

  void reset() {
    if (workspace) osqp_cleanup(workspace);
    workspace = nullptr;
    last_plan.reset();
    for (auto& prefix : prefixes)
      if (prefix.planner) prefix.planner->reset();
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

  void center_vectors(const Vector& goal, Vector& gradient, Vector& lower_bounds,
                      Vector& upper_bounds) const {
    const Vector centered_goal = goal - position_origin;
    lower_bounds.head(joint_count) -= position_origin;
    upper_bounds.head(joint_count) -= position_origin;
    for (int k = 0; k < horizon; ++k) {
      const auto row = k * joint_count;
      gradient.segment(row, joint_count) =
          -position_weights.segment(row, joint_count).cwiseProduct(centered_goal);
      lower_bounds.segment(trajectory_size + row, joint_count) -= position_origin;
      upper_bounds.segment(trajectory_size + row, joint_count) -= position_origin;
    }
    require(gradient.allFinite() && lower_bounds.allFinite() && upper_bounds.allFinite(),
            "nonfinite centered MPC vectors");
  }

  void validate_solution(QPResult& result, const Vector& lower_bound,
                         const Vector& upper_bound) const {
    result.status = "error";
    result.violation = INFINITY;
    if (!result.x.allFinite()) {
      result.x.resize(0);
      result.detail = "nonfinite solution";
      return;
    }
    const Vector values = constraint * result.x;
    if (!values.allFinite()) {
      result.x.resize(0);
      result.detail = "nonfinite constraint evaluation";
      return;
    }
    result.violation = 0.;
    for (Eigen::Index i = 0; i < values.size(); ++i)
      result.violation = std::max({result.violation, lower_bound[i] - values[i],
                                   values[i] - upper_bound[i]});
    if (result.violation > 1e-6) {
      result.x.resize(0);
      result.detail = "solution exceeds absolute constraint tolerance";
      return;
    }
    result.status = "solved";
  }

  PlanResult make_plan(const RobotState& state, QPResult result) {
    if (!result.success()) return {std::move(result), std::nullopt};
    JointPlan plan{Matrix(horizon + 1, joint_count), Matrix(horizon, joint_count), state.stamp, dt};
    plan.positions.row(0) = state.q.transpose();
    for (int k = 0; k < horizon; ++k) {
      const auto velocity = result.x.segment(trajectory_size + k * joint_count, joint_count);
      plan.velocities.row(k) = velocity.transpose();
      plan.positions.row(k + 1) = plan.positions.row(k) + dt * velocity.transpose();
    }
    const double violation = detail::joint_plan_violation(plan, limits, state);
    if (violation > 1e-6)
      return {{"error", {}, result.iterations, violation,
               "reconstructed sparse MPC plan violates joint bounds"}, std::nullopt};
    last_plan = plan;
    return {std::move(result), std::move(plan)};
  }

  Vector travel_estimate(int steps) const {
    const long double duration = static_cast<long double>(dt) * steps;
    Vector travel(joint_count);
    for (Eigen::Index j = 0; j < joint_count; ++j) {
      long double speed = limits.velocity[j], time = duration;
      if (limits.acceleration.size()) {
        const long double acceleration = limits.acceleration[j];
        speed = std::min(speed, .5L * duration * acceleration);
        time -= speed / acceleration;
      }
      const long double range = speed * time;
      travel[j] = range <= std::numeric_limits<double>::max() ? static_cast<double>(range) : INFINITY;
    }
    return travel;
  }

  std::optional<PlanResult> solve_prefix(const RobotState& state, const Vector& goal,
                                        const Vector& lower_bound, const Vector& upper_bound,
                                        int& iterations) {
    if (!tail_feedback.size()) return std::nullopt;
    if ((goal.array() < limits.lower.array()).any() ||
        (goal.array() > limits.upper.array()).any())
      return std::nullopt;
    const Vector distance = (goal - state.q).cwiseAbs();
    // Choose one speculative workspace using the distance that a
    // rest-to-rest motion can cover at the speed and acceleration limits.
    auto prefix = std::find_if(prefixes.begin(), prefixes.end(), [&](const PrefixTrial& trial) {
      return (distance.array() <= trial.travel.array()).all() &&
             (trial.terminal_weight * goal).allFinite();
    });
    if (prefix == prefixes.end())
      return std::nullopt;
    const int steps = prefix->horizon;
    if (!prefix->planner)
      prefix->planner = std::make_unique<SparseJointMPC>(limits, steps, dt, tracking_weight,
                                                        control_weight, prefix->terminal_weight);
    auto first = prefix->planner->solve(state, goal);
    iterations = first.solver.iterations;
    if (!first.solver.success()) return std::nullopt;
    const Eigen::Index prefix_size = steps * joint_count;
    QPResult result{"solved", Vector(2 * trajectory_size), iterations, INFINITY,
                    "unconstrained suffix"};
    result.x.head(prefix_size) = first.solver.x.head(prefix_size);
    result.x.segment(trajectory_size, prefix_size) = first.solver.x.tail(prefix_size);
    Vector position = first.solver.x.segment(prefix_size - joint_count, joint_count);
    Vector previous_velocity = first.solver.x.tail(joint_count);
    for (int k = steps; k < horizon; ++k) {
      const auto row = k * joint_count;
      auto velocity = result.x.segment(trajectory_size + row, joint_count);
      velocity = tail_feedback[k - prefixes.front().horizon] * (goal - position);
      position += dt * velocity;
      if (!(velocity.cwiseAbs().array() <= limits.velocity.array()).all() ||
          !(position.array() >= limits.lower.array()).all() ||
          !(position.array() <= limits.upper.array()).all())
        return std::nullopt;
      if (limits.acceleration.size() &&
          !((velocity - previous_velocity).cwiseAbs().array() <=
            dt * limits.acceleration.array()).all())
        return std::nullopt;
      result.x.segment(row, joint_count) = position;
      previous_velocity = velocity;
    }
    validate_solution(result, lower_bound, upper_bound);
    auto complete = make_plan(state, std::move(result));
    if (!complete.plan) return std::nullopt;
    return complete;
  }

  QPResult solve_qp(const Vector& gradient, const Vector& lower_bound, const Vector& upper_bound,
                    const Vector& warm, const Vector& measured_velocity, bool slew_variables,
                    bool* retryable = nullptr) {
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
    if (slew_variables) {
      scaled_gradient.conservativeResize(3 * trajectory_size);
      scaled_gradient.tail(trajectory_size).setZero();
      lower_bounds.conservativeResize(5 * trajectory_size);
      upper_bounds.conservativeResize(5 * trajectory_size);
      lower_bounds.tail(trajectory_size) = lower.tail(trajectory_size).cwiseMax(-OSQP_INFTY);
      upper_bounds.tail(trajectory_size) = upper.tail(trajectory_size).cwiseMin(OSQP_INFTY);
      lower_bounds.segment(3 * trajectory_size, trajectory_size).setZero();
      upper_bounds.segment(3 * trajectory_size, trajectory_size).setZero();
      lower_bounds.segment(3 * trajectory_size, joint_count) = measured_velocity.cwiseMax(-OSQP_INFTY);
      upper_bounds.segment(3 * trajectory_size, joint_count) = measured_velocity.cwiseMin(OSQP_INFTY);
      if (scaled_warm.size()) {
        scaled_warm.conservativeResize(3 * trajectory_size);
        auto differences = scaled_warm.tail(trajectory_size);
        const auto velocities = warm.tail(trajectory_size);
        differences.head(joint_count) = (velocities.head(joint_count) - measured_velocity) / slew_scale;
        differences.tail(trajectory_size - joint_count) =
            (velocities.tail(trajectory_size - joint_count) -
             velocities.head(trajectory_size - joint_count)) / slew_scale;
        if (!differences.allFinite()) scaled_warm.resize(0);
      }
    }
    if (!scaled_gradient.allFinite() || !scaled_warm.allFinite())
      return {"error", {}, 0, INFINITY, "nonfinite scaled MPC vectors"};
    if (workspace && workspace_has_slew_variables != slew_variables) {
      osqp_cleanup(workspace);
      workspace = nullptr;
    }
    c_int error = 0;
    if (!workspace) {
      CscStorage hessian_csc(slew_variables ? slew_hessian : scaled_hessian);
      CscStorage constraint_csc(slew_variables ? slew_constraint : scaled_constraint);
      OSQPData data{};
      data.n = scaled_gradient.size();
      data.m = lower_bounds.size();
      data.P = &hessian_csc.matrix;
      data.A = &constraint_csc.matrix;
      data.q = scaled_gradient.data();
      data.l = lower_bounds.data();
      data.u = upper_bounds.data();
      auto settings = detail::osqp_settings();
      settings.adaptive_rho_interval = 100;
      error = osqp_setup(&workspace, &data, &settings);
      workspace_has_slew_variables = slew_variables;
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
    const Eigen::Map<const Vector> solution(workspace->solution->x, scaled_gradient.size());
    if (slew_variables && !solution.tail(trajectory_size).allFinite()) {
      result.detail = "nonfinite solution";
      return result;
    }
    result.x = solution.head(2 * trajectory_size);
    result.x.head(trajectory_size) *= position_scale;
    validate_solution(result, lower_bound, upper_bound);
    return result;
  }
};

SparseJointMPC::SparseJointMPC(JointLimits limits, int horizon, double dt, double tracking_weight,
                               double control_weight, double terminal_weight)
    : impl_(std::make_unique<Impl>(std::move(limits), horizon, dt, tracking_weight, control_weight,
                                   terminal_weight)) {}
SparseJointMPC::~SparseJointMPC() = default;
void SparseJointMPC::reset() { impl_->reset(); }

std::size_t SparseJointMPC::fixed_matrix_bytes() const {
  const auto& planner = *impl_;
  std::size_t prefix_bytes = 0;
  for (const auto& prefix : planner.prefixes)
    if (prefix.planner) prefix_bytes += prefix.planner->fixed_matrix_bytes();
  return prefix_bytes + detail::sparse_matrix_bytes(planner.hessian) +
         detail::sparse_matrix_bytes(planner.constraint) +
         detail::sparse_matrix_bytes(planner.scaled_hessian) +
         detail::sparse_matrix_bytes(planner.scaled_constraint) +
         detail::sparse_matrix_bytes(planner.slew_hessian) +
         detail::sparse_matrix_bytes(planner.slew_constraint);
}

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
  int prefix_iterations = 0;
  if (auto result = planner.solve_prefix(state, goal, lower_bounds, upper_bounds, prefix_iterations))
    return std::move(*result);
  const bool centered = planner.position_origin.size() != 0;
  Vector physical_lower, physical_upper;
  if (centered) {
    physical_lower = lower_bounds;
    physical_upper = upper_bounds;
    planner.center_vectors(goal, gradient, lower_bounds, upper_bounds);
  }
  Vector warm;
  if (planner.last_plan && state.stamp >= planner.last_plan->stamp) {
    const auto offset =
        detail::interval(state.stamp, planner.last_plan->stamp, planner.dt, planner.horizon);
    warm.resize(2 * planner.trajectory_size);
    Vector position = state.q;
    if (centered) position -= planner.position_origin;
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
  const bool slew_variables = planner.slew_hessian.rows() != 0;
  const bool reused_workspace = planner.workspace != nullptr &&
                                planner.workspace_has_slew_variables == slew_variables;
  const auto solve = [&](const Vector& guess, bool use_slew_variables, bool* retryable = nullptr) {
    auto result = planner.solve_qp(
        gradient, lower_bounds, upper_bounds, guess, state.v, use_slew_variables, retryable);
    if (centered && result.success()) {
      for (int k = 0; k < planner.horizon; ++k)
        result.x.segment(k * planner.joint_count, planner.joint_count) += planner.position_origin;
      planner.validate_solution(result, physical_lower, physical_upper);
    }
    return result;
  };
  bool retryable = false;
  auto result = solve(warm, slew_variables, &retryable);
  if (reused_workspace && retryable) {
    // Keep the same formulation for the cold retry. A poor warm start
    // can fail even when the velocity-difference model converges cold.
    const auto first_iterations = result.iterations;
    const auto first_status = result.detail;
    planner.reset();
    result = solve(Vector(), slew_variables);
    result.iterations += first_iterations;
    result.detail = "cold retry after " + first_status + "; " + result.detail;
  }
  const bool retry_original = slew_variables && !result.success() && result.status != "infeasible";
  if (retry_original) {
    const auto first_iterations = result.iterations;
    planner.reset();
    result = solve(Vector(), false);
    result.iterations += first_iterations;
    result.detail = "retry after velocity-difference solve: " + result.detail;
  }
  result.iterations += prefix_iterations;
  return planner.make_plan(state, std::move(result));
}

}  // namespace wholebodyx
