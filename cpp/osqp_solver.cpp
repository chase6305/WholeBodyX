#include "wholebodyx/control.hpp"
#include "fixed_qp.hpp"
#include "osqp_execution.hpp"

#include <osqp/osqp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wholebodyx {
namespace {

// Keep the C view together with its owning index and value buffers.
struct CscStorage {
  std::vector<c_int> column_offsets, row_indices;
  std::vector<c_float> values;
  csc matrix{};

  // matrix contains pointers into these vectors, so copies would be invalid.
  CscStorage(const CscStorage&) = delete;
  CscStorage& operator=(const CscStorage&) = delete;

  void finish(Eigen::Index row_count, Eigen::Index column_count) {
    matrix.nzmax = values.size();
    matrix.m = row_count;
    matrix.n = column_count;
    matrix.p = column_offsets.data();
    matrix.i = row_indices.data();
    matrix.x = values.data();
    matrix.nz = -1;
  }

  CscStorage(const Matrix& input, bool upper_triangle) {
    column_offsets.reserve(input.cols() + 1);
    column_offsets.push_back(0);
    for (Eigen::Index column = 0; column < input.cols(); ++column) {
      const auto row_count = upper_triangle ? std::min(input.rows(), column + 1) : input.rows();
      for (Eigen::Index row = 0; row < row_count; ++row) {
        const double value = input(row, column);
        if (value != 0) {
          row_indices.push_back(row);
          values.push_back(value);
        }
      }
      column_offsets.push_back(values.size());
    }
    finish(input.rows(), input.cols());
  }

  CscStorage(const Eigen::SparseMatrix<double>& input, bool upper_triangle) {
    column_offsets.reserve(input.cols() + 1);
    row_indices.reserve(input.nonZeros());
    values.reserve(input.nonZeros());
    column_offsets.push_back(0);
    for (Eigen::Index column = 0; column < input.cols(); ++column) {
      for (Eigen::SparseMatrix<double>::InnerIterator entry(input, column); entry; ++entry) {
        if ((!upper_triangle || entry.row() <= column) && entry.value() != 0) {
          row_indices.push_back(entry.row());
          values.push_back(entry.value());
        }
      }
      column_offsets.push_back(values.size());
    }
    finish(input.rows(), input.cols());
  }
};

void validate_warm_start(const Vector& warm_start, Eigen::Index variable_count) {
  if (warm_start.size() && (warm_start.size() != variable_count || !warm_start.allFinite()))
    throw std::invalid_argument("invalid warm start");
}

bool inconsistent_bounds(const Vector& lower, const Vector& upper) {
  for (Eigen::Index row = 0; row < lower.size(); ++row) {
    if (lower[row] > upper[row] || lower[row] == INFINITY || upper[row] == -INFINITY)
      return true;
  }
  return false;
}

}  // namespace

struct OSQPSolver::Impl {
  OSQPWorkspace* workspace = nullptr;
  std::vector<c_int> hessian_column_offsets, hessian_row_indices;
  std::vector<c_int> constraint_column_offsets, constraint_row_indices;
  std::vector<c_float> constraint_values;  // Original A for physical residual checks.
  Eigen::Index variable_count = -1, constraint_count = -1;
  Matrix hessian, constraint;  // Value snapshots for caller-owned dense QPs.
  std::shared_ptr<const detail::FixedQP> fixed;

  ~Impl() {
    if (workspace) osqp_cleanup(workspace);
  }

  c_int update_vectors(Vector& gradient, Vector& lower, Vector& upper) {
    c_int error = osqp_update_lin_cost(workspace, gradient.data());
    if (!error) error = osqp_update_bounds(workspace, lower.data(), upper.data());
    return error;
  }

  c_int update_matrices(CscStorage& hessian_csc, CscStorage& constraint_csc,
                        Vector& gradient, Vector& lower, Vector& upper) {
    const bool same_pattern =
        workspace && variable_count == gradient.size() && constraint_count == lower.size() &&
        hessian_column_offsets == hessian_csc.column_offsets &&
        hessian_row_indices == hessian_csc.row_indices &&
        constraint_column_offsets == constraint_csc.column_offsets &&
        constraint_row_indices == constraint_csc.row_indices;

    c_int error = 0;
    if (same_pattern) {
      error = osqp_update_P_A(workspace,
                              hessian_csc.values.data(), nullptr, hessian_csc.values.size(),
                              constraint_csc.values.data(), nullptr, constraint_csc.values.size());
      if (!error) error = update_vectors(gradient, lower, upper);
    } else {
      if (workspace) osqp_cleanup(workspace);
      workspace = nullptr;
      OSQPData data{};
      data.n = gradient.size();
      data.m = lower.size();
      data.P = &hessian_csc.matrix;
      data.A = &constraint_csc.matrix;
      data.q = gradient.data();
      data.l = lower.data();
      data.u = upper.data();
      auto settings = detail::osqp_settings();
      error = osqp_setup(&workspace, &data, &settings);
      hessian_column_offsets = hessian_csc.column_offsets;
      hessian_row_indices = hessian_csc.row_indices;
      constraint_column_offsets = constraint_csc.column_offsets;
      constraint_row_indices = constraint_csc.row_indices;
      variable_count = gradient.size();
      constraint_count = lower.size();
    }
    if (!error) constraint_values = std::move(constraint_csc.values);
    return error;
  }
};

OSQPSolver::OSQPSolver() : impl_(std::make_unique<Impl>()) {}
OSQPSolver::~OSQPSolver() = default;
void OSQPSolver::reset() { impl_ = std::make_unique<Impl>(); }

QPResult OSQPSolver::solve(const QuadraticProgram& problem, const Vector& warm_start) {
  // Caller-owned dense QPs do not carry a fixed model's validation certificate.
  problem.validate_against(impl_->workspace && !impl_->fixed ? &impl_->hessian : nullptr);
  return solve_impl(problem.hessian, problem.gradient, problem.constraint,
                    problem.lower, problem.upper, warm_start);
}

QPResult OSQPSolver::solve_fixed(std::shared_ptr<const detail::FixedQP> fixed,
                                const Vector& gradient, const Vector& lower,
                                const Vector& upper, const Vector& warm_start) {
  if (!(gradient.size() == fixed->hessian.rows() && lower.size() == fixed->constraint.rows() &&
        upper.size() == lower.size() && gradient.allFinite() &&
        !lower.array().isNaN().any() && !upper.array().isNaN().any()))
    throw std::invalid_argument("invalid fixed QP vectors");
  validate_warm_start(warm_start, gradient.size());
  if (inconsistent_bounds(lower, upper))
    return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};

  Vector solver_gradient = gradient;
  Vector solver_lower = lower.cwiseMax(-OSQP_INFTY);
  Vector solver_upper = upper.cwiseMin(OSQP_INFTY);
  c_int error = 0;
  if (impl_->workspace && impl_->fixed == fixed) {
    error = impl_->update_vectors(solver_gradient, solver_lower, solver_upper);
  } else {
    CscStorage hessian_csc(fixed->hessian, true);
    CscStorage constraint_csc(fixed->constraint, false);
    error = impl_->update_matrices(hessian_csc, constraint_csc,
                                   solver_gradient, solver_lower, solver_upper);
    if (!error) {
      impl_->fixed = std::move(fixed);
      impl_->hessian.resize(0, 0);
      impl_->constraint.resize(0, 0);
    }
  }
  if (error) {
    reset();
    return {"error", {}, 0, INFINITY, "OSQP setup/update failed"};
  }
  return run(lower, upper, warm_start);
}

QPResult OSQPSolver::solve_impl(const Matrix& hessian, const Vector& gradient,
                               const Matrix& constraint, const Vector& lower,
                               const Vector& upper, const Vector& warm_start) {
  validate_warm_start(warm_start, gradient.size());
  if (inconsistent_bounds(lower, upper))
    return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};

  Vector solver_gradient = gradient;
  Vector solver_lower = lower.cwiseMax(-OSQP_INFTY);
  Vector solver_upper = upper.cwiseMin(OSQP_INFTY);
  const bool same_matrices =
      impl_->workspace && !impl_->fixed &&
      impl_->variable_count == gradient.size() && impl_->constraint_count == lower.size() &&
      (impl_->hessian.array() == hessian.array()).all() &&
      (impl_->constraint.array() == constraint.array()).all();

  c_int error = 0;
  if (same_matrices) {
    error = impl_->update_vectors(solver_gradient, solver_lower, solver_upper);
  } else {
    CscStorage hessian_csc(hessian, true);
    CscStorage constraint_csc(constraint, false);
    error = impl_->update_matrices(hessian_csc, constraint_csc,
                                   solver_gradient, solver_lower, solver_upper);
    if (!error) {
      impl_->hessian = hessian;
      impl_->constraint = constraint;
      impl_->fixed.reset();
    }
  }
  if (error) {
    reset();
    return {"error", {}, 0, INFINITY, "OSQP setup/update failed"};
  }
  return run(lower, upper, warm_start);
}

QPResult OSQPSolver::run(const Vector& lower, const Vector& upper, const Vector& warm_start) {
  auto* workspace = impl_->workspace;
  if (warm_start.size() && osqp_warm_start_x(workspace, warm_start.data())) {
    reset();
    return {"error", {}, 0, INFINITY, "OSQP warm start failed"};
  }
  const auto error = detail::execute_osqp(workspace);
  const auto status = workspace->info->status_val;
  QPResult result;
  result.iterations = static_cast<int>(workspace->info->iter);
  result.detail = workspace->info->status;
  if (error || status != OSQP_SOLVED) {
    result.status = detail::rejected_osqp_status(status);
    return result;
  }

  Vector solution = Eigen::Map<const Vector>(workspace->solution->x, impl_->variable_count);
  if (!solution.allFinite()) {
    result.detail = "nonfinite solution";
    return result;
  }
  Vector constraint_values = Vector::Zero(lower.size());
  for (Eigen::Index column = 0; column < solution.size(); ++column) {
    const auto end = impl_->constraint_column_offsets[column + 1];
    for (c_int offset = impl_->constraint_column_offsets[column]; offset < end; ++offset)
      constraint_values[impl_->constraint_row_indices[offset]] +=
          impl_->constraint_values[offset] * solution[column];
  }
  if (!constraint_values.allFinite()) {
    result.detail = "nonfinite constraint evaluation";
    return result;
  }
  result.violation = 0;
  for (Eigen::Index row = 0; row < constraint_values.size(); ++row)
    result.violation = std::max({result.violation,
                                 lower[row] - constraint_values[row],
                                 constraint_values[row] - upper[row]});
  if (result.violation > 1e-6) {
    result.detail = "solution exceeds absolute constraint tolerance";
    return result;
  }
  result.status = "solved";
  result.x = std::move(solution);
  return result;
}

}  // namespace wholebodyx
