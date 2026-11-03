#include "wholebodyx/control.hpp"
#include "fixed_qp.hpp"
#include "osqp_execution.hpp"

#include <osqp/osqp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wholebodyx {
namespace {
struct CscStorage {
  std::vector<c_int> columns, rows;
  std::vector<c_float> values;
  csc matrix{};
  void finish(Eigen::Index m, Eigen::Index n) {
    matrix.nzmax = values.size(); matrix.m = m; matrix.n = n;
    matrix.p = columns.data(); matrix.i = rows.data(); matrix.x = values.data(); matrix.nz = -1;
  }
  CscStorage(const Matrix& input, bool upper) {
    columns.push_back(0);
    for (Eigen::Index j = 0; j < input.cols(); ++j) {
      for (Eigen::Index i = 0; i < input.rows(); ++i) {
        if ((!upper || i <= j) && input(i, j) != 0) {
          rows.push_back(i); values.push_back(input(i, j));
        }
      }
      columns.push_back(values.size());
    }
    finish(input.rows(), input.cols());
  }
  CscStorage(const Eigen::SparseMatrix<double>& input, bool upper) {
    columns.reserve(input.cols() + 1);
    rows.reserve(input.nonZeros()); values.reserve(input.nonZeros());
    columns.push_back(0);
    for (Eigen::Index j = 0; j < input.cols(); ++j) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(input, j); it; ++it) {
        if ((!upper || it.row() <= j) && it.value() != 0) {
          rows.push_back(it.row()); values.push_back(it.value());
        }
      }
      columns.push_back(values.size());
    }
    finish(input.rows(), input.cols());
  }
};
void validate_warm(const Vector& warm, Eigen::Index n) {
  if (warm.size() && (warm.size() != n || !warm.allFinite()))
    throw std::invalid_argument("invalid warm start");
}
bool inconsistent(const Vector& lower, const Vector& upper) {
  for (Eigen::Index i = 0; i < lower.size(); ++i)
    if (lower[i] > upper[i] || lower[i] == INFINITY || upper[i] == -INFINITY) return true;
  return false;
}
}  // namespace

struct OSQPSolver::Impl {
  OSQPWorkspace* work = nullptr;
  std::vector<c_int> pcols, prows, acols, arows;
  std::vector<c_float> avalues;  // Original, unscaled A for independent residuals.
  Eigen::Index n = -1, m = -1;
  Matrix hessian, constraint;  // Snapshots only for the mutable public QP path.
  std::shared_ptr<const detail::FixedQP> fixed;
  ~Impl() { if (work) osqp_cleanup(work); }

  c_int update_vectors(Vector& q, Vector& lo, Vector& hi) {
    c_int error = osqp_update_lin_cost(work, q.data());
    if (!error) error = osqp_update_bounds(work, lo.data(), hi.data());
    return error;
  }
  c_int update_matrices(CscStorage& h, CscStorage& a, Vector& q, Vector& lo, Vector& hi) {
    const bool same_pattern = work && n == q.size() && m == lo.size() &&
        pcols == h.columns && prows == h.rows && acols == a.columns && arows == a.rows;
    c_int error = 0;
    if (same_pattern) {
      error = osqp_update_P_A(work, h.values.data(), nullptr, h.values.size(),
                              a.values.data(), nullptr, a.values.size());
      if (!error) error = update_vectors(q, lo, hi);
    } else {
      if (work) osqp_cleanup(work);
      work = nullptr;
      OSQPData data{};
      data.n = q.size(); data.m = lo.size(); data.P = &h.matrix; data.A = &a.matrix;
      data.q = q.data(); data.l = lo.data(); data.u = hi.data();
      auto settings = detail::osqp_settings();
      error = osqp_setup(&work, &data, &settings);
      pcols = h.columns; prows = h.rows; acols = a.columns; arows = a.rows;
      n = q.size(); m = lo.size();
    }
    if (!error) avalues = std::move(a.values);
    return error;
  }
};

OSQPSolver::OSQPSolver() : impl_(std::make_unique<Impl>()) {}
OSQPSolver::~OSQPSolver() = default;
void OSQPSolver::reset() { impl_ = std::make_unique<Impl>(); }

QPResult OSQPSolver::solve(const QuadraticProgram& p, const Vector& warm_start) {
  // A caller-owned dense QP never acquires a fixed template's trust token.
  p.validate_against(impl_->work && !impl_->fixed ? &impl_->hessian : nullptr);
  return solve_impl(p.hessian, p.gradient, p.constraint, p.lower, p.upper, warm_start);
}

QPResult OSQPSolver::solve_fixed(std::shared_ptr<const detail::FixedQP> fixed, const Vector& gradient,
                                const Vector& lower, const Vector& upper, const Vector& warm_start) {
  if (!(gradient.size() == fixed->hessian.rows() && lower.size() == fixed->constraint.rows() &&
        upper.size() == lower.size() && gradient.allFinite() && !lower.array().isNaN().any() &&
        !upper.array().isNaN().any()))
    throw std::invalid_argument("invalid fixed QP vectors");
  validate_warm(warm_start, gradient.size());
  if (inconsistent(lower, upper)) return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};
  Vector q = gradient, lo = lower.cwiseMax(-OSQP_INFTY), hi = upper.cwiseMin(OSQP_INFTY);
  c_int error = 0;
  if (impl_->work && impl_->fixed == fixed) {
    error = impl_->update_vectors(q, lo, hi);
  } else {
    CscStorage h(fixed->hessian, true), a(fixed->constraint, false);
    error = impl_->update_matrices(h, a, q, lo, hi);
    if (!error) {
      impl_->fixed = std::move(fixed);
      impl_->hessian.resize(0, 0); impl_->constraint.resize(0, 0);
    }
  }
  if (error) { reset(); return {"error", {}, 0, INFINITY, "OSQP setup/update failed"}; }
  return run(lower, upper, warm_start);
}

QPResult OSQPSolver::solve_impl(const Matrix& hessian, const Vector& gradient, const Matrix& constraint,
                               const Vector& lower, const Vector& upper, const Vector& warm_start) {
  validate_warm(warm_start, gradient.size());
  if (inconsistent(lower, upper)) return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};
  Vector q = gradient, lo = lower.cwiseMax(-OSQP_INFTY), hi = upper.cwiseMin(OSQP_INFTY);
  const bool same_matrices = impl_->work && !impl_->fixed && impl_->n == gradient.size() && impl_->m == lower.size() &&
      (impl_->hessian.array() == hessian.array()).all() &&
      (impl_->constraint.array() == constraint.array()).all();
  c_int error = 0;
  if (same_matrices) {
    error = impl_->update_vectors(q, lo, hi);
  } else {
    CscStorage h(hessian, true), a(constraint, false);
    error = impl_->update_matrices(h, a, q, lo, hi);
    if (!error) {
      impl_->hessian = hessian; impl_->constraint = constraint;
      impl_->fixed.reset();
    }
  }
  if (error) { reset(); return {"error", {}, 0, INFINITY, "OSQP setup/update failed"}; }
  return run(lower, upper, warm_start);
}

QPResult OSQPSolver::run(const Vector& lower, const Vector& upper, const Vector& warm_start) {
  auto* work = impl_->work;
  if (warm_start.size() && osqp_warm_start_x(work, warm_start.data())) {
    reset(); return {"error", {}, 0, INFINITY, "OSQP warm start failed"};
  }
  const auto error = detail::execute_osqp(work);
  const auto status = work->info->status_val;
  QPResult result;
  result.iterations = static_cast<int>(work->info->iter);
  result.detail = work->info->status;
  if (error || status != OSQP_SOLVED) {
    result.status = detail::rejected_osqp_status(status);
    return result;
  }
  Vector x = Eigen::Map<const Vector>(work->solution->x, impl_->n);
  if (!x.allFinite()) { result.detail = "nonfinite solution"; return result; }
  Vector ax = Vector::Zero(lower.size());
  for (Eigen::Index j = 0; j < x.size(); ++j)
    for (c_int k = impl_->acols[j]; k < impl_->acols[j + 1]; ++k)
      ax[impl_->arows[k]] += impl_->avalues[k] * x[j];
  if (!ax.allFinite()) { result.detail = "nonfinite constraint evaluation"; return result; }
  result.violation = 0;
  for (Eigen::Index i = 0; i < ax.size(); ++i)
    result.violation = std::max({result.violation, lower[i] - ax[i], ax[i] - upper[i]});
  if (result.violation > 1e-6) { result.detail = "solution exceeds absolute constraint tolerance"; return result; }
  result.status = "solved";
  result.x = std::move(x);
  return result;
}
}  // namespace wholebodyx
