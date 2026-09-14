#include "wholebodyx/control.hpp"

#include <osqp/osqp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace wholebodyx {
namespace {
static_assert(std::is_same<c_float, double>::value, "WholeBodyX requires double-precision OSQP");
struct CscStorage {
  std::vector<c_int> columns, rows;
  std::vector<c_float> values;
  csc matrix{};
  CscStorage(const Matrix& input, bool upper) {
    columns.push_back(0);
    for (Eigen::Index j = 0; j < input.cols(); ++j) {
      for (Eigen::Index i = 0; i < input.rows(); ++i) {
        if ((!upper || i <= j) && input(i, j) != 0) {
          rows.push_back(i);
          values.push_back(input(i, j));
        }
      }
      columns.push_back(values.size());
    }
    matrix.nzmax = values.size(); matrix.m = input.rows(); matrix.n = input.cols();
    matrix.p = columns.data(); matrix.i = rows.data(); matrix.x = values.data(); matrix.nz = -1;
  }
};
}  // namespace
struct OSQPSolver::Impl {
  OSQPWorkspace* work = nullptr;
  std::vector<c_int> pcols, prows, acols, arows;
  Eigen::Index n = -1, m = -1;
  Matrix hessian, constraint;
  ~Impl() { if (work) osqp_cleanup(work); }
};
OSQPSolver::OSQPSolver() : impl_(std::make_unique<Impl>()) {}
OSQPSolver::~OSQPSolver() = default;
void OSQPSolver::reset() { impl_ = std::make_unique<Impl>(); }
QPResult OSQPSolver::solve(const QuadraticProgram& p, const Vector& warm_start) {
  p.validate_against(impl_->work ? &impl_->hessian : nullptr);
  if (warm_start.size() && (warm_start.size() != p.gradient.size() || !warm_start.allFinite()))
    throw std::invalid_argument("invalid warm start");
  for (Eigen::Index i = 0; i < p.lower.size(); ++i)
    if (p.lower[i] > p.upper[i] || p.lower[i] == INFINITY || p.upper[i] == -INFINITY)
      return {"infeasible", {}, 0, INFINITY, "inconsistent constraint bounds"};
  Vector q = p.gradient;
  Vector lo = p.lower.cwiseMax(-OSQP_INFTY), hi = p.upper.cwiseMin(OSQP_INFTY);
  const bool same_matrices = impl_->work && impl_->n == p.gradient.size() && impl_->m == p.lower.size() &&
      (impl_->hessian.array() == p.hessian.array()).all() &&
      (impl_->constraint.array() == p.constraint.array()).all();
  c_int error = 0;
  if (same_matrices) {
    error = osqp_update_lin_cost(impl_->work, q.data());
    if (!error) error = osqp_update_bounds(impl_->work, lo.data(), hi.data());
  } else {
    CscStorage h(p.hessian, true), a(p.constraint, false);
    const bool same_pattern = impl_->work && impl_->n == q.size() && impl_->m == lo.size() &&
        impl_->pcols == h.columns && impl_->prows == h.rows && impl_->acols == a.columns && impl_->arows == a.rows;
    if (same_pattern) {
      error = osqp_update_P_A(impl_->work, h.values.data(), nullptr, h.values.size(),
                             a.values.data(), nullptr, a.values.size());
      if (!error) error = osqp_update_lin_cost(impl_->work, q.data());
      if (!error) error = osqp_update_bounds(impl_->work, lo.data(), hi.data());
    } else {
      reset();
      OSQPData data{};
      data.n = q.size(); data.m = lo.size(); data.P = &h.matrix; data.A = &a.matrix;
      data.q = q.data(); data.l = lo.data(); data.u = hi.data();
      OSQPSettings settings;
      osqp_set_default_settings(&settings);
      settings.verbose = 0; settings.eps_abs = 1e-8; settings.eps_rel = 1e-8;
      settings.max_iter = 20000; settings.polish = 0; settings.warm_start = 1;
      error = osqp_setup(&impl_->work, &data, &settings);
      impl_->pcols = h.columns; impl_->prows = h.rows; impl_->acols = a.columns; impl_->arows = a.rows;
      impl_->n = q.size(); impl_->m = lo.size();
    }
    if (!error) { impl_->hessian = p.hessian; impl_->constraint = p.constraint; }
  }
  if (error) { reset(); return {"error", {}, 0, INFINITY, "OSQP setup/update failed"}; }
  auto* work = impl_->work;
  if (warm_start.size() && osqp_warm_start_x(work, warm_start.data())) {
    reset(); return {"error", {}, 0, INFINITY, "OSQP warm start failed"};
  }
  error = osqp_solve(work);
  const auto status = work->info->status_val;
  QPResult result;
  result.iterations = static_cast<int>(work->info->iter);
  result.detail = work->info->status;
  if (error || status != OSQP_SOLVED) {
    if (status == OSQP_PRIMAL_INFEASIBLE || status == OSQP_DUAL_INFEASIBLE ||
        status == OSQP_PRIMAL_INFEASIBLE_INACCURATE || status == OSQP_DUAL_INFEASIBLE_INACCURATE)
      result.status = "infeasible";
    else if (status == OSQP_MAX_ITER_REACHED) result.status = "iteration_limit";
    return result;
  }
  Vector x = Eigen::Map<const Vector>(work->solution->x, p.gradient.size());
  if (!x.allFinite()) { result.detail = "nonfinite solution"; return result; }
  Vector ax = p.constraint * x;
  if (!ax.allFinite()) { result.detail = "nonfinite constraint evaluation"; return result; }
  result.violation = 0;
  for (Eigen::Index i = 0; i < ax.size(); ++i)
    result.violation = std::max({result.violation, p.lower[i] - ax[i], ax[i] - p.upper[i]});
  if (result.violation > 1e-6) { result.detail = "solution exceeds absolute constraint tolerance"; return result; }
  result.status = "solved";
  result.x = std::move(x);
  return result;
}
}  // namespace wholebodyx
