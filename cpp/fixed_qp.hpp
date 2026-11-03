#pragma once

#include "wholebodyx/control.hpp"
#include <Eigen/SparseCore>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace wholebodyx::detail {

inline std::size_t sparse_matrix_bytes(const Eigen::SparseMatrix<double>& matrix) {
  return matrix.nonZeros() * (sizeof(double) + sizeof(int)) + (matrix.outerSize() + 1) * sizeof(int);
}

// Only JointMPC constructs this template. Its Hessian is T' W T + r I with
// positive W and r, so convexity follows from the validated MPC settings.
// A solver keeps shared ownership of the exact template it cached: identity
// cannot be reused when a planner is destroyed and a new one is allocated.
struct FixedQP {
  const Eigen::SparseMatrix<double> hessian, constraint;

 private:
  friend class wholebodyx::JointMPC;
  FixedQP(Eigen::SparseMatrix<double> h, Eigen::SparseMatrix<double> a)
      : hessian(std::move(h)), constraint(std::move(a)) {
    if (!(hessian.rows() > 0 && hessian.cols() == hessian.rows() &&
          constraint.cols() == hessian.cols() &&
          Eigen::Map<const Vector>(hessian.valuePtr(), hessian.nonZeros()).allFinite() &&
          Eigen::Map<const Vector>(constraint.valuePtr(), constraint.nonZeros()).allFinite()))
      throw std::invalid_argument("invalid fixed MPC matrices");
    const Eigen::SparseMatrix<double> difference = hessian - Eigen::SparseMatrix<double>(hessian.transpose());
    for (Eigen::Index i = 0; i < difference.nonZeros(); ++i)
      if (std::abs(difference.valuePtr()[i]) > 1e-10)
        throw std::invalid_argument("nonsymmetric fixed MPC Hessian");
  }
};

}  // namespace wholebodyx::detail
