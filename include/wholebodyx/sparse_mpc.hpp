#pragma once

#include "wholebodyx/control.hpp"
#include <Eigen/SparseCore>

namespace wholebodyx {

// Diagnostic snapshot of the uncondensed joint-integrator formulation.
// Decision order: [q_1, ..., q_N, u_0, ..., u_(N-1)]. Changing this returned
// value does not change the planner's fixed matrices or solver workspace.
struct SparseMPCProblem {
  Eigen::SparseMatrix<double> hessian, constraint;
  Vector gradient, lower, upper;
};

// Specialized sparse formulation with positive diagonal costs and O(n*N)
// storage. The fixed matrices are owned and validated at construction; solve()
// updates only vectors and independently checks constraints and output bounds.
// One instance per serial caller, like JointMPC and OSQPSolver.
class SparseJointMPC final : public Planner {
 public:
  SparseJointMPC(JointLimits, int horizon = 15, double dt = .1,
                 double tracking_weight = 10., double control_weight = .1,
                 double terminal_weight = 30.);
  ~SparseJointMPC() override;
  void reset() override;
  SparseMPCProblem formulate(const RobotState&, const Vector& goal) const;
  PlanResult solve(const RobotState&, const Vector& goal) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace wholebodyx
