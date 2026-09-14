#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wholebodyx {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Fixed-base Euclidean joints, SI units, model order, monotonic timestamps.
struct RobotState {
  Vector q, v;
  double stamp = 0.0;
  void validate() const;
};

struct JointLimits {
  Vector lower, upper, velocity;
  Vector acceleration;  // Empty means no velocity-slew constraint.
  void validate() const;
  std::pair<Vector, Vector> velocity_bounds(const RobotState&, double dt) const;
};

struct LinearTask {
  Matrix jacobian;
  Vector target_velocity, weight;
  void validate(Eigen::Index joints) const;
};

struct QuadraticProgram {
  Matrix hessian;
  Vector gradient;
  Matrix constraint;
  Vector lower, upper;
  void validate() const;

 private:
  friend class OSQPSolver;
  void validate_against(const Matrix* previously_validated_hessian) const;
};

struct QPResult {
  std::string status = "error";
  Vector x;  // Empty unless accepted as solved.
  int iterations = 0;
  double violation = std::numeric_limits<double>::infinity();
  std::string detail;
  bool success() const { return status == "solved" && x.size() > 0; }
};

class QPSolver {
 public:
  virtual ~QPSolver() = default;
  virtual void reset() = 0;
  virtual QPResult solve(const QuadraticProgram&, const Vector& warm_start = Vector()) = 0;
};

// One mutable instance per loop. Uses the OSQP 0.6 C API behind this interface.
class OSQPSolver final : public QPSolver {
 public:
  OSQPSolver();
  ~OSQPSolver() override;
  void reset() override;
  QPResult solve(const QuadraticProgram&, const Vector& warm_start = Vector()) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct VelocityCommand {
  Vector velocity;
  double stamp, valid_until;
};
struct ControlResult {
  QPResult solver;
  std::optional<VelocityCommand> command;
};

class Controller {
 public:
  virtual ~Controller() = default;
  virtual void reset() = 0;
  virtual ControlResult step(const RobotState&, const std::vector<LinearTask>&, double dt) = 0;
};

class KinematicWBC final : public Controller {
 public:
  KinematicWBC(JointLimits, double regularization = 1e-6, double smoothing = 0.0,
               std::shared_ptr<QPSolver> solver = nullptr);
  void reset() override;
  QuadraticProgram formulate(const RobotState&, const std::vector<LinearTask>&, double dt) const;
  ControlResult step(const RobotState&, const std::vector<LinearTask>&, double dt) override;

 private:
  JointLimits limits_;
  double regularization_, smoothing_;
  std::shared_ptr<QPSolver> solver_;
};

struct JointPlan {
  Matrix positions, velocities;  // Time along rows: N+1 positions, N velocities.
  double stamp, dt;
  void validate() const;
  double valid_until() const;
  std::pair<Vector, Vector> sample(double time) const;
};
struct PlanResult {
  QPResult solver;
  std::optional<JointPlan> plan;
};

class Planner {
 public:
  virtual ~Planner() = default;
  virtual void reset() = 0;
  virtual PlanResult solve(const RobotState&, const Vector& goal) = 0;
};

class JointMPC final : public Planner {
 public:
  JointMPC(JointLimits, int horizon = 15, double dt = 0.1, double tracking_weight = 10.0,
           double control_weight = 0.1, double terminal_weight = 30.0,
           std::shared_ptr<QPSolver> solver = nullptr);
  void reset() override;
  QuadraticProgram formulate(const RobotState&, const Vector& goal) const;
  PlanResult solve(const RobotState&, const Vector& goal) override;

 private:
  JointLimits limits_;
  int horizon_;
  double dt_, tracking_weight_, control_weight_, terminal_weight_;
  std::shared_ptr<QPSolver> solver_;
  std::optional<JointPlan> last_plan_;
  Matrix hessian_, constraint_, gradient_map_;
  Vector lower_, upper_;
};
}  // namespace wholebodyx
