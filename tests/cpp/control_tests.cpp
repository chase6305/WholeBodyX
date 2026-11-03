#include "wholebodyx/control.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace wholebodyx;
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  try {
    JointLimits limits{Vector::Constant(2, -2), Vector::Constant(2, 2), Vector::Ones(2), {}};
    limits.velocity[0] = 0.2;
    RobotState state{Vector::Zero(2), Vector::Zero(2), 0};
    KinematicWBC wbc(limits);
    LinearTask task{Matrix::Ones(1, 2), Vector::Ones(1), Vector::Ones(1)};
    auto control = wbc.step(state, {task}, 0.1);
    check(control.command.has_value(), "WBC must solve");
    check(std::abs(control.command->velocity[0] - 0.2) < 1e-5, "active bound");
    check(std::abs(control.command->velocity[1] - 0.8) < 1e-5, "coupled constrained optimum");
    for (double direction : {-1., 1.}) {
      JointLimits braking{Vector::Constant(1, -1), Vector::Ones(1), Vector::Ones(1), Vector::Ones(1)};
      KinematicWBC safe(braking);
      RobotState measured{Vector::Constant(1, direction * .96), Vector::Constant(1, direction * .3), 0};
      LinearTask drive{Matrix::Identity(1, 1), Vector::Constant(1, direction * .4), Vector::Ones(1)};
      for (int i = 0; i < 30; ++i) {
        const auto result = safe.step(measured, {drive}, .1);
        if (!result.command) {
          const auto bounds = braking.velocity_bounds(measured, .1);
          std::cerr.precision(17);
          std::cerr << "braking step " << i << ", q=" << measured.q[0] << ", v=" << measured.v[0]
                    << ", lower=" << bounds.first[0] << ", upper=" << bounds.second[0]
                    << ", solver=" << result.solver.detail << '\n';
        }
        check(result.command.has_value(), "braking lost next-step feasibility");
        measured = {measured.q + .1 * result.command->velocity, result.command->velocity, measured.stamp + .1};
        check(std::abs(measured.q[0]) <= 1 + 1e-7, "braking crossed joint limit");
      }
      check(std::abs(std::abs(measured.q[0]) - 1) < 1e-6, "braking did not reach the limit");
    }
    state.q[0] = 10;
    check(!wbc.step(state, {task}, 0.1).command, "infeasible WBC must not emit command");
    state.q.setZero();
    check(wbc.step(state, {task}, 0.1).solver.success(), "solver recovers after failure");

    JointLimits single{Vector::Constant(1, -10), Vector::Constant(1, 10), Vector::Constant(1, 10), {}};
    JointMPC mpc(single, 1, 0.2, 2, 0.5, 3);
    RobotState initial{Vector::Zero(1), Vector::Zero(1), 0};
    auto result = mpc.solve(initial, Vector::Ones(1));
    check(result.plan.has_value(), "MPC must solve");
    const double exact = 0.2 * 3 / (0.2 * 0.2 * 3 + 0.5);
    check(std::abs(result.plan->velocities(0, 0) - exact) < 1e-6, "one-step analytic MPC optimum");
    auto ref = result.plan->sample(0.1);
    check(std::abs(ref.first[0] - exact * 0.1) < 1e-6, "reference interpolation");
    bool expired = false;
    try { result.plan->sample(0.2); } catch (const std::invalid_argument&) { expired = true; }
    check(expired, "expired reference rejected");
    bool invalid = false;
    try { wbc.step(state, {task}, 0); } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "zero dt rejected");
    OSQPSolver cached;
    QuadraticProgram problem{Matrix::Identity(1, 1), -Vector::Ones(1), Matrix::Identity(1, 1),
                             Vector::Constant(1, -2), Vector::Constant(1, 2)};
    check(std::abs(cached.solve(problem).x[0] - 1) < 1e-6, "initial cache solve");
    problem.gradient[0] = -0.5;
    check(std::abs(cached.solve(problem).x[0] - 0.5) < 1e-6, "vector-only update");
    problem.hessian(0, 0) = 2;
    check(std::abs(cached.solve(problem).x[0] - 0.25) < 1e-6, "changed Hessian cannot reuse old values");
    problem.gradient[0] = -8;
    problem.constraint(0, 0) = 2;
    check(std::abs(cached.solve(problem).x[0] - 1) < 1e-6, "changed constraint values update bounds");
    problem.hessian(0, 0) = -1;
    invalid = false;
    try { cached.solve(problem); } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "cached validation must reject a mutated nonconvex Hessian");
    problem.hessian(0, 0) = 2;
    check(cached.solve(problem).success(), "cache recovers after rejected mutation");
    for (double eigenvalue : {0., -5e-11, -2e-10}) {
      QuadraticProgram semidefinite{Matrix::Identity(2, 2), Vector::Zero(2), Matrix::Identity(2, 2),
                                   Vector::Constant(2, -1), Vector::Ones(2)};
      semidefinite.hessian(0, 0) = eigenvalue;
      bool rejected = false;
      try { semidefinite.validate(); } catch (const std::invalid_argument&) { rejected = true; }
      check(rejected == (eigenvalue < -1e-10), "semidefinite validation tolerance changed");
    }
    std::cout << "C++ constrained optimum, infeasibility, MPC and reference checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
