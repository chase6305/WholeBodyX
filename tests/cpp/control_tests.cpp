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
    std::cout << "C++ constrained optimum, infeasibility, MPC and reference checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
