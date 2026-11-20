#include "wholebodyx/control.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace wholebodyx;
namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class CustomSolver : public QPSolver {
 public:
  int calls = 0, resets = 0;
  bool malformed = false;
  double forced_velocity = 0.;
  OSQPSolver solver;
  void reset() override { ++resets; solver.reset(); }
  QPResult solve(const QuadraticProgram& problem, const Vector& warm) override {
    ++calls;
    problem.validate();  // An injected solver still receives the public contract.
    if (malformed) return {"solved", Vector::Zero(1), 0, 0, "injected malformed solution"};
    if (forced_velocity != 0)
      return {"solved", Vector::Constant(problem.gradient.size(), forced_velocity), 0, 0, "injected unsafe solution"};
    return solver.solve(problem, warm);
  }
};

void same_optimum(JointMPC& planner, const RobotState& state, const Vector& goal) {
  OSQPSolver independent;
  const auto expected = independent.solve(planner.formulate(state, goal));
  const auto actual = planner.solve(state, goal);
  check(expected.success() && actual.plan.has_value(), "fixed or independent MPC solve failed");
  check((actual.solver.x - expected.x).cwiseAbs().maxCoeff() < 2e-6, "fixed cache changed optimum");
}
}

int main() {
  try {
    const JointLimits limits{Vector::Constant(2, -1), Vector::Ones(2), Vector::Ones(2), Vector::Constant(2, 3)};
    const RobotState state{(Vector(2) << .1, -.2).finished(), Vector::Zero(2), 0};
    const Vector goal = (Vector(2) << .7, -.6).finished();
    auto shared = std::make_shared<OSQPSolver>();
    JointMPC first(limits, 5, .1, 2, .4, 9, shared);
    JointMPC second(limits, 5, .1, 7, 2, 12, shared);
    {
      JointMPC planner(limits, 60, .075, 7., .3, 14.);
      const Vector near_goal = state.q + (Vector(2) << .006, -.008).finished();
      const auto problem = planner.formulate(state, near_goal);
      const auto direct = planner.solve(state, near_goal);
      check(direct.plan && direct.solver.iterations == 0, "inactive MPC did not solve directly");
      const Vector values = problem.constraint * direct.solver.x;
      check((values.array() >= problem.lower.array()).all() &&
                (values.array() <= problem.upper.array()).all(),
            "inactive MPC violated exact QP bounds");
      check((problem.hessian * direct.solver.x + problem.gradient).cwiseAbs().maxCoeff() < 5e-14,
            "inactive MPC did not satisfy stationarity");
      same_optimum(planner, state, goal);
      planner.reset();
      same_optimum(planner, state, near_goal);
    }
    {
      const JointLimits tight{Vector::Constant(1, -1.), Vector::Ones(1),
                              Vector::Constant(1, .5 - 2e-9), Vector()};
      JointMPC planner(tight, 1);
      const RobotState origin{Vector::Zero(1), Vector::Zero(1), 0};
      const Vector near_bound = Vector::Constant(1, .5 * (.1 + .1 * .1 * 30.) / (.1 * 30.));
      const auto constrained = planner.solve(origin, near_bound);
      check(constrained.plan && constrained.solver.iterations > 0,
            "inactive MPC bypassed a slightly violated velocity bound");
    }
    {
      JointMPC planner(limits, 60, .004);
      const RobotState initial{Vector::Zero(2), Vector::Zero(2), 0};
      const auto goal = (Vector(2) << 20., -20.).finished();
      const auto initial_bytes = planner.fixed_matrix_bytes();
      const auto outcome = planner.solve(initial, goal);
      check(outcome.plan && outcome.solver.x.size() == 120,
            "short-step condensed MPC did not return a velocity-only solution");
      check(planner.fixed_matrix_bytes() > initial_bytes,
            "fixed storage omitted the short-step solver's matrices");
      const auto problem = planner.formulate(initial, goal);
      const Vector values = problem.constraint * outcome.solver.x;
      check((problem.lower - values).maxCoeff() <= 1e-6 &&
                (values - problem.upper).maxCoeff() <= 1e-6,
            "short-step sparse solution violates condensed bounds");
    }
    for (int k = 0; k < 10; ++k) {
      same_optimum(first, state, goal);
      same_optimum(second, state, -goal);
      // Generic QPs share the solver without acquiring trusted fixed identity.
      auto generic = first.formulate(state, goal);
      generic.hessian *= 2;
      generic.constraint *= 2;
      auto result = shared->solve(generic);
      check(result.success(), "generic/fixed cache transition failed");
      same_optimum(first, state, goal);
      generic.hessian.setIdentity(); generic.hessian(0, 0) = -1;
      bool rejected = false;
      try { shared->solve(generic); } catch (const std::invalid_argument&) { rejected = true; }
      check(rejected, "fixed cache bypassed mutated generic convexity validation");
      same_optimum(first, state, goal);
      second.reset();
      same_optimum(first, state, goal);
    }
    auto diagnostic = first.formulate(state, goal);
    diagnostic.hessian.setConstant(100);
    diagnostic.constraint.setZero();
    same_optimum(first, state, goal);

    // Reuse the exact planner object storage with different settings. The solver
    // must retain the old template's lifetime rather than trust an owner address.
    std::optional<JointMPC> recycled;
    for (int k = 0; k < 20; ++k) {
      recycled.emplace(limits, 5, .1, 1. + k, .2 + .1 * k, 2. + k, shared);
      same_optimum(*recycled, state, goal);
      recycled.reset();
    }
    auto custom = std::make_shared<CustomSolver>();
    JointMPC external(limits, 5, .1, 2, .4, 9, custom);
    same_optimum(external, state, goal);
    same_optimum(external, state, -goal);
    check(custom->calls == 2, "custom solver was bypassed");
    external.reset();
    check(custom->resets == 1, "custom reset was bypassed");
    custom->malformed = true;
    const auto malformed = external.solve(state, goal);
    check(!malformed.plan && !malformed.solver.success(), "malformed custom result escaped");
    custom->malformed = false;
    same_optimum(external, state, goal);
    for (double velocity : {.5, 2., std::numeric_limits<double>::quiet_NaN()}) {
      custom->forced_velocity = velocity;
      const auto unsafe = external.solve(state, goal);
      check(!unsafe.plan && !unsafe.solver.success() && unsafe.solver.violation > 1e-6,
            "custom solved vector bypassed velocity/slew validation");
    }
    custom->forced_velocity = 0;
    const auto dense_diagnostic = first.formulate(state, goal);
    check(first.fixed_matrix_bytes() < static_cast<std::size_t>(
              (dense_diagnostic.hessian.size() + dense_diagnostic.constraint.size()) * sizeof(double)),
          "condensed model matrices lost compressed storage");
    same_optimum(external, state, goal);
    JointLimits no_slew = limits;
    no_slew.acceleration.resize(0);
    JointMPC position_check(no_slew, 5, .1, 2, .4, 9, custom);
    custom->forced_velocity = .5;
    const auto unsafe_position = position_check.solve({Vector::Constant(2, .99), Vector::Zero(2), 0}, goal);
    check(!unsafe_position.plan && !unsafe_position.solver.success(), "custom solved vector bypassed position validation");
    custom->forced_velocity = 0;
    std::cout << "Fixed MPC shared ownership, public QP validation and custom solver checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
