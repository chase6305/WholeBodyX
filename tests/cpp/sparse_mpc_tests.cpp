#include "wholebodyx/reference.hpp"
#include "wholebodyx/sparse_mpc.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace wholebodyx;
namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Call> void rejects(Call call, const char* message) {
  bool rejected = false;
  try { call(); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, message);
}
}

int main() {
  try {
    JointLimits one{Vector::Constant(1, -10), Vector::Constant(1, 10), Vector::Constant(1, 10), {}};
    SparseJointMPC analytic(one, 1, .2, 2, .5, 3);
    RobotState initial{Vector::Zero(1), Vector::Zero(1), 0};
    auto solved = analytic.solve(initial, Vector::Ones(1));
    const double exact = .2 * 3 / (.2 * .2 * 3 + .5);
    check(solved.plan && std::abs(solved.plan->velocities(0, 0) - exact) < 1e-6, "one-step analytic optimum");
    check(solved.solver.x.size() == 2, "uncondensed decision order");
    check(std::abs(solved.solver.x[0] - .2 * solved.solver.x[1]) < 1e-6, "uncondensed dynamics");

    {
      const Vector origin = (Vector(3) << 1000., -1100., 0.).finished();
      const Vector radius = Vector::Constant(3, 2.);
      const Vector velocity = Vector::Ones(3), acceleration = Vector::Constant(3, 3.);
      SparseJointMPC absolute({origin - radius, origin + radius, velocity, acceleration});
      SparseJointMPC reference({-radius, radius, velocity, acceleration});
      const Vector v = (Vector(3) << .04, -.02, 0.).finished();
      for (double stamp : {0., .08, .16}) {
        const Vector measured = origin + (Vector(3) << .15, -.2, .1).finished();
        const Vector goal = origin + (Vector(3) << .75 - 4 * stamp, -.55 + 5 * stamp, .2).finished();
        const auto problem = absolute.formulate({measured, v, stamp}, goal);
        const auto actual = absolute.solve({measured, v, stamp}, goal);
        const auto expected = reference.solve({measured - origin, v, stamp}, goal - origin);
        check(actual.plan && expected.plan, "sparse solve failed with distant joint coordinates");
        check((problem.lower.head(3).array() == measured.array()).all(), "formulation changed origin");
        check((actual.plan->velocities - expected.plan->velocities).cwiseAbs().maxCoeff() < 1e-7,
              "centering changed velocities");
        const Vector values = problem.constraint * actual.solver.x;
        const double violation = std::max({(problem.lower - values).maxCoeff(),
                                           (values - problem.upper).maxCoeff(), 0.});
        check(violation == actual.solver.violation && violation <= 1e-6,
              "solver decision or residual did not use physical coordinates");
      }
    }

    std::mt19937 random(137);
    std::uniform_real_distribution<double> uniform(-1., 1.);
    for (bool slew : {false, true}) {
      JointLimits limits{Vector::Constant(3, -1), Vector::Constant(3, 1), Vector::Constant(3, .7),
                         slew ? Vector::Constant(3, 3.) : Vector()};
      for (int horizon : {1, 5, 20}) {
        JointMPC dense(limits, horizon, .08, 2.7, .4, 9.1);
        SparseJointMPC sparse(limits, horizon, .08, 2.7, .4, 9.1);
        RobotState state{Vector::Zero(3), Vector::Zero(3), 0};
        for (int k = 0; k < 25; ++k) {
          Vector goal(3);
          for (int i = 0; i < 3; ++i) goal[i] = 2 * uniform(random);
          auto a = dense.solve(state, goal), b = sparse.solve(state, goal);
          check(a.plan && b.plan, "dense/sparse randomized solve failed");
          check((a.plan->velocities - b.plan->velocities).cwiseAbs().maxCoeff() < 3e-6, "dense/sparse optimum mismatch");
          check((a.plan->positions - b.plan->positions).cwiseAbs().maxCoeff() < 2e-6, "dense/sparse positions mismatch");
          b.plan->validate();
          const auto reference = a.plan->sample(state.stamp + .04);
          state = {reference.first, reference.second, state.stamp + .04};
        }
        auto snapshot = sparse.formulate(state, Vector::Zero(3));
        const Matrix expected = snapshot.constraint;
        snapshot.constraint.setZero();
        check((Matrix(sparse.formulate(state, Vector::Zero(3)).constraint) - expected).norm() == 0,
              "formulation mutation changed planner matrices");
        auto warm = sparse.solve(state, Vector::Zero(3));
        sparse.reset();
        auto cold = sparse.solve(state, Vector::Zero(3));
        check(warm.plan && cold.plan && (warm.plan->positions - cold.plan->positions).norm() < 1e-6, "reset changed optimum");
      }
    }
    JointLimits bounds{Vector::Constant(1, -1), Vector::Ones(1), Vector::Ones(1), Vector::Ones(1)};
    SparseJointMPC planner(bounds, 5);
    auto failed = planner.solve({Vector::Constant(1, 3.), Vector::Zero(1), 0}, Vector::Zero(1));
    check(!failed.plan && !failed.solver.success() && failed.solver.x.size() == 0, "infeasible plan escaped");
    check(planner.solve(initial, Vector::Constant(1, .3)).plan.has_value(), "failure recovery");
    rejects([&] { planner.solve(initial, Vector::Zero(2)); }, "wrong goal dimension accepted");
    rejects([&] { planner.solve({Vector::Zero(1), Vector::Zero(1), 1e16}, Vector::Zero(1)); }, "unresolved timestamp accepted");
    rejects([&] { SparseJointMPC bad(bounds, 0); }, "zero horizon accepted");
    rejects([&] { SparseJointMPC bad(bounds, 1, .1, -1); }, "negative cost accepted");
    rejects([&] { SparseJointMPC bad(bounds, std::numeric_limits<int>::max()); }, "overflowing sparse indices accepted");
    MPCReferenceManager references(planner, bounds);
    auto update = references.update(initial, Vector::Constant(1, .3), .02);
    check(update.success() && update.replanned(), "sparse Planner reference integration");
    references.reset();
    check(!references.plan(), "sparse reference reset");
    std::cout << "Sparse MPC analytic/random optima, snapshots, bounds, recovery and references passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
