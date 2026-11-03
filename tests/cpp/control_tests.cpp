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
    // Same four-step weak-constraint fixture as tests/data/scaled_qp_weak_constraint.json.
    {
      Matrix h(3,3),a(6,3);Vector scale(3);
      h << 1.2433763372515545,-0.16175143146111184,-0.21466110675384828,-0.16175143146111184,2.026322680540485,0.8041543979203117,-0.21466110675384828,0.8041543979203117,1.669587122259462;
      a << 1.458659270914995,-2.763093111109902,0.0,-1.0296005948033065,0.0,0.0,-1.970444552908006,-0.6729936510544398,-2.2883669697364195,-0.00031125791256597006,0.0,0.0,0.9037960599240652,-1.8562610356027633,0.0,0.7475220929175335,-0.546122769528383,1.1220224780325383;
      scale << 0.2,0.7745966692414834,3.0;
      h=scale.asDiagonal()*h*scale.asDiagonal();a=a*scale.asDiagonal();OSQPSolver solver;
      {Vector g(3),l(6),u(6),expected(3);g << 0.47998123768084644,0.14538261224098162,-0.05555037747317496;l << -0.9459868401907392,-0.2964974282306695,-1.6519618543106929,-0.23627180180858728,-1.518478577838124,-0.11664991148582748;u << 2.2335714851102706,-0.16255673705166876,-0.03550413888149129,0.22326569487456205,1.6005040123669656,0.40783705644242196;expected << 0.15788329753512176,-0.09940451431226416,0.10144924811598355;
      QuadraticProgram p{h,scale.asDiagonal()*g,a,l,u};auto result=solver.solve(p);check(result.success(), "weak feasible constraint falsely rejected");check((scale.asDiagonal()*result.x-expected).cwiseAbs().maxCoeff()<1e-6, "weak constraint optimum differs");}
      {Vector g(3),l(6),u(6),expected(3);g << -0.5879845173471125,0.4915763679643912,0.037819269113322104;l << -0.8084267778259446,-1.7790799327945461,-1.863368038569567,-1.9149533580716698,-1.3722100423423238,-1.7627663221178576;u << -0.5108042500046339,1.8122285607224133,1.8357466227630037,0.2985813514652814,1.6179885694555267,0.6188601932676275;expected << 0.1142603497536735,0.2451857904269782,-0.12605466636070084;
      QuadraticProgram p{h,scale.asDiagonal()*g,a,l,u};auto result=solver.solve(p);check(result.success(), "weak feasible constraint falsely rejected");check((scale.asDiagonal()*result.x-expected).cwiseAbs().maxCoeff()<1e-6, "weak constraint optimum differs");}
      {Vector g(3),l(6),u(6),expected(3);g << -1.3049406807806103,-0.5246186691369124,-0.5698605830817992;l << -1.4051202096341793,-0.448183145558937,-2.3306030192063916,-1.9693995205800034,-1.104587715339929,-0.3458017528477981;u << 1.2461109939561215,0.9291983919762273,-1.269084420415561,0.7630023951141159,0.9774484123506015,2.475767785787792;expected << 0.4352980639493097,0.16812061925432942,0.31631006544170015;
      QuadraticProgram p{h,scale.asDiagonal()*g,a,l,u};auto result=solver.solve(p);check(result.success(), "weak feasible constraint falsely rejected");check((scale.asDiagonal()*result.x-expected).cwiseAbs().maxCoeff()<1e-6, "weak constraint optimum differs");}
      {Vector g(3),l(6),u(6),expected(3);g << 0.21975164252926815,-0.065107587766042,0.29864143101368096;l << -1.2967724727028047,-1.3800644763515526,0.32205216100183365,6.434087411578278e-05,-0.6575402352070576,-0.8893372939183116;u << -0.45642909719444846,1.5000252190101708,1.0851801699356176,1.3844532004156456,1.1079259275909874,1.0294019104673555;expected << -0.20671241281985386,0.12012429509775968,-0.2633064310740245;
      QuadraticProgram p{h,scale.asDiagonal()*g,a,l,u};auto result=solver.solve(p);check(result.success(), "weak feasible constraint falsely rejected");check((scale.asDiagonal()*result.x-expected).cwiseAbs().maxCoeff()<1e-6, "weak constraint optimum differs");}
    }
    std::cout << "C++ constrained optimum, infeasibility, MPC and reference checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
