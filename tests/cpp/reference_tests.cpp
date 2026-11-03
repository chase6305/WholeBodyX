#include "wholebodyx/reference.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace wholebodyx;
namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
class InspectPlanner : public Planner {
 public:
  explicit InspectPlanner(JointLimits limits) : implementation(std::move(limits)) {}
  JointMPC implementation;
  int calls = 0, resets = 0, failure = 0;
  PlanResult latest;
  void reset() override {
    ++resets;
    if (failure == 4) throw std::runtime_error("reset failed");
    implementation.reset();
  }
  PlanResult solve(const RobotState& state, const Vector& goal) override {
    ++calls;
    if (failure == 1) return {{"infeasible", {}, 0, INFINITY, "injected"}, std::nullopt};
    latest = implementation.solve(state, goal);
    if (failure == 2 && latest.plan) latest.plan->stamp += .01;
    if (failure == 3 && latest.plan) latest.plan->positions.col(0).array() += .1;
    if (failure == 5) throw 7;
    return latest;
  }
};
}  // namespace

int main() {
  try {
    JointLimits limits{Vector::Constant(2, -1), Vector::Constant(2, 1), Vector::Ones(2), Vector::Constant(2, 4)};
    Vector goal = Vector::Constant(2, .3);
    {
      InspectPlanner coarse_clock(limits);
      MPCReferenceManager guarded(coarse_clock, limits);
      const double origin = 1e8;
      const double tick = std::nextafter(origin, INFINITY) - origin;
      RobotState measured{Vector::Zero(2), Vector::Zero(2), origin};
      check(guarded.update(measured, Vector::Zero(2), tick).replanned(), "initial coarse-clock plan missing");
      measured.stamp += .05;
      check(guarded.update(measured, Vector::Zero(2), tick).reason == "reused",
            "roundoff tolerance caused a materially early replan");
    }
    for (double origin : {0., 1e8}) {
      InspectPlanner planner(limits);
      MPCReferenceManager references(planner, limits);
      RobotState state{Vector::Zero(2), Vector::Zero(2), origin};
      for (int tick = 0; tick <= 600; ++tick) {
        auto update = references.update(state, goal, .005);
        check(update.success(), "valid scheduled reference rejected");
        check(update.replanned() == (tick % 20 == 0), "5 ms state clock caused replan drift");
        const auto next = references.plan()->sample(state.stamp + .005);
        state = {next.first, next.second, state.stamp + .005};
      }
      check(planner.calls == 31, "unexpected planner count");
    }
    InspectPlanner planner(limits);
    MPCReferenceManager references(planner, limits);
    RobotState state{Vector::Zero(2), Vector::Zero(2), 0};
    auto first = references.update(state, goal, .02);
    check(first.replanned() && first.reason == "initial", "initial reference missing");
    planner.latest.plan->positions.setConstant(99);
    check(references.plan()->positions.maxCoeff() < 1, "planner workspace mutation changed accepted reference");
    state.stamp = .02;
    check(references.update(state, goal, .02).reason == "reused", "valid plan was not reused");
    goal[0] += .1;
    check(references.update(state, goal, .02).reason == "goal_changed", "target change did not refresh immediately");
    for (int failure : {1, 2, 3}) {
      planner.failure = failure;
      references.invalidate();
      check(!references.update(state, goal, .02).success() && !references.plan(), "failed or invalid plan survived");
    }
    planner.failure = 0;
    references.reset();
    state.stamp = 1.;
    check(references.update(state, goal, .02).success(), "reference did not recover");
    state.stamp = .9;
    check(references.update(state, goal, .02).reason == "clock_rollback", "rollback not detected");
    state.stamp = 2.;
    check(!references.update(state, goal, .02).success(), "rollback did not latch");
    planner.failure = 4;
    try { references.reset(); } catch (const std::runtime_error&) {}
    check(!references.update(state, goal, .02).success(), "failed reset cleared rollback latch");
    planner.failure = 0;
    references.reset();
    check(references.update(state, goal, .02).success(), "successful reset did not clear rollback latch");
    planner.failure = 5;
    references.invalidate();
    bool propagated = false;
    try { references.update(state, goal, .02); } catch (int) { propagated = true; }
    check(propagated && !references.plan(), "non-standard cancellation retained a reference");
    std::cout << "Reference scheduling, acceptance, clocks and recovery checks passed\n";
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << '\n';
    return 1;
  }
}
