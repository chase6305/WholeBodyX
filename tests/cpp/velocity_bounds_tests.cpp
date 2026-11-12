#include "wholebodyx/control.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace wholebodyx;

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent oracle: simulate held commands under maximum braking, then
// bisect the largest starting speed that stays within the available distance.
double rollout_speed(double distance, double maximum, double acceleration, double dt) {
  double left = 0., right = maximum;
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double speed = .5 * left + .5 * right;
    double travel = 0.;
    for (double command = speed; command > 0.; command -= acceleration * dt)
      travel += dt * command;
    if (travel <= std::max(0., distance))
      left = speed;
    else
      right = speed;
  }
  return left;
}

void compare_rollout(const JointLimits& limits, const RobotState& state, double dt) {
  const auto bounds = limits.velocity_bounds(state, dt);
  for (Eigen::Index i = 0; i < state.q.size(); ++i) {
    const double lower = std::max({-limits.velocity[i],
        (limits.lower[i] - state.q[i]) / dt, state.v[i] - limits.acceleration[i] * dt,
        -rollout_speed(state.q[i] - limits.lower[i], limits.velocity[i], limits.acceleration[i], dt)});
    const double upper = std::min({limits.velocity[i],
        (limits.upper[i] - state.q[i]) / dt, state.v[i] + limits.acceleration[i] * dt,
        rollout_speed(limits.upper[i] - state.q[i], limits.velocity[i], limits.acceleration[i], dt)});
    check(std::abs(bounds.first[i] - lower) <= 1e-11, "lower bound differs from braking rollout");
    check(std::abs(bounds.second[i] - upper) <= 1e-11, "upper bound differs from braking rollout");
  }
}
}  // namespace

int main() {
  try {
    std::mt19937_64 rng(20261112);
    std::uniform_real_distribution<double> unit(0., 1.);
    for (int iteration = 0; iteration < 300; ++iteration) {
      const int n = 6;
      JointLimits limits{Vector(n), Vector(n), Vector(n), Vector(n)};
      RobotState state{Vector(n), Vector(n), 0.};
      for (int i = 0; i < n; ++i) {
        limits.lower[i] = -(.5 + 2. * unit(rng));
        limits.upper[i] = .5 + 2. * unit(rng);
        limits.velocity[i] = .1 + unit(rng);
        limits.acceleration[i] = .5 + 5. * unit(rng);
        state.q[i] = limits.lower[i] - .2 +
            unit(rng) * (limits.upper[i] - limits.lower[i] + .4);
        state.v[i] = (2. * unit(rng) - 1.) * limits.velocity[i];
      }
      compare_rollout(limits, state, .005 + .2 * unit(rng));
    }

    // Straddle the shortcut's stopping-distance threshold on both sides.
    const JointLimits limits{Vector::Constant(1, -10.), Vector::Constant(1, 10.),
                             Vector::Constant(1, .7), Vector::Constant(1, 2.)};
    const double dt = .02, travel = .7 * (.7 / 2. + dt);
    for (double direction : {-1., 1.}) {
      for (double distance : {std::nextafter(travel, 0.), travel,
                              std::nextafter(travel, INFINITY), .005, 5.}) {
        auto boundary = limits;
        if (direction < 0.)
          boundary.lower[0] = -distance;
        else
          boundary.upper[0] = distance;
        // Keep q at zero so subtracting a much larger limit cannot round
        // these neighboring stopping distances back to the same value.
        compare_rollout(boundary, {Vector::Zero(1),
                                    Vector::Constant(1, direction * .5), 0.}, dt);
      }
    }

    // Retain the existing envelope and roundoff normalization when the slew
    // product falls below double range, even far from the position limits.
    const JointLimits tiny{Vector::Constant(1, -1.), Vector::Ones(1),
                           Vector::Constant(1, 1e-200), Vector::Constant(1, 1e-300)};
    for (double direction : {-1., 1.}) {
      const auto bounds = tiny.velocity_bounds(
          {Vector::Zero(1), Vector::Constant(1, direction * 1e-200), 0.}, 1e-100);
      const double normalized = direction * .5e-200;
      check(bounds.first[0] == normalized && bounds.second[0] == normalized,
            "zero-slew envelope changed");
    }

    // This finite stopping distance exceeds the range of double arithmetic.
    const JointLimits huge{Vector::Constant(1, -1e300), Vector::Constant(1, 1e300),
                           Vector::Constant(1, 1e200), Vector::Constant(1, 1e-100)};
    const auto bounds = huge.velocity_bounds({Vector::Zero(1), Vector::Zero(1), 0.}, .01);
    check(bounds.first.allFinite() && bounds.second.allFinite(), "nonfinite extreme bounds");
    check(bounds.first[0] <= 0. && bounds.second[0] >= 0., "extreme bounds exclude rest");
    std::cout << "C++ velocity bounds match independent braking rollouts\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
