#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wholebodyx::detail {

inline double ulp(double value) {
  const double next = std::nextafter(value, INFINITY);
  return std::isfinite(next) ? next - value : value - std::nextafter(value, -INFINITY);
}

inline double time_window(double stamp, double dt, Eigen::Index count) {
  const double end = stamp + dt * count;
  if (!(std::isfinite(stamp) && stamp >= 0 && std::isfinite(end) && end > stamp &&
        dt > 8 * std::max(ulp(stamp), ulp(end))))
    throw std::invalid_argument("plan time window must be finite and resolve each time step");
  return end;
}

inline Eigen::Index interval(double time, double start, double dt, Eigen::Index count) {
  if (time >= start + count * dt) return count;
  const double elapsed = (time - start) / dt;
  const auto nearest = static_cast<Eigen::Index>(std::round(elapsed));
  const double knot = start + nearest * dt;
  if (std::abs(time - knot) <= 4 * std::max(ulp(time), ulp(knot))) return std::min(nearest, count - 1);
  return std::min(static_cast<Eigen::Index>(elapsed), count - 1);
}

}  // namespace wholebodyx::detail
