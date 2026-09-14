#pragma once

#include "wholebodyx/control.hpp"

#include <functional>

namespace wholebodyx {

// One thread owns every method call. Drivers exchange model-order values only.
class VelocityRobotBackend {
 public:
  virtual ~VelocityRobotBackend() = default;
  virtual RobotState read() = 0;
  virtual void write(const VelocityCommand&) = 0;
  virtual void stop() = 0;
};

// Validates a fresh measured state and a candidate velocity command. It does
// not write to the driver and does not change the controller's algorithm state.
class CommandGate {
 public:
  CommandGate(JointLimits, double control_dt);
  void reset();
  void latch_fault(std::string detail);
  const std::string& fault() const { return fault_; }
  RobotState observe(const RobotState&);
  VelocityCommand validate(const RobotState&, const VelocityCommand&) const;

 private:
  JointLimits limits_;
  double control_dt_;
  std::optional<double> last_stamp_, observed_stamp_;
  std::string fault_;
};

struct CycleResult {
  RobotState state;
  ControlResult control;
  double control_ms = 0.0;
};

// A synchronous cycle: read → construct tasks → WBC → gate → write. Any
// exception latches a fault and attempts one backend stop; reset before reuse.
class VelocityControlLoop {
 public:
  using TaskFactory = std::function<std::vector<LinearTask>(const RobotState&)>;
  VelocityControlLoop(VelocityRobotBackend&, Controller&, JointLimits, double control_dt);
  void reset();
  const std::string& fault() const { return gate_.fault(); }
  CycleResult step(const TaskFactory&);

 private:
  [[noreturn]] void fail(const std::exception&);
  VelocityRobotBackend& backend_;
  Controller& controller_;
  CommandGate gate_;
  double control_dt_;
};
}  // namespace wholebodyx
