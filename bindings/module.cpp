#include "wholebodyx/control.hpp"
#include "wholebodyx/sparse_mpc.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;
using namespace wholebodyx;

PYBIND11_MODULE(wholebodyx_native, module) {
  module.doc() = "WholeBodyX optional C++ control kernels";
  py::class_<RobotState>(module, "RobotState")
      .def(py::init([](Vector position, Vector velocity, double stamp) {
        return RobotState{std::move(position), std::move(velocity), stamp};
      }))
      .def_readonly("q", &RobotState::q)
      .def_readonly("v", &RobotState::v)
      .def_readonly("stamp", &RobotState::stamp);
  py::class_<JointLimits>(module, "JointLimits")
      .def(py::init([](Vector lower, Vector upper, Vector velocity, Vector acceleration) {
        return JointLimits{std::move(lower), std::move(upper), std::move(velocity),
                           std::move(acceleration)};
      }));
  py::class_<LinearTask>(module, "LinearTask")
      .def(py::init([](Matrix jacobian, Vector target, Vector weight) {
        return LinearTask{std::move(jacobian), std::move(target), std::move(weight)};
      }));
  py::class_<QuadraticProgram>(module, "QuadraticProgram")
      .def_readonly("hessian", &QuadraticProgram::hessian)
      .def_readonly("gradient", &QuadraticProgram::gradient)
      .def_readonly("constraint", &QuadraticProgram::constraint)
      .def_readonly("lower", &QuadraticProgram::lower)
      .def_readonly("upper", &QuadraticProgram::upper);
  py::class_<QPResult>(module, "QPResult")
      .def_readonly("status", &QPResult::status)
      .def_readonly("x", &QPResult::x)
      .def_readonly("iterations", &QPResult::iterations)
      .def_readonly("violation", &QPResult::violation)
      .def_readonly("detail", &QPResult::detail)
      .def_property_readonly("success", &QPResult::success);
  py::class_<VelocityCommand>(module, "VelocityCommand")
      .def_readonly("velocity", &VelocityCommand::velocity)
      .def_readonly("stamp", &VelocityCommand::stamp)
      .def_readonly("valid_until", &VelocityCommand::valid_until);
  py::class_<ControlResult>(module, "ControlResult")
      .def_readonly("solver", &ControlResult::solver)
      .def_readonly("command", &ControlResult::command);
  py::class_<JointPlan>(module, "JointPlan")
      .def(py::init([](Matrix positions, Matrix velocities, double stamp, double dt) {
        JointPlan plan{std::move(positions), std::move(velocities), stamp, dt};
        plan.validate();
        return plan;
      }))
      .def_readonly("positions", &JointPlan::positions)
      .def_readonly("velocities", &JointPlan::velocities)
      .def_readonly("stamp", &JointPlan::stamp)
      .def_readonly("dt", &JointPlan::dt)
      .def_property_readonly("valid_until", &JointPlan::valid_until)
      .def("sample", &JointPlan::sample);
  py::class_<PlanResult>(module, "PlanResult")
      .def_readonly("solver", &PlanResult::solver)
      .def_readonly("plan", &PlanResult::plan);
  py::class_<KinematicWBC>(module, "KinematicWBC")
      .def(py::init([](JointLimits limits, double regularization, double smoothing) {
        return std::make_unique<KinematicWBC>(std::move(limits), regularization, smoothing);
      }))
      .def("reset", &KinematicWBC::reset)
      .def("formulate", &KinematicWBC::formulate)
      .def("step", &KinematicWBC::step, py::call_guard<py::gil_scoped_release>());
  py::class_<JointMPC>(module, "JointMPC")
      .def(py::init([](JointLimits limits, int horizon, double dt, double tracking_weight,
                       double control_weight, double terminal_weight) {
        return std::make_unique<JointMPC>(std::move(limits), horizon, dt, tracking_weight,
                                          control_weight, terminal_weight);
      }))
      .def("reset", &JointMPC::reset)
      .def("formulate", &JointMPC::formulate)
      .def_property_readonly("fixed_matrix_bytes", &JointMPC::fixed_matrix_bytes)
      .def("solve", &JointMPC::solve, py::call_guard<py::gil_scoped_release>());
  py::class_<SparseMPCProblem>(module, "SparseMPCProblem")
      .def_readonly("hessian", &SparseMPCProblem::hessian)
      .def_readonly("gradient", &SparseMPCProblem::gradient)
      .def_readonly("constraint", &SparseMPCProblem::constraint)
      .def_readonly("lower", &SparseMPCProblem::lower)
      .def_readonly("upper", &SparseMPCProblem::upper);
  py::class_<SparseJointMPC>(module, "SparseJointMPC")
      .def(py::init<JointLimits, int, double, double, double, double>())
      .def("reset", &SparseJointMPC::reset)
      .def("formulate", &SparseJointMPC::formulate)
      .def("solve", &SparseJointMPC::solve, py::call_guard<py::gil_scoped_release>());
}
