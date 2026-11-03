#include "wholebodyx/control.hpp"
#include "wholebodyx/sparse_mpc.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace wholebodyx;

PYBIND11_MODULE(wholebodyx_native, m) {
  m.doc() = "WholeBodyX optional C++ control kernels";
  py::class_<RobotState>(m, "RobotState")
      .def(py::init([](Vector q, Vector v, double stamp) { return RobotState{q, v, stamp}; }))
      .def_readonly("q", &RobotState::q).def_readonly("v", &RobotState::v).def_readonly("stamp", &RobotState::stamp);
  py::class_<JointLimits>(m, "JointLimits")
      .def(py::init([](Vector l, Vector u, Vector v, Vector a) { return JointLimits{l, u, v, a}; }));
  py::class_<LinearTask>(m, "LinearTask")
      .def(py::init([](Matrix j, Vector target, Vector weight) { return LinearTask{j, target, weight}; }));
  py::class_<QuadraticProgram>(m, "QuadraticProgram")
      .def_readonly("hessian", &QuadraticProgram::hessian).def_readonly("gradient", &QuadraticProgram::gradient)
      .def_readonly("constraint", &QuadraticProgram::constraint).def_readonly("lower", &QuadraticProgram::lower)
      .def_readonly("upper", &QuadraticProgram::upper);
  py::class_<QPResult>(m, "QPResult")
      .def_readonly("status", &QPResult::status).def_readonly("x", &QPResult::x)
      .def_readonly("iterations", &QPResult::iterations).def_readonly("violation", &QPResult::violation)
      .def_readonly("detail", &QPResult::detail).def_property_readonly("success", &QPResult::success);
  py::class_<VelocityCommand>(m, "VelocityCommand")
      .def_readonly("velocity", &VelocityCommand::velocity).def_readonly("stamp", &VelocityCommand::stamp)
      .def_readonly("valid_until", &VelocityCommand::valid_until);
  py::class_<ControlResult>(m, "ControlResult")
      .def_readonly("solver", &ControlResult::solver).def_readonly("command", &ControlResult::command);
  py::class_<JointPlan>(m, "JointPlan")
      .def(py::init([](Matrix q, Matrix v, double stamp, double dt) {
        JointPlan plan{std::move(q), std::move(v), stamp, dt};
        plan.validate();
        return plan;
      }))
      .def_readonly("positions", &JointPlan::positions).def_readonly("velocities", &JointPlan::velocities)
      .def_readonly("stamp", &JointPlan::stamp).def_readonly("dt", &JointPlan::dt)
      .def_property_readonly("valid_until", &JointPlan::valid_until).def("sample", &JointPlan::sample);
  py::class_<PlanResult>(m, "PlanResult")
      .def_readonly("solver", &PlanResult::solver).def_readonly("plan", &PlanResult::plan);
  py::class_<KinematicWBC>(m, "KinematicWBC")
      .def(py::init([](JointLimits l, double r, double s) { return std::make_unique<KinematicWBC>(l, r, s); }))
      .def("reset", &KinematicWBC::reset)
      .def("formulate", &KinematicWBC::formulate)
      .def("step", &KinematicWBC::step, py::call_guard<py::gil_scoped_release>());
  py::class_<JointMPC>(m, "JointMPC")
      .def(py::init([](JointLimits l, int n, double dt, double q, double r, double f) {
        return std::make_unique<JointMPC>(l, n, dt, q, r, f);
      }))
      .def("reset", &JointMPC::reset).def("formulate", &JointMPC::formulate)
      .def_property_readonly("fixed_matrix_bytes", &JointMPC::fixed_matrix_bytes)
      .def("solve", &JointMPC::solve, py::call_guard<py::gil_scoped_release>());
  py::class_<SparseMPCProblem>(m, "SparseMPCProblem")
      .def_readonly("hessian", &SparseMPCProblem::hessian).def_readonly("gradient", &SparseMPCProblem::gradient)
      .def_readonly("constraint", &SparseMPCProblem::constraint).def_readonly("lower", &SparseMPCProblem::lower)
      .def_readonly("upper", &SparseMPCProblem::upper);
  py::class_<SparseJointMPC>(m, "SparseJointMPC")
      .def(py::init<JointLimits, int, double, double, double, double>())
      .def("reset", &SparseJointMPC::reset).def("formulate", &SparseJointMPC::formulate)
      .def("solve", &SparseJointMPC::solve, py::call_guard<py::gil_scoped_release>());
}
