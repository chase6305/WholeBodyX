#include "wholebodyx/sparse_mpc.hpp"
#include <atomic>
#include <cmath>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace wholebodyx;
namespace {
volatile std::sig_atomic_t interrupts = 0;
void handler(int) { ++interrupts; }
}

int main() {
  try {
    auto original = std::signal(SIGINT, handler);
    if (original == SIG_ERR) throw std::runtime_error("could not install test SIGINT handler");
    std::atomic<bool> failed{false};
    for (int batch = 0; batch < 4; ++batch) {
      std::vector<std::thread> threads;
      for (int index = 0; index < 4; ++index) threads.emplace_back([&, index] {
        try {
          JointLimits limits{Vector::Constant(3, -1), Vector::Ones(3), Vector::Ones(3), Vector::Constant(3, 4)};
          RobotState state{Vector::Zero(3), Vector::Zero(3), 0};
          std::unique_ptr<Planner> planner;
          if (index % 2) planner = std::make_unique<SparseJointMPC>(limits);
          else planner = std::make_unique<JointMPC>(limits);
          KinematicWBC controller(limits);
          for (int k = 0; k < 100; ++k) {
            auto plan = planner->solve(state, Vector::Constant(3, .3 + .01 * std::sin(k)));
            if (!plan.plan || plan.solver.violation > 1e-6) failed = true;
            auto command = controller.step(state, {{Matrix::Identity(3, 3), Vector::Constant(3, .2), Vector::Ones(3)}}, .02);
            if (!command.command || command.solver.violation > 1e-6) failed = true;
          }
        } catch (...) { failed = true; }
      });
      for (auto& thread : threads) thread.join();
      if (failed) throw std::runtime_error("independent concurrent solver failed");
      std::raise(SIGINT);
      if (interrupts != batch + 1) throw std::runtime_error("native solvers lost the caller's SIGINT handler");
    }
    std::signal(SIGINT, original);
    std::cout << "Independent native instances preserve solver results and the caller's SIGINT handler\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
