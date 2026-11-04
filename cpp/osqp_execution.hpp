#pragma once

#include <osqp/osqp.h>
#include <mutex>
#include <type_traits>

namespace wholebodyx::detail {

static_assert(std::is_same<c_float, double>::value, "WholeBodyX requires double-precision OSQP");

inline OSQPSettings osqp_settings() {
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0; settings.eps_abs = 1e-8; settings.eps_rel = 1e-8;
  settings.eps_prim_inf = 1e-8; settings.eps_dual_inf = 1e-8;
  settings.max_iter = 20000; settings.polish = 0; settings.warm_start = 1;
  return settings;
}

inline const char* rejected_osqp_status(c_int status) {
  if (status == OSQP_PRIMAL_INFEASIBLE || status == OSQP_DUAL_INFEASIBLE ||
      status == OSQP_PRIMAL_INFEASIBLE_INACCURATE || status == OSQP_DUAL_INFEASIBLE_INACCURATE)
    return "infeasible";
  return status == OSQP_MAX_ITER_REACHED ? "iteration_limit" : "error";
}

inline c_int execute_osqp(OSQPWorkspace* work) {
#ifdef CTRLC
  // OSQP 0.6 saves/restores the process SIGINT handler in global storage.
  // Independent workspaces must not overlap these listener lifetimes. Keep
  // this lock shared by the dense and sparse entry points, including bindings
  // that release the GIL. It does not coordinate unrelated third-party OSQP
  // libraries or provide a preemptible scheduling/deadline guarantee.
  static std::mutex interrupt_listener;
  const std::lock_guard<std::mutex> lock(interrupt_listener);
#endif
  return osqp_solve(work);
}

}  // namespace wholebodyx::detail
