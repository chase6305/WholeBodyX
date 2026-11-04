"""Paced joint MPC in a spawned process, with WBC in the owner process."""

import argparse
import json
import time

import numpy as np

from wholebodyx import (AsyncMPCReferenceManager, BackendRegistry, JointPlan,
                        ProcessMPCWorker, RobotState, VelocitySession)
from wholebodyx.model import PlanarDualArm
from wholebodyx.simulation import JointIntegrator
from wholebodyx.types import positive


def run(backend="sparse", seconds=6., control_dt=.02, lookahead=.2):
    seconds = positive(seconds, "duration")
    control_dt = positive(control_dt, "control dt")
    model = PlanarDualArm()
    goal = np.array([.3, -.2, .1, -.3, .2, -.1])
    driver = JointIntegrator(model.limits, RobotState(np.zeros(6), np.zeros(6)), control_dt=control_dt)
    controller = BackendRegistry().create_controller("native" if backend.startswith("native") else "python", model.limits)
    steps = adopted = 0
    max_cycle_ms = max_solve_ms = 0.
    # The startup barrier and teardown execute outside the measured cycle.
    with ProcessMPCWorker(model.limits, backend=backend, horizon=60, dt=control_dt) as worker:
        references = AsyncMPCReferenceManager(worker, lookahead=lookahead, max_adoption_delay=0.)
        session = VelocitySession(model, driver, controller, control_dt, reference_manager=references)
        session.reset()
        measured = driver.read()  # Quiescent simulation; a real driver needs its own startup policy.
        hold = JointPlan(np.tile(measured.q, (61, 1)), np.zeros((60, 6)), measured.stamp, control_dt)
        references.seed(hold, measured, goal, control_dt)
        started = due = time.monotonic()
        while time.monotonic() - started < seconds:
            step = session.step(goal)
            steps += 1
            adopted += int(step.reference.replanned)
            max_cycle_ms = max(max_cycle_ms, step.cycle.cycle_ms)
            if step.reference.replanned:
                max_solve_ms = max(max_solve_ms, step.reference.solve_ms)
            due += control_dt
            now = time.monotonic()
            if due < now - control_dt:
                due = now
            if due > now:
                time.sleep(due - now)
        final = driver.read()
        stats = worker.statistics
    return dict(backend=backend, steps=steps, adopted_plans=adopted,
                final_joint_error_rad=float(np.max(np.abs(final.q - goal))),
                max_control_cycle_ms=max_cycle_ms, max_adopted_child_solve_ms=max_solve_ms,
                worker_statistics=stats, worker_reaped=worker._shutdown_complete)


if __name__ == "__main__":  # Required by the spawn start method.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native", "sparse", "native-sparse"), default="sparse")
    parser.add_argument("--seconds", type=float, default=6.)
    parser.add_argument("--control-dt", type=float, default=.02)
    parser.add_argument("--lookahead", type=float, default=.2)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
