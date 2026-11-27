"""Checked working-set solve for the fixed-contact inverse-dynamics QP."""

from typing import NamedTuple

import numpy as np
from scipy.linalg.lapack import dsysv


class ScaledContactQP(NamedTuple):
    hessian: np.ndarray
    gradient: np.ndarray
    constraint: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


ACTIVE_TOLERANCE = 1e-8
_DUAL_SIGN_TOLERANCE = 1e-9
_ACTIVE_SET_ATTEMPTS = 5


def constraint_violation(constraint, lower, upper, solution):
    values = constraint @ solution
    if not np.isfinite(values).all():
        return float("inf")
    return float(max((lower - values).max(initial=0.0), (values - upper).max(initial=0.0)))


def solve_contact_working_set(problem, dual_seed):
    """Return a checked scaled primal/dual pair, or None for OSQP to continue."""
    hessian, gradient, constraint, lower, upper = problem
    if dual_seed is None or dual_seed.shape != lower.shape or not np.isfinite(dual_seed).all():
        return None
    equality = (lower == upper) & np.isfinite(lower)
    active = np.zeros(lower.shape, dtype=np.int8)
    active[(dual_seed > _DUAL_SIGN_TOLERANCE) & np.isfinite(upper)] = 1
    active[(dual_seed < -_DUAL_SIGN_TOLERANCE) & np.isfinite(lower)] = -1
    active[equality] = -1
    bounds = np.where(active > 0, upper, lower)
    variables = gradient.size
    for _ in range(_ACTIVE_SET_ATTEMPTS):
        rows = np.flatnonzero(active)
        matrix = constraint[rows]
        # DSYSV reads the lower triangle of [H, A.T; A, 0] and pivots as needed.
        size = variables + rows.size
        kkt = np.zeros((size, size), order="F")
        kkt[:variables, :variables] = hessian
        kkt[variables:, :variables] = matrix
        rhs = np.empty((size, 1))
        rhs[:variables, 0] = -gradient
        rhs[variables:, 0] = bounds[rows]
        _, _, answer, status = dsysv(kkt, rhs, lower=True, overwrite_a=True, overwrite_b=True)
        if status or not np.isfinite(answer).all():
            return None
        x, multipliers = answer[:variables, 0], answer[variables:, 0]
        values = constraint @ x
        bound_error = values[rows] - bounds[rows]
        stationarity = hessian @ x + gradient + matrix.T @ multipliers
        if not (
            np.isfinite(values).all()
            and np.max(np.abs(stationarity), initial=0.0) <= ACTIVE_TOLERANCE
            and np.max(np.abs(bound_error), initial=0.0) <= ACTIVE_TOLERANCE
            and np.max(np.abs(bound_error * multipliers), initial=0.0) <= ACTIVE_TOLERANCE
        ):
            return None

        wrong_sign = -(multipliers * active[rows])
        wrong_sign[equality[rows]] = 0.0
        if wrong_sign.max(initial=0.0) > 0.0:
            active[rows[np.argmax(wrong_sign)]] = 0
            continue
        lower_gap, upper_gap = lower - values, values - upper
        # The active side already passed its equality check. Keep exact
        # comparisons for the other bounds: a small positive gap on a weak
        # row can still be far from the optimal decision.
        lower_gap[active < 0] = -np.inf
        upper_gap[(active > 0) | equality] = -np.inf
        lower_index, upper_index = np.argmax(lower_gap), np.argmax(upper_gap)
        if max(lower_gap[lower_index], upper_gap[upper_index]) <= 0.0:
            dual = np.zeros_like(lower)
            dual[rows] = multipliers
            return x, dual
        if lower_gap[lower_index] > upper_gap[upper_index]:
            active[lower_index], bounds[lower_index] = -1, lower[lower_index]
        else:
            active[upper_index], bounds[upper_index] = 1, upper[upper_index]
    return None
