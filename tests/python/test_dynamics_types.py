"""Numeric floating-base contracts work without robot files or Pinocchio."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from wholebodyx import (CoMReference, DynamicsControlResult, DynamicsState,
                        QPResult, TorqueCommand)


def state_vectors():
    return np.r_[np.zeros(6), 1., .2], np.zeros(7)


def test_state_owns_values_and_uses_distinct_configuration_and_tangent_dimensions():
    q, v = state_vectors()
    state = DynamicsState(q, v, np.float32(1e8))
    q[:] = v[:] = 3.
    assert state.q[-1] == .2 and np.max(state.v) == 0
    assert type(state.stamp) is float and state.q.size == state.v.size + 1
    with pytest.raises(FrozenInstanceError):
        state.stamp = 3
    with pytest.raises(ValueError):
        state.q[0] = 9
    with pytest.raises(ValueError):
        state.q.setflags(write=True)


@pytest.mark.parametrize("q,v,stamp", [
    (np.zeros(8), np.zeros(7), 0),
    ([0, 0, 0, 0, 0, 0, 1], np.zeros(7), 0),
    (np.r_[np.zeros(6), 1., 0], np.zeros(7), -1),
    (np.r_[np.zeros(6), 1., 0], np.zeros(7), np.inf),
    (np.r_[np.zeros(6), 1., 0], np.full(7, np.nan), 0),
])
def test_malformed_floating_state_is_rejected_at_boundary(q, v, stamp):
    with pytest.raises(ValueError):
        DynamicsState(q, v, stamp)


def test_commands_and_references_are_time_bounded_numeric_snapshots():
    torque, target = np.array([3.]), np.array([0., 0., .8])
    command = TorqueCommand(torque, np.float32(2), 2.005)
    reference = CoMReference(target, 2, 2.005)
    torque[:] = 9
    target[:] = 9
    assert command.mode == "torque" and command.torque[0] == 3
    assert type(command.stamp) is float
    np.testing.assert_array_equal(reference.position, [0, 0, .8])
    assert not np.any(reference.velocity) and not np.any(reference.acceleration)
    other = CoMReference([0, 0, .8], 2, 2.005)
    assert not np.shares_memory(reference.velocity, other.velocity)
    with pytest.raises(ValueError):
        command.torque.setflags(write=True)
    assert CoMReference([0, 0, 1], .1, .105).covers(.1, .005)
    assert not CoMReference([0, 0, 1], .1, .104).covers(.1, .005)
    for start, end in ((1, 1), (-1, 1), (0, np.inf), (np.nan, 1)):
        with pytest.raises(ValueError, match="interval"):
            TorqueCommand([1.], start, end)
        with pytest.raises(ValueError, match="interval"):
            CoMReference([0, 0, 1], start, end)


@pytest.mark.parametrize("stamp", [1e8, 1e12, 1e13])
def test_validity_roundoff_cannot_cover_a_short_command_on_a_coarse_clock(stamp):
    dt = .01
    short_end = np.nextafter(stamp + dt, stamp)
    for value in (TorqueCommand([0.], stamp, short_end),
                  CoMReference([0, 0, 1], stamp, short_end)):
        assert not value.covers(stamp, dt)
    for value in (TorqueCommand([0.], stamp, stamp + dt),
                  CoMReference([0, 0, 1], stamp, stamp + dt)):
        assert value.covers(stamp, dt)


def test_result_keeps_mapping_compatibility_and_detaches_output_arrays():
    acceleration, wrench = np.zeros(7), np.zeros(6)
    terms = SimpleNamespace(mass=np.eye(7), contact_jacobian=np.zeros((6, 7)))
    result = DynamicsControlResult(QPResult("solved", np.zeros(14), 10, 1e-9),
                                   TorqueCommand([2], 0, .005), acceleration,
                                   wrench, np.zeros(6), terms, 1e-10, 0.)
    acceleration[:] = wrench[:] = 9
    assert result["terms"] is terms and result["torque"] is result.command.torque
    assert result.get("iterations") == 10 and len(result) == 9
    assert dict(result)["violation"] == 1e-9
    assert not np.any(result.acceleration) and not np.any(result.wrench)
    with pytest.raises(KeyError):
        result["missing"]
    with pytest.raises(ValueError):
        result.acceleration[:] = 9

