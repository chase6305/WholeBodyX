"""Absolute symmetry tolerance and mutated storage retain their public contract."""

import numpy as np
import pytest

from wholebodyx import OSQPSolver, QuadraticProgram
from wholebodyx.qp import _FixedQP


@pytest.mark.parametrize('delta', [0., np.nextafter(1e-10, 0.), 1e-10,
                                   np.nextafter(1e-10, np.inf), -1e-10, -1.01e-10])
@pytest.mark.parametrize('template', [False, True])
def test_symmetry_tolerance_boundary_preserves_absolute_decision(delta, template):
    h = np.array([[2., delta], [0., 2.]])
    values = dict(hessian=h, gradient=[0., 0.], constraint=np.eye(2),
                  lower=[-1., -1.], upper=[1., 1.])
    build = _FixedQP.with_problem if template else QuadraticProgram
    if abs(delta) <= 1e-10:
        build(**values)
    else:
        with pytest.raises(ValueError, match='symmetric'):
            build(**values)


@pytest.mark.parametrize('entry', [1e10, 1e150, np.finfo(float).max])
def test_large_entries_do_not_enable_a_relative_symmetry_tolerance(entry):
    h = np.array([[entry, entry], [np.nextafter(entry, 0.), entry]])
    with pytest.raises(ValueError, match='symmetric'):
        QuadraticProgram(h, [0., 0.], np.eye(2), [-1., -1.], [1., 1.])


def test_symmetric_singular_psd_and_negative_curvature_still_checked():
    h = np.ones((2, 2))
    p = QuadraticProgram(h, [0., 0.], np.eye(2), [-1., -1.], [1., 1.])
    assert OSQPSolver().solve(p).success
    with pytest.raises(ValueError, match='positive semidefinite'):
        QuadraticProgram([[1., 2.], [2., 1.]], [0., 0.], np.eye(2), [-1., -1.], [1., 1.])


def test_integer_metadata_cannot_hide_asymmetry_through_overflow():
    p = QuadraticProgram(np.eye(2), [0., 0.], np.eye(2), [-1., -1.], [1., 1.])
    solver = OSQPSolver()
    assert solver.solve(p).success
    workspace = solver._solver
    p.hessian.setflags(write=True)
    p.hessian.dtype = np.int64
    p.hessian[:] = [[2**62, 0], [np.iinfo(np.int64).min, 2**62]]
    with pytest.raises(ValueError, match='symmetric'):
        solver.solve(p)
    assert solver._solver is workspace
    p.hessian.dtype = np.float64
    p.hessian[:] = np.eye(2)
    assert solver.solve(p).success


def test_changed_finite_asymmetry_rejected_before_workspace_update():
    p = QuadraticProgram(np.eye(2), [0., 0.], np.eye(2), [-1., -1.], [1., 1.])
    solver = OSQPSolver()
    assert solver.solve(p).success
    workspace = solver._solver
    p.hessian.setflags(write=True)
    p.hessian[0, 1] = np.nextafter(1e-10, np.inf)
    with pytest.raises(ValueError, match='symmetric'):
        solver.solve(p)
    assert solver._solver is workspace
    p.hessian[0, 1] = 0
    assert solver.solve(p).success
