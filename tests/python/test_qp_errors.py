"""Backend failures reset a workspace; invalid caller data must preserve it."""

import numpy as np
import osqp
import pytest

from wholebodyx import OSQPSolver, QuadraticProgram


ERROR_TYPES = [ValueError]
if hasattr(osqp, "OSQPException"):
    ERROR_TYPES.append(osqp.OSQPException)


@pytest.mark.parametrize('operation', ['setup', 'update', 'warm_start', 'solve'])
@pytest.mark.parametrize('error_type', ERROR_TYPES)
def test_backend_failure_returns_error_and_recovers_with_fresh_workspace(
    monkeypatch, operation, error_type
):
    problem = QuadraticProgram([[1.]], [-.2], [[1.]], [-1.], [1.])
    solver = OSQPSolver()
    if operation != 'setup':
        assert solver.solve(problem).success

    def fail(*args, **kwargs):
        raise error_type('injected ' + operation + ' failure')

    with monkeypatch.context() as patch:
        target = osqp.OSQP if operation == 'setup' else solver._solver
        patch.setattr(target, operation, fail)
        result = solver.solve(problem, warm_start=[0.] if operation == 'warm_start' else None)
    assert result.status == 'error' and result.x is None
    assert 'injected ' + operation + ' failure' in result.detail
    assert solver._solver is None
    recovered = solver.solve(problem)
    assert recovered.success
    np.testing.assert_allclose(recovered.x, [.2], atol=1e-7)


def test_missing_osqp_exception_export_does_not_mask_validation_errors(monkeypatch):
    # This attribute is absent in supported OSQP 1.0 installations.
    monkeypatch.delattr(osqp, 'OSQPException', raising=False)
    problem = QuadraticProgram(np.eye(2), [-.2, -.3], np.eye(2), [-1.] * 2, [1.] * 2)
    solver = OSQPSolver()
    assert solver.solve(problem).success
    workspace = solver._solver
    problem.hessian.setflags(write=True)
    problem.hessian[0, 0] = -1.
    with pytest.raises(ValueError, match='positive semidefinite'):
        solver.solve(problem)
    assert solver._solver is workspace
    problem.hessian[0, 0] = 1.
    assert solver.solve(problem).success
