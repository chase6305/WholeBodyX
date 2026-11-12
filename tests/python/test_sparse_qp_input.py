"""Sparse QP construction must reject complex storage before float casting."""

import warnings

import numpy as np
import pytest
from scipy import sparse

from wholebodyx import QuadraticProgram, SparseQuadraticProgram


@pytest.mark.parametrize('field', ['hessian', 'constraint'])
@pytest.mark.parametrize('container', ['dense', 'sparse', 'list'])
@pytest.mark.parametrize('imaginary', [0., .5])
def test_complex_matrix_input_is_rejected_without_discarding_imaginary_values(field, container, imaginary):
    value = np.eye(2, dtype=complex)
    value[0, 0] += imaginary * 1j
    if container == 'sparse':
        value = sparse.csc_matrix(value)
    elif container == 'list':
        value = value.tolist()
    values = dict(hessian=np.eye(2), gradient=[0., 0.], constraint=np.eye(2),
                  lower=[-1., -1.], upper=[1., 1.])
    values[field] = value
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='real-valued'):
            SparseQuadraticProgram(**values)


@pytest.mark.parametrize('constructor', [QuadraticProgram, SparseQuadraticProgram])
@pytest.mark.parametrize('field', ['lower', 'upper'])
@pytest.mark.parametrize('container', ['array', 'list'])
@pytest.mark.parametrize('imaginary', [0., .5])
def test_complex_bounds_are_rejected_before_float_conversion(constructor, field, container, imaginary):
    value = np.array([-1., -1.] if field == 'lower' else [1., 1.], dtype=complex)
    value[0] += imaginary * 1j
    if container == 'list':
        value = value.tolist()
    values = dict(hessian=np.eye(2), gradient=[0., 0.], constraint=np.eye(2),
                  lower=[-1., -1.], upper=[1., 1.])
    values[field] = value
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='real-valued'):
            constructor(**values)
