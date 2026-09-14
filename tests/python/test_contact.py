"""Independent point-force realizability checks for the surface wrench cone."""
import numpy as np
import pytest
from scipy.optimize import linprog

from wholebodyx.contact import rectangular_wrench_cone


def rays(mu=.6, x=.10, y=.035):
    result = []
    for px in (-x, x):
        for py in (-y, y):
            for fx, fy in ((mu, 0), (-mu, 0), (0, mu), (0, -mu)):
                result.append([fx, fy, 1., py, -px, px * fy - py * fx])
    return np.array(result).T


def test_point_force_rays_and_positive_combinations_are_feasible():
    cone = rectangular_wrench_cone()
    basis = rays()
    assert np.max(cone @ basis) < 1e-10
    force = basis @ np.random.default_rng(321).uniform(0, 100, (16, 100))
    assert np.max(cone @ force) < 1e-8
    with pytest.raises(ValueError):
        cone.setflags(write=True)


def test_corner_pressure_cannot_independently_create_yaw():
    # CoP at one corner with zero total shear requires zero yaw moment.
    impossible = np.array([0., 0., 100., 3.5, 10., 1.5])
    assert abs(impossible[3]) <= .035 * impossible[2]
    assert abs(impossible[4]) <= .10 * impossible[2]
    assert abs(impossible[5]) <= .015 * impossible[2]
    assert np.max(rectangular_wrench_cone() @ impossible) > 1.
    result = linprog(np.zeros(16), A_eq=rays(), b_eq=impossible, bounds=(0, None), method="highs")
    assert result.status == 2


def test_facet_classification_matches_nonnegative_corner_forces():
    cone, basis = rectangular_wrench_cone(), rays()
    rng = np.random.default_rng(19)
    for _ in range(40):
        wrench = np.r_[rng.uniform(-.8, .8, 2), 1., rng.uniform(-.05, .05),
                       rng.uniform(-.13, .13), rng.uniform(-.08, .08)]
        by_facets = np.max(cone @ wrench) <= 1e-9
        result = linprog(np.zeros(16), A_eq=basis, b_eq=wrench, bounds=(0, None), method="highs")
        assert by_facets == result.success


@pytest.mark.parametrize("args", [(0., .1, .035), (.6, 0., .035), (.6, .1, -1.), (np.nan, .1, .035)])
def test_invalid_contact_geometry(args):
    with pytest.raises(ValueError):
        rectangular_wrench_cone(*args)
