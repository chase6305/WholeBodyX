"""Linearized rectangular surface-contact wrench cones, local [force; moment]."""
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .qp import _immutable


@lru_cache(maxsize=16)
def rectangular_wrench_cone(friction=.6, half_length=.10, half_width=.035):
    """Return H for H @ wrench <= 0, including nonnegative normal force.

Four corners each contribute the four rays of an inscribed friction diamond
(|fx| + |fy| <= mu*fz). Intersect their summed wrench cone with fz=1, compute
its 5-D convex hull, then homogenize the facets. This couples shear, CoP and
yaw moment; independent bounds on these quantities alone are insufficient.
"""
    values = np.asarray([friction, half_length, half_width], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("friction and contact half-dimensions must be positive")
    vertices = []
    indices = [0, 1, 3, 4, 5]
    for x in (-half_length, half_length):
        for y in (-half_width, half_width):
            for fx, fy in ((friction, 0.), (-friction, 0.), (0., friction), (0., -friction)):
                force = np.array([fx, fy, 1.])
                wrench = np.r_[force, np.cross([x, y, 0.], force)]
                vertices.append(wrench[indices])
    try:
        hull = ConvexHull(vertices)
    except QhullError as exc:
        raise ValueError("contact geometry is numerically degenerate") from exc
    # Qhull triangulates coplanar facets. Keep one unrounded representative
    # of each facet; rounding only supplies the duplicate-detection key.
    rows = {}
    for equation in hull.equations:
        row = np.zeros(6)
        row[indices], row[2] = equation[:5], equation[5]
        row[np.abs(row) < 1e-14] = 0.
        rows.setdefault(tuple(np.round(row, 11)), row)
    cone = np.vstack(([0., 0., -1., 0., 0., 0.], *rows.values()))
    return _immutable(cone)
