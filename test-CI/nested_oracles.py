"""Independent integer-family oracle: Issan et al. 2610.07304, (4)--(7).

Test mathematics only; production GVEC never imports this module.
tau=0 recovers Landreman 2609.26742, (2)--(5). Units: mu0=1.
"""

import numpy as np

EPS = 0.5
DELTA = 1 / 64


def field(points, tau):
    """Return psi, normalized pressure and Cartesian B, including complex points."""
    x, y, z = np.asarray(points).T
    a, b = np.sqrt(1 + EPS), np.sqrt(1 - EPS)
    s = (x / a) ** 2 + (y / b) ** 2
    f = np.sqrt(s * (2 - s) - 4 * z**2 - 4 * tau * s * z)
    magnetic = np.column_stack(
        (
            ((2 * z + tau * s) * x - a / b * f * y) / s,
            ((2 * z + tau * s) * y + b / a * f * x) / s,
            1 - s - 2 * tau * z,
        )
    )
    omega2 = 1 - tau**2
    q = omega2 * (x**2 + y**2 + 4 * z**2) / 2 + 2 * tau * z
    psi = (
        omega2
        / 2
        * (
            q
            + np.sum(magnetic**2, axis=1) / 2
            - 1
            + tau**2 / (2 * omega2)
            + omega2 * EPS**2 / 2
        )
    )
    pressure = 2 * (DELTA - psi) / omega2
    return psi, pressure, magnetic


def surface_distance(points, labels, tau):
    """Distance to a point on the exact surface in the same geometric-phi plane.

    Intersect a ray from the analytic magnetic axis through each candidate.
    This is a sampled upper bound on closest-surface distance, not a norm bound.
    """
    from scipy.optimize import brentq

    phi = np.arctan2(points[:, 1], points[:, 0])
    omega2 = 1 - tau**2
    ax = np.sqrt((1 + EPS) * (1 - omega2 * EPS) / omega2)
    ay = np.sqrt((1 - EPS) * (1 + omega2 * EPS) / omega2)
    t = np.arctan2(ax * np.sin(phi), ay * np.cos(phi))
    radius = ax * ay / np.sqrt((ay * np.cos(phi)) ** 2 + (ax * np.sin(phi)) ** 2)
    axis = np.column_stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            -tau / (2 * omega2) + EPS / 2 * np.sin(2 * t + np.arcsin(tau)),
        )
    )
    distance = []
    for point, center, label in zip(points, axis, labels):
        ray = point - center

        def residual(scale):
            return field((center + scale * ray)[None, :], tau)[0][0] - DELTA * label

        # The candidate is close to its surface. A 20% radial bracket stays
        # inside the analytic field domain for these interior test surfaces.
        scale = brentq(residual, 0, 1.2, xtol=1e-12)
        distance.append(abs(scale - 1) * np.linalg.norm(ray))
    return np.asarray(distance)
