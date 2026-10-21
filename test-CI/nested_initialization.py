"""Analytic chart projection for exact-interior fixed-point tests only.

The chart uses the oracle definition in nested_oracles and geometric phi=-zeta.
No numerical equilibrium state or fitted field is used as input.
"""

import numpy as np
from scipy.interpolate import BSpline

from nested_oracles import DELTA, EPS, field


def chart(rho, theta, zeta, tau):
    # GVEC theta=-VMEC theta; phi=-zeta; material angle=2 phi-VMEC theta.
    alpha = theta - 2 * zeta
    om2 = 1 - tau * tau
    u = -om2 * EPS / 2 + np.sqrt(DELTA) * rho * np.cos(alpha)
    v = np.sqrt(DELTA) * rho * np.sin(alpha)
    ell = np.sqrt((1 + np.sqrt(1 - 4 * (u * u + v * v))) / 2)

    def xyz(t):
        x = (
            np.sqrt(1 + EPS)
            / np.sqrt(om2)
            * (ell * np.cos(t) + (u * np.cos(t) + v * np.sin(t)) / ell)
        )
        y = (
            np.sqrt(1 - EPS)
            / np.sqrt(om2)
            * (ell * np.sin(t) + (v * np.cos(t) - u * np.sin(t)) / ell)
        )
        z = (
            -tau / (2 * om2)
            + (
                v * (np.sqrt(om2) * np.cos(2 * t) - tau * np.sin(2 * t))
                - u * (np.sqrt(om2) * np.sin(2 * t) + tau * np.cos(2 * t))
            )
            / om2
        )
        return x, y, z

    phi = -zeta
    lo, hi = phi - np.pi / 2, phi + np.pi / 2
    for _ in range(52):
        t = (lo + hi) / 2
        x, y, z = xyz(t)
        ray = y * np.cos(phi) - x * np.sin(phi)
        lo, hi = np.where(ray < 0, t, lo), np.where(ray < 0, hi, t)
    return np.stack(xyz((lo + hi) / 2), axis=-1)


def axis(p, tau):
    zeta = 2 * np.pi * np.arange(256) / 256
    x = chart(0.0, 0.0, zeta, tau)
    for component, values in [("X1", np.linalg.norm(x[:, :2], axis=1)), ("X2", x[:, 2])]:
        for trig, fun in [("cos", np.cos), ("sin", np.sin)]:
            p[f"{component}_a_{trig}"] = {
                (0, n): float(np.mean(values * fun(-2 * n * zeta)) * (1 if n == 0 else 2))
                for n in range(13)
                if trig == "cos" or n > 0
            }
    p["init_average_axis"] = False


def exact_state(state, tau):
    state.save()
    path = state.statefile
    state.unbind()
    lines = path.read_text().splitlines()
    grid = np.fromstring(lines[5], sep=",")
    # Native clamped quintic splines, simple internal knots (C4).
    degree = int(lines[9].split(",")[1])
    knots = np.r_[np.repeat(0.0, degree + 1), grid[1:-1], np.repeat(1.0, degree + 1)]
    nbase = len(knots) - degree - 1
    r = np.array([np.mean(knots[i + 1 : i + degree + 1]) for i in range(nbase)])
    matrix = BSpline.design_matrix(r, knots, degree).toarray()
    a = 2 * np.pi * np.arange(64) / 64
    rr, tt, zz = np.meshgrid(r, a, a, indexing="ij")
    x = chart(rr, tt, zz, tau)
    np.testing.assert_allclose(
        field(x.reshape(-1, 3), tau)[0].reshape(rr.shape), DELTA * rr**2, atol=1e-14
    )
    for quantity, values in [
        ("X1", np.linalg.norm(x[..., :2], axis=-1)),
        ("X2", x[..., 2]),
        ("LA", np.zeros_like(rr)),
    ]:
        start = (
            next(i for i, line in enumerate(lines) if line.startswith(f"## {quantity}:")) + 1
        )
        header = {"X1": 9, "X2": 11, "LA": 13}[quantity]
        count = int(lines[header].split(",")[3])
        sincos = int(lines[header].split(",")[4])
        # sin_cos=1 sine, =2 cosine, =3 concatenated sine then cosine.
        for j in range(count):
            entry = np.fromstring(lines[start + j], sep=",")
            m, n = entry[:2].astype(int)
            cosine = sincos == 2 or (sincos == 3 and j >= count // 2)
            if sincos == 3:
                # The cosine section includes (0,0), sine excludes it.
                cosine = j >= (count - 1) // 2 if quantity != "LA" else j >= count // 2
            basis = (np.cos if cosine else np.sin)(m * tt - n * zz)
            coef = np.mean(values * basis, axis=(1, 2)) * (1 if m == 0 and n == 0 else 2)
            dofs = np.linalg.solve(matrix, coef)
            lines[start + j] = f"{m},{n}," + ",".join(f"{v:.16e}" for v in dofs)
    exact = path.with_name("exact.dat")
    exact.write_text("\n".join(lines) + "\n")
    return exact
