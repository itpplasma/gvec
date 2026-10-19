"""Independent checks of a converged free-boundary state, shared by the real-case free-boundary tests.

All fields are evaluated on the cell-centred grid of one field period that GVEC and BIEST use for the boundary
force, theta_i = (i + 1/2) 2 pi / nt, zeta_j = (j + 1/2) 2 pi / (nfp nz).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.constants import mu_0


@dataclass
class BoundaryFields:
    theta: np.ndarray  # (nt,)
    zeta: np.ndarray  # (nz,)
    B_ext: np.ndarray  # exterior field (coils + BIEST plasma part), (3, nz, nt)
    B_coils: np.ndarray  # coil field, (3, nz, nt)
    B_int: np.ndarray  # interior (GVEC) field at rho = 1, (3, nz, nt)
    normal: np.ndarray  # unit normal e_theta x e_zeta / |...|, (3, nz, nt)
    e_theta: np.ndarray  # (3, nz, nt)
    p_edge: float  # pressure at rho = 1 [Pa]
    I_tor: float  # GVEC's I_tor at rho = 1 [A]


def boundary_fields(state, coil_set, digits=8, refine=1, ntheta=None, nzeta=None):
    """Exterior field (coils + BIEST) and interior field on the boundary of `state`.

    `refine` > 1 (odd) evaluates on a grid refined by that factor in both directions, so that the checks are not
    restricted to the points the free-boundary iteration used."""
    from gvec.core.run.free_boundary import _get_B2_fp

    _, _, nt_state, _, nz_state, _ = state.get_integration_points()
    nt = ntheta or nt_state * refine
    nz = nzeta or nz_state * refine
    _, B_plas, B_coils, _ = _get_B2_fp(state, coil_set, nt, nz, _I_tor(state), True, digits)
    theta = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    zeta = (np.arange(nz) + 0.5) * 2 * np.pi / state.nfp / nz
    ev = state.evaluate("e_theta", "e_zeta", "B", "p", rho=1.0, theta=theta, zeta=zeta).isel(
        rad=0
    )

    def vec(name):
        return ev[name].transpose("xyz", "tor", "pol").values

    e_theta, e_zeta = vec("e_theta"), vec("e_zeta")
    normal = np.cross(e_theta, e_zeta, axis=0)
    normal /= np.linalg.norm(normal, axis=0)
    return BoundaryFields(
        theta=theta,
        zeta=zeta,
        B_ext=B_plas + B_coils,
        B_coils=B_coils,
        B_int=vec("B"),
        normal=normal,
        e_theta=e_theta,
        p_edge=float(np.asarray(ev.p.values).ravel()[0]),
        I_tor=_I_tor(state),
    )


def _I_tor(state):
    return float(np.asarray(state.evaluate("I_tor", rho=1.0).I_tor.values).ravel()[0])


def normal_field_ratio(f: BoundaryFields):
    """max |B_ext . n| / |B_ext| on the boundary (the exterior vacuum field is tangential)."""
    return (np.abs((f.B_ext * f.normal).sum(axis=0)) / np.linalg.norm(f.B_ext, axis=0)).max()


def pressure_balance_residual(f: BoundaryFields):
    """rms and max of [[p + B^2 / (2 mu_0)]] = (B_int^2 + 2 mu_0 p - B_ext^2) / (2 mu_0), relative to the mean
    exterior magnetic pressure B_ext^2 / (2 mu_0)."""
    B2_ext = (f.B_ext**2).sum(axis=0)
    jump = (f.B_int**2).sum(axis=0) + 2 * mu_0 * f.p_edge - B2_ext
    scale = B2_ext.mean()
    return np.sqrt((jump**2).mean()) / scale, np.abs(jump).max() / scale


def poloidal_circulation(f: BoundaryFields, B: np.ndarray):
    """Circulation of B along theta (with GVEC's orientation) at each zeta, (nz,)."""
    return (B * f.e_theta).sum(axis=(0, 2)) * 2 * np.pi / f.theta.size


def stellarator_symmetry_defect(f: BoundaryFields, B: np.ndarray):
    """Stellarator symmetry maps (theta, zeta) -> (-theta, -zeta); on the cell-centred grid that is index
    (i, j) -> (nt-1-i, nz-1-j). |B|^2 is even under it. Returns max | |B|^2 - |B|^2 o S | / max |B|^2."""
    B2 = (B**2).sum(axis=0)
    return np.abs(B2 - B2[::-1, ::-1]).max() / B2.max()


def boundary_coefficients(state):
    """Boundary Fourier coefficients of the final state, {name: {(m, n): value}}."""
    return {
        k: dict(state.parameters[k])
        for k in ("X1_b_cos", "X1_b_sin", "X2_b_cos", "X2_b_sin")
        if k in state.parameters
    }


def plasma_volume(state, n=64):
    """Volume enclosed by the boundary, V = (1/3) oint x . n dA, by the midpoint rule over all field periods."""
    theta = (np.arange(n) + 0.5) * 2 * np.pi / n
    zeta = (np.arange(n) + 0.5) * 2 * np.pi / state.nfp / n
    ev = state.evaluate("pos", "e_theta", "e_zeta", rho=1.0, theta=theta, zeta=zeta).isel(rad=0)
    x = ev.pos.transpose("xyz", "tor", "pol").values
    dA = np.cross(
        ev.e_theta.transpose("xyz", "tor", "pol").values,
        ev.e_zeta.transpose("xyz", "tor", "pol").values,
        axis=0,
    )
    return abs((x * dA).sum()) / 3 * (2 * np.pi / n) * (2 * np.pi / state.nfp / n) * state.nfp


class LogRecorder(logging.Handler):
    """Collects the messages of the `gvec` logger (which does not propagate to the root logger)."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append((record.levelname, record.getMessage()))

    def __enter__(self):
        logger = logging.getLogger("gvec")
        logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("gvec").removeHandler(self)

    def any(self, text, level=None):
        return any(text in m and (level is None or lv == level) for lv, m in self.messages)


NEWTON_CONVERGED = (
    "Newton free boundary: |g| below newton_rtol.",
    "Newton free boundary: |g| below newton_gtol.",
    "Newton free boundary: boundary step below newton_xtol.",
    "Newton free boundary: free-boundary target below tolerance.",
)
NEWTON_FAILED = ("stalled", "no decrease of |g|")


def newton_converged(log: LogRecorder):
    """The Newton update stopped on one of its convergence criteria and did not report a stall."""
    return any(log.any(m) for m in NEWTON_CONVERGED) and not any(
        log.any(m, "WARNING") for m in NEWTON_FAILED
    )


def axisymmetric_contour(state, n=512):
    """R-Z contour at phi=0; a separate array survives rebinding the global State."""
    theta = np.arange(n) * 2 * np.pi / n
    xyz = state.evaluate("pos", rho=1.0, theta=theta, zeta=0.0).pos
    return xyz.isel(rad=0, tor=0).transpose("pol", "xyz").values[:, (0, 2)].copy()


def contour_distance(a, b):
    """Symmetric distance to polygon segments, insensitive to tangential parameterization.

    For these smooth ~0.5 m contours, 512 vertices give chord errors below 2e-5 m,
    comfortably below the return tolerance. No matching of Fourier gauges is assumed.
    """

    def directed(points, polygon):
        edge = np.roll(polygon, -1, axis=0) - polygon
        delta = points[:, None, :] - polygon[None, :, :]
        t = np.clip((delta * edge).sum(axis=-1) / (edge**2).sum(axis=-1), 0, 1)
        distance2 = ((delta - t[:, :, None] * edge) ** 2).sum(axis=-1)
        return np.sqrt(distance2.min(axis=1)).max()

    return max(directed(a, b), directed(b, a))


def boundary_symmetry_defect(state, ntheta=48, nzeta=32):
    """Max position error [m] under stellarator reflection, accounting for field-period rotation."""
    theta = (np.arange(ntheta) + 0.5) * 2 * np.pi / ntheta
    zeta = (np.arange(nzeta) + 0.5) * 2 * np.pi / state.nfp / nzeta
    xyz = state.evaluate("pos", rho=1.0, theta=theta, zeta=zeta).pos
    xyz = xyz.isel(rad=0).transpose("xyz", "tor", "pol").values
    angle = (-1 if state.clockwise_zeta else 1) * 2 * np.pi / state.nfp
    c, s = np.cos(angle), np.sin(angle)
    rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    reflected = rotate @ (xyz * np.array([1, -1, -1])[:, None, None]).reshape(3, -1)
    return np.linalg.norm(xyz[:, ::-1, ::-1] - reflected.reshape(xyz.shape), axis=0).max()
