# Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
# License: MIT
"""Tests for Hamada coordinates on an axisymmetric (Solov'ev) tokamak equilibrium, with oracles independent of the transform."""

import shutil
import subprocess
from pathlib import Path

import pytest

try:
    import numpy as np

    from gvec import State, EvaluationsHamada
except ImportError:
    pass  # tests will be skipped via the `check_import` fixture


# === FIXTURES === #

EXAMPLE = Path(__file__).parent / "../examples/analytic_gs_elliptok"


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    rundir = tmp_path_factory.mktemp("hamada")
    shutil.copy(EXAMPLE / "parameter.ini", rundir)
    subprocess.run(["gvec", "parameter.ini"], cwd=rundir, capture_output=True, check=True)
    statefile = sorted(rundir.glob("*State*"))[-1]
    return State(rundir / "parameter.ini", statefile)


@pytest.fixture(scope="module")
def ev_hamada(state):
    return EvaluationsHamada(
        rho=[0.3, 0.6, 0.9], theta_H=32, zeta_H=4, state=state, radial_derivative=False
    )


# === TESTS === #


def test_hamada_equal_volume_angle(state, ev_hamada):
    """In axisymmetry, the Hamada poloidal angle is the equal-volume angle:

    theta_H(theta) - theta_H(theta_0) = 2 pi int_{theta_0}^{theta} Jac dtheta' / int_0^{2pi} Jac dtheta' ,

    which is evaluated by quadrature of the logical Jacobian, at the logical theta found for the Hamada grid.
    """
    ds = ev_hamada
    # the transform is independent of zeta
    for var in ["theta", "NU_H"]:
        values = ds[var].transpose("rad", "pol", "tor").values
        np.testing.assert_allclose(
            values, np.broadcast_to(values[:, :, :1], values.shape), atol=1e-12
        )
    zeta_H = (ds.zeta + ds.NU_H).transpose("rad", "pol", "tor")
    np.testing.assert_allclose(zeta_H, ds.zeta_H.broadcast_like(zeta_H), atol=1e-12)

    nt = 4096
    ref = state.evaluate("Jac", rho=ds.rho.values, theta=nt, zeta=1)
    for r in range(ds.rho.size):
        Jac = ref.Jac.isel(rad=r, tor=0).values
        # cumulative trapezoidal rule on the periodic, equidistant grid
        theta_ref = np.linspace(0, 2 * np.pi, nt + 1)
        cumJ = np.concatenate([[0.0], np.cumsum(0.5 * (Jac + np.roll(Jac, -1)))])
        theta_ev = 2 * np.pi * cumJ / cumJ[-1]  # equal-volume angle on the reference grid

        theta = np.mod(ds.theta.isel(rad=r, tor=0).values, 2 * np.pi)
        theta_ev_H = np.interp(theta, theta_ref, theta_ev)
        theta_H = ds.theta_H.values
        dtheta = (
            np.mod(theta_ev_H - theta_ev_H[0] - (theta_H - theta_H[0]) + np.pi, 2 * np.pi)
            - np.pi
        )
        np.testing.assert_allclose(dtheta, 0.0, atol=1e-6)


def test_plot_hamada(state):
    """The flux surface plots accept Hamada coordinates and use the Hamada angles as axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = state.plot_on_flux_surface("Jac_H", rho=0.6, ntheta=9, nzeta=5, sfl="hamada")
    ax = np.asarray(axs).flat[0]
    assert ax.get_xlabel() == r"$\zeta_H / (2\pi)$"
    assert ax.get_ylabel() == r"$\theta_H / (2\pi)$"
    plt.close(fig)

    fig, axs = state.plot_fourier_on_surface(
        "mod_B", rho=0.6, ntheta=9, nzeta=5, sfl="hamada", contour=True
    )
    assert "Hamada coordinates" in fig._suptitle.get_text()
    plt.close(fig)
