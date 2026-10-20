"""Physical regressions using tracked coils and an in-memory analytic coil set.

State fixtures deliberately have function scope: pygvec has one global Fortran state.
All files produced by State.new, runs and restarts stay under tmp_path.
"""

import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

try:
    import numpy as np
    from scipy.constants import mu_0

    import gvec
    from gvec.coils import Coil, CoilSet

    from free_boundary_oracles import (
        LogRecorder,
        boundary_coefficients,
        boundary_fields,
        boundary_symmetry_defect,
        newton_converged,
        normal_field_ratio,
        plasma_volume,
        poloidal_circulation,
        pressure_balance_residual,
        stellarator_symmetry_defect,
    )
except ImportError:
    pass


DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(autouse=True)
def work_in_tmp_path(tmp_path, monkeypatch):
    pytest.importorskip("fbtools")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _parameters():
    return dict(
        nfp=1,
        which_hmap=1,
        PhiEdge=-0.1,
        I_tor=dict(type="polynomial", coefs=[0.0, 0.0, 5e4]),
        iota=dict(type="polynomial", coefs=[0.3]),
        pres=dict(type="polynomial", coefs=[0.0]),
        X1_b_cos={(0, 0): 3.0, (1, 0): 0.5},
        X2_b_sin={(1, 0): 0.5},
        X1_a_cos={(0, 0): 3.0},
        X1_mn_max=[2, 0],
        X2_mn_max=[2, 0],
        LA_mn_max=[2, 0],
        sgrid_nElems=2,
        X1X2_deg=3,
        LA_deg=3,
        totalIter=20000,
        minimize_tol=1e-3,
        picard_current=dict(target="iota_and_force", iota_tol=1e-5, max_updates=8),
        free_boundary=dict(
            method="newton",
            updates=12,
            tolerance=1e-2,
            newton_gtol=0.0,
            newton_rtol=1e-6,
            newton_xtol=1e-5,
            newton_fd_step=1e-3,
            newton_max_step=0.05,
            ramp_force_tol=False,
            digits_BIEST=8,
        ),
    )


def _gframe_coils(frame):
    """Planar analytic loops, sampled densely around a frame read from tracked data."""
    dense = gvec.gframe.to_axis(frame, nzeta=16)["axis"]
    t = np.linspace(0, 2 * np.pi, 65)
    x, normal, binormal = (dense[k] for k in ("xyz", "Nxyz", "Bxyz"))
    return CoilSet(
        [
            Coil(
                x[:, i, None]
                + 0.3 * (normal[:, i, None] * np.cos(t) + binormal[:, i, None] * np.sin(t)),
                20000.0,
            )
            for i in range(x.shape[1])
        ]
    )


def _run(parameters, coils, path, restartstate=None):
    with LogRecorder() as log:
        run = gvec.run(
            deepcopy(parameters),
            coil_set=coils,
            runpath=path,
            restartstate=restartstate,
            loglevel="INFO",
            quiet=True,
        )
    return run, log


@pytest.fixture
def stellarator(tmp_path):
    p = _parameters()
    p.update(
        nfp=2,
        PhiEdge=-0.3,
        I_tor=dict(type="polynomial", coefs=[0.0]),
        iota=dict(type="polynomial", coefs=[0.0]),
        X1_b_cos={(0, 0): 4.0, (1, 0): 0.44, (1, 1): 0.16, (0, 1): -0.012, (2, 1): -0.02},
        X2_b_sin={(1, 0): 0.44, (1, 1): -0.16, (0, 1): 0.015, (2, 1): -0.02},
        X1_a_cos={(0, 0): 4.0},
        X1_mn_max=[2, 2],
        X2_mn_max=[2, 2],
        LA_mn_max=[2, 2],
    )
    coils = CoilSet.load(DATA / "ellipstell-coils.nc")
    run, log = _run(p, coils, tmp_path / "stellarator")
    return run, log, coils


@pytest.fixture
def gframe(tmp_path):
    p = _parameters()
    p.update(
        nfp=3,
        which_hmap=21,
        hmap_ncfile=str(DATA / "N3.471-v3.1-lo.nc"),
        PhiEdge=0.001,
        I_tor=dict(type="polynomial", coefs=[0.0, 0.0, 1000.0]),
        X1_b_cos={(1, 0): 0.08},
        X2_b_sin={(1, 0): 0.08},
        X1_a_cos={(0, 0): 0.0},
        X1_mn_max=[2, 2],
        X2_mn_max=[2, 2],
        LA_mn_max=[2, 2],
    )
    # Verified: N3.471 (51698 bytes) is the smallest suitable smooth-axis input for
    # this fast coil model. The smaller N2.12 low-resolution probe timed out at 280 s
    # above its target (0.038 vs 0.03); the hi file has the same distorted axis.
    # A runtime resampling keeps the case small; no generated NetCDF is committed.
    frame = gvec.gframe.to_axis(
        gvec.gframe.read_Gframe_ncfile(DATA / "N3.471-v3.1-lo.nc"), nzeta=5
    )
    # Verified parity of a stellarator-symmetric frame: x,N -> S(x,N), B -> -S B
    # under zeta -> -zeta, with S=diag(1,-1,-1). Remove the source's ~1e-6 asymmetry.
    for name, parity in (("xyz", [1, -1, -1]), ("Nxyz", [1, -1, -1]), ("Bxyz", [-1, 1, 1])):
        values = frame["axis"][name]
        reverse = (-np.arange(values.shape[1])) % values.shape[1]
        frame["axis"][name] = 0.5 * (values + np.array(parity)[:, None] * values[:, reverse])
    gvec.gframe.write_Gframe_ncfile(tmp_path / "frame.nc", frame)
    p["hmap_ncfile"] = str(tmp_path / "frame.nc")
    coils = _gframe_coils(frame)
    initial = gvec.State.new(p, rundir=tmp_path)
    p["PhiEdge"] = gvec.coils.get_phi_edge_from_coils(initial, coils)
    p["free_boundary"].update(tolerance=0.03, newton_max_step=0.01, newton_fd_step=1e-4)
    run, log = _run(p, coils, tmp_path / "gframe")
    return run, log, coils


def _check_physics(run, log, coils, pressure_tolerances=(0.03, 0.08)):
    f = boundary_fields(run.state, coils, ntheta=48, nzeta=32)
    # Conservative BIEST bound: 100 times its requested 10**(-8) relative accuracy.
    assert normal_field_ratio(f) < 100 * 10.0**-8
    rms, maximum = pressure_balance_residual(f)
    # Pressure tolerances account for the deliberately small equilibrium Fourier basis.
    assert rms < pressure_tolerances[0], (rms, maximum)
    assert maximum < pressure_tolerances[1], (rms, maximum)
    np.testing.assert_allclose(
        poloidal_circulation(f, f.B_ext), mu_0 * f.I_tor, rtol=1e-3, atol=1e-7
    )
    np.testing.assert_allclose(poloidal_circulation(f, f.B_coils), 0.0, atol=1e-7)
    assert stellarator_symmetry_defect(f, f.B_ext) < 1e-6
    assert boundary_symmetry_defect(run.state) < 1e-9
    assert newton_converged(log), log.messages[-20:]
    return f


def test_free_boundary_stellarator_real_coils(stellarator, record_property):
    run, log, coils = stellarator
    fields = _check_physics(run, log, coils)
    assert abs(fields.I_tor) < 1e-2
    volume = plasma_volume(run.state)
    # R00 depends on the boundary parameterization. Reference for the complete
    # X1/X2 trace basis and translation-preserving slide gauge; retain the 0.5 mm
    # regression band. The independent field/pressure/symmetry checks above are
    # the physical acceptance criteria.
    record_property("volume", volume)
    R00 = run.state.parameters["X1_b_cos"][(0, 0)]
    assert R00 == pytest.approx(3.9885406108215005, abs=5e-4)
    record_property("R00", R00)
    record_property("newton_updates", len(run.newton_history))
    record_property("pressure_rms", pressure_balance_residual(fields)[0])


def test_free_boundary_gframe_nonzero_current(gframe, record_property):
    run, log, coils = gframe
    assert not run.state.clockwise_zeta
    f = _check_physics(run, log, coils, pressure_tolerances=(0.08, 0.2))
    assert f.I_tor == pytest.approx(1000.0, rel=1e-3)
    record_property("newton_updates", len(run.newton_history))


@pytest.fixture
def current_fit_state(tmp_path):
    from gvec.core.minimize import MinimizationState

    p = _parameters()
    p["I_tor"]["coefs"] = [0.0, 5e4]
    state = MinimizationState.new(p, rundir=tmp_path)
    state.knots_per_sgrid_element = 2
    return state


def test_free_boundary_current_fit_axis_limit(current_fit_state):
    # Verified analytic oracle for concentric circular sections and I_tor = I_edge*rho**2:
    # iota_axis = mu_0*I_edge*R_axis/(2*PhiEdge), from poloidal Ampere and toroidal flux.
    spline, *_ = current_fit_state._get_new_iota(iota_tol=1e-3)
    expected = mu_0 * 5e4 * 3.0 / (2 * -0.1)
    assert spline(0.0) == pytest.approx(expected, rel=1e-3)
