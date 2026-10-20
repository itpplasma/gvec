"""Newton return to a matched, current-controlled axisymmetric equilibrium."""

import tempfile
from copy import deepcopy

import pytest

try:
    import numpy as np
    from scipy.optimize import root
    from scipy.linalg import orth

    import gvec
    from gvec.coils import Coil, CoilSet
    from gvec.core.minimize import MinimizationState
    from gvec.core.run.free_boundary import (
        _boundary_mode_list,
        _get_B2_fp,
        _normal_displacement_basis,
    )

    from free_boundary_oracles import (
        LogRecorder,
        axisymmetric_contour,
        boundary_coefficients,
        boundary_fields,
        contour_distance,
        newton_converged,
        pressure_balance_residual,
    )
except ImportError:
    pass


def _axis_wire(current, length=1e4):
    coils = []
    for j in range(4):
        c, s = np.cos(j * np.pi / 2), np.sin(j * np.pi / 2)
        points = [
            [0, 0, -length],
            [0, 0, length],
            [length * c, length * s, length],
            [length * c, length * s, -length],
            [0, 0, -length],
        ]
        coils.append(Coil(np.array(points).T, current / 4))
    return coils


def _vertical_pair(z, scale=1.0):
    t = np.linspace(0, 2 * np.pi, 257)
    return [
        Coil(np.array([6 * np.cos(t), 6 * np.sin(t), 0 * t + q]), 25000 * scale)
        for q in (-z, z)
    ]


@pytest.fixture
def matched_axisymmetric_state(tmp_path, monkeypatch):
    pytest.importorskip("fbtools")
    # Function scope is mandatory: every State binds the same global Fortran library.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    p = dict(
        ProjectName="offset",
        nfp=1,
        which_hmap=1,
        FreeBoundary=True,
        PhiEdge=-0.1,
        I_tor=dict(type="polynomial", coefs=[0.0, 0.0, 50000.0]),
        iota=dict(type="polynomial", coefs=[0.3]),
        pres=dict(type="polynomial", coefs=[0.0]),
        X1_b_cos={(0, 0): 3.0, (1, 0): 0.5},
        X2_b_sin={(1, 0): 0.5},
        X2_b_cos={(0, 0): 0.0},
        X2_sin_cos="_sincos_",
        X1_a_cos={(0, 0): 3.0},
        X1_mn_max=[2, 0],
        X2_mn_max=[2, 0],
        LA_mn_max=[2, 0],
        sgrid_nElems=2,
        X1X2_deg=3,
        LA_deg=3,
        totalIter=100000,
        minimize_tol=1e-8,
        picard_current=dict(target="iota_and_force", iota_tol=1e-7, max_updates=15),
        free_boundary=dict(
            method="newton",
            updates=12,
            tolerance=1e-5,
            ramp_force_tol=False,
            newton_gtol=1e-7,
            newton_rtol=1e-6,
            newton_xtol=1e-7,
            newton_fd_step=1e-4,
            newton_max_step=0.05,
            newton_max_backtracks=3,
        ),
    )
    state = MinimizationState.new(p, rundir=tmp_path)
    state.solve(maxIter=100000, minimize_tol=1e-8)

    # BIEST is linear in coil currents and the prescribed plasma circulation. Fit four
    # analytic circuit amplitudes to all four symmetric normal force modes of this boundary.
    # This constructs a base equilibrium without using a stored statefile or free-boundary
    # descent as the oracle for the subsequent displacement/return experiment.
    full_projection, _ = _normal_displacement_basis(state)
    symmetric = [
        i
        for i, (var, sc, _, _) in enumerate(_boundary_mode_list(state))
        if var == "X1" or sc == "s"
    ]
    # The odd equations vanish by symmetry. The complete basis adds a fourth
    # even shape equation; matching only the old three leaves a non-equilibrium
    # base and confounds a return test with the correction of that base.
    projection = full_projection @ orth(full_projection[symmetric, :].T, rcond=1e-10)
    assert projection.shape[1] == 4
    nt, nz = state.t_n, state.z_n
    _, plasma, _, _ = _get_B2_fp(state, CoilSet(_axis_wire(0)), nt, nz, 50000.0, True, 8)
    circuits = [
        CoilSet(_axis_wire(2e6)),
        CoilSet(_vertical_pair(3)),
        CoilSet(_vertical_pair(1)),
        CoilSet(_vertical_pair(2)),
    ]
    response = []
    for circuit in circuits:
        _, induced, coils, _ = _get_B2_fp(state, circuit, nt, nz, 0.0, True, 8)
        response.append(induced + coils)

    def residual(amplitudes):
        field = plasma + np.einsum("k,kijt->ijt", amplitudes, response)
        state.set_Bext_squared((field**2).sum(axis=0).ravel())
        return projection.T @ state._eval_boundary_forces()

    solution = root(residual, [1.0, 1.0, 0.0, 0.0], tol=1e-10)
    if not solution.success or np.max(np.abs(solution.fun)) > 1e-9:
        pytest.fail(f"Could not construct a matched base: {solution.message}, {solution.fun}")
    # Verify all equations, including the odd ones, at the matched base.
    np.testing.assert_allclose(full_projection.T @ state._eval_boundary_forces(), 0, atol=1e-9)
    a = solution.x
    coils = CoilSet(
        _axis_wire(2e6 * a[0])
        + _vertical_pair(3, a[1])
        + _vertical_pair(1, a[2])
        + _vertical_pair(2, a[3])
    )
    fields = boundary_fields(state, coils, ntheta=48, nzeta=16)
    if pressure_balance_residual(fields)[0] > 0.01:
        pytest.fail("Matched base exceeds 1% RMS pressure imbalance on the independent grid")
    state.save()
    return state, coils, p


def _return_from_shift(base, direction, tmp_path, record_property):
    state, coils, p = base
    contour = axisymmetric_contour(state)
    mode = ("X1", "c", 0, 0) if direction == "R" else ("X2", "c", 0, 0)
    displacement = np.zeros_like(state._get_boundary_dofs())
    displacement[_boundary_mode_list(state).index(mode)] = 0.01
    state._checkpoint_dofs()
    if not state._apply_boundary_perturbation(displacement, 0.6):
        pytest.fail("Rigid offset made the base geometry invalid")
    translation = np.array([0.01, 0.0]) if direction == "R" else np.array([0.0, 0.01])
    np.testing.assert_allclose(
        axisymmetric_contour(state), contour + translation, rtol=0, atol=1e-12
    )
    state.save()
    parameters = deepcopy(p)
    parameters.update(boundary_coefficients(state))
    parameters["iota"] = deepcopy(state.parameters["iota"])
    with LogRecorder() as log:
        run = gvec.run(
            parameters,
            restartstate=state,
            coil_set=coils,
            runpath=tmp_path / "return",
            loglevel="INFO",
            quiet=True,
        )
    distance = contour_distance(contour, axisymmetric_contour(run.state))
    record_property("return_distance_m", distance)
    record_property("newton_updates", len(run.newton_history))
    record_property("interior_iterations", run.GVEC_iter_used)
    # Verified geometric oracle: 0.5 mm is twenty times smaller than the imposed offset;
    # the polygon discretization error is < 0.02 mm and parameterization changes are ignored.
    assert distance < 5e-4, (direction, distance, run.newton_history, log.messages[-12:])
    assert newton_converged(log), log.messages[-12:]


def test_free_boundary_newton_radial_return(
    matched_axisymmetric_state, tmp_path, record_property
):
    _return_from_shift(matched_axisymmetric_state, "R", tmp_path, record_property)


def test_free_boundary_newton_vertical_return(
    matched_axisymmetric_state, tmp_path, record_property
):
    _return_from_shift(matched_axisymmetric_state, "Z", tmp_path, record_property)
