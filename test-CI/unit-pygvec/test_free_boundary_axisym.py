"""Free-boundary helpers on an axisymmetric (nfp=1, n_max=0) state, which has a single toroidal integration point."""

import pytest

try:
    import numpy as np
    from scipy.constants import mu_0
    import gvec
    from gvec.coils import Coil, CoilSet, get_phi_edge_from_coils
except ImportError:
    pass

R0, A0 = 3.0, 0.5


def _params():
    return {
        "nfp": 1,
        "which_hmap": 1,
        "PhiEdge": -1.0,
        "iota": {"type": "polynomial", "coefs": [0.5]},
        "pres": {"type": "polynomial", "coefs": [0.0]},
        "X1_b_cos": {(0, 0): R0, (1, 0): A0},
        "X2_b_sin": {(1, 0): A0},
        "X1_a_cos": {(0, 0): R0},
        "X1_mn_max": [1, 0],
        "X2_mn_max": [1, 0],
        "LA_mn_max": [1, 0],
        "sgrid_nElems": 2,
        "X1X2_deg": 3,
        "LA_deg": 3,
    }


def _axis_wire(current, length=1e4, nret=4):
    """Closed circuits: current up the z axis, back along nret far-away return wires (field ~ mu0 I / (2 pi R))."""
    coils = []
    for j in range(nret):
        c, s = np.cos(2 * np.pi * j / nret), np.sin(2 * np.pi * j / nret)
        pts = [[0, 0, -length], [0, 0, length], [length * c, length * s, length]]
        pts += [[length * c, length * s, -length], [0, 0, -length]]
        coils.append(Coil(np.array(pts).T, current / nret))
    return CoilSet(coils)


@pytest.fixture(scope="module")
def state():
    return gvec.State.new(_params())


def test_axisymmetric_state_has_one_toroidal_point(state):
    assert state.get_integration_points()[4] == 1


def test_winding_number_axisymmetric(state):
    # zeta = -phi for which_hmap=1
    assert state.winding_nr == pytest.approx(-1.0, abs=1e-12)
    assert state.clockwise_zeta


def test_phi_edge_from_coils_axisymmetric(state):
    current = 1e6
    phi = get_phi_edge_from_coils(state, _axis_wire(current))
    expected = mu_0 * current * (R0 - np.sqrt(R0**2 - A0**2))  # int B_phi dA, circular section
    assert abs(phi) == pytest.approx(expected, rel=1e-3)


def test_B2_exterior_axisymmetric(state):
    pytest.importorskip("fbtools")
    from gvec.core.run.free_boundary import _get_B2_fp

    nt, nz = state.get_integration_points()[2], state.get_integration_points()[4]
    # a far-away loop without current: the coil field vanishes, the exterior field is the plasma part only
    loop = Coil(np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]).T * 20.0, 0.0)
    i_tor = 2e6
    _, B_plas, B_coils, B2 = _get_B2_fp(state, CoilSet([loop]), nt, nz, i_tor, True, 8)
    assert B_plas.shape == (3, nz, nt) and B2.shape == (nz, nt)
    assert np.all(B_coils == 0.0)
    theta = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    zeta = (np.arange(nz) + 0.5) * 2 * np.pi / state.nfp / nz
    ev = state.evaluate("e_theta", rho=1.0, theta=theta, zeta=zeta).isel(rad=0)
    e_theta = ev.e_theta.transpose("xyz", "tor", "pol").values
    # Ampere: the poloidal circulation of the exterior plasma field is mu_0 I_tor
    circulation = (B_plas * e_theta).sum(axis=(0, 2)) * 2 * np.pi / nt
    assert abs(circulation[0]) == pytest.approx(mu_0 * i_tor, rel=1e-3)


def test_free_boundary_run_axisymmetric(tmp_path):
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 2000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(updates=3, tolerance=1e-3, dt=1e-3)
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    assert np.isfinite(run.state.evaluate("X1", rho=1.0, theta=0.0, zeta=0.0).X1.values).all()


def _minimization_state(tmp_path, mn_max=1):
    from gvec.core.minimize import MinimizationState

    p = _params()
    p["FreeBoundary"] = True
    for q in ("X1", "X2", "LA"):
        p[f"{q}_mn_max"] = [mn_max, 0]
    return MinimizationState.new(p, rundir=tmp_path)


def test_boundary_perturbation_and_checkpoint(tmp_path):
    """A prescribed boundary perturbation moves the boundary by exactly that amount; restore undoes it."""
    from gvec.core.run.free_boundary import _boundary_mode_list

    state = _minimization_state(tmp_path)
    modes = _boundary_mode_list(state)
    i_R0 = modes.index(("X1", "c", 0, 0))
    x0 = state._get_boundary_dofs()
    R0_edge = state.evaluate("X1", rho=1.0, theta=[0.3, 2.0], zeta=0.0).X1.values.ravel()
    state._checkpoint_dofs()
    pert = np.zeros_like(x0)
    pert[i_R0] = 0.01
    assert state._apply_boundary_perturbation(pert, 0.6)
    np.testing.assert_allclose(state._get_boundary_dofs() - x0, pert, atol=1e-14)
    R1_edge = state.evaluate("X1", rho=1.0, theta=[0.3, 2.0], zeta=0.0).X1.values.ravel()
    np.testing.assert_allclose(R1_edge - R0_edge, 0.01, atol=1e-12)
    state._restore_dofs()
    np.testing.assert_array_equal(state._get_boundary_dofs(), x0)


def test_normal_displacement_basis_circle(tmp_path):
    """For a circle R = R0 + a cos(t), Z = a sin(t) the unit normal is (cos t, sin t): a normal displacement 1
    changes a by 1, a normal displacement cos(t) gives dR = cos^2 t = 1/2 + cos(2t)/2, dZ = sin(2t)/2."""
    from gvec.core.run.free_boundary import _boundary_mode_list, _normal_displacement_basis

    state = _minimization_state(tmp_path, mn_max=2)
    modes = _boundary_mode_list(state)
    P, _ = _normal_displacement_basis(state)
    cols = [md for md in modes if md[0] == "X1"]
    expect = {
        ("c", 0): {("X1", "c", 1, 0): 1.0, ("X2", "s", 1, 0): 1.0},
        ("c", 1): {("X1", "c", 0, 0): 0.5, ("X1", "c", 2, 0): 0.5, ("X2", "s", 2, 0): 0.5},
    }
    for (sc, m), dx in expect.items():
        col = P[:, cols.index(("X1", sc, m, 0))]
        want = np.zeros(len(modes))
        for md, v in dx.items():
            want[modes.index(md)] = v
        np.testing.assert_allclose(col, want, atol=1e-12)


def test_free_boundary_descent_does_not_stop_before_full_window(tmp_path):
    """ftol/xtol must not stop the descent loop from the first update's two-entry moving average."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(updates=6, tolerance=1e-12, dt=1e-3, force_ratio_limit=0.0, gtol=0.0)
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    ds = run.diagnostics_stage[list(run.diagnostics_stage.children)[-1]].to_dataset()
    assert np.isfinite(ds.free_boundary_target.values).sum() == 6


def test_normal_displacement_basis_3d(tmp_path):
    """nfp=3 torus with a helical boundary: each column of P, evaluated as a boundary displacement, equals
    phi_k(theta, zeta) times the in-plane unit normal (up to the truncation of the boundary basis)."""
    from gvec.core.minimize import MinimizationState
    from gvec.core.run.free_boundary import _boundary_mode_list, _normal_displacement_basis

    p = _params()
    p.update(nfp=3, FreeBoundary=True)
    p["X1_b_cos"] = {(0, 0): R0, (1, 0): A0, (1, 1): 0.02}
    p["X2_b_sin"] = {(1, 0): A0, (1, 1): 0.02}
    for q in ("X1", "X2", "LA"):
        p[f"{q}_mn_max"] = [4, 2]
    state = MinimizationState.new(p, rundir=tmp_path)
    modes = _boundary_mode_list(state)
    assert max(md[3] for md in modes) == 2
    P, _ = _normal_displacement_basis(state, oversample=8)
    th = np.linspace(0, 2 * np.pi, 37)[:-1] + 0.1
    ze = np.linspace(0, 2 * np.pi / 3, 13)[:-1] + 0.05
    TH, ZE = np.meshgrid(th, ze, indexing="ij")
    ev = state.evaluate("dX1_dt", "dX2_dt", rho=[1.0], theta=th, zeta=ze).isel(rad=0)
    Rt, Zt = ev.dX1_dt.transpose("pol", "tor").values, ev.dX2_dt.transpose("pol", "tor").values
    nR, nZ = Zt / np.hypot(Rt, Zt), -Rt / np.hypot(Rt, Zt)

    def field(coefs, var):
        f = np.zeros_like(TH)
        for (v, sc, m, n), c in zip(modes, coefs):
            if v == var:
                f += c * (np.sin if sc == "s" else np.cos)(m * TH - n * 3 * ZE)
        return f

    k = [md for md in modes if md[0] == "X1"].index(("X1", "c", 1, 1))
    phi = np.cos(TH - 3 * ZE)
    err = np.hypot(field(P[:, k], "X1") - phi * nR, field(P[:, k], "X2") - phi * nZ).max()
    assert err < 0.05  # truncation of phi * n at m <= 4, |n| <= 2; a wrong zeta frequency gives O(1)


def test_free_boundary_final_parameters_hold_final_boundary(tmp_path):
    """The final parameter file of a free-boundary run describes the moved boundary, not the input one."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(updates=4, tolerance=1e-12, dt=1e-1, force_ratio_limit=0.0, gtol=0.0, ftol=0.0, xtol=0.0)
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    st = run.state
    for theta in (0.0, np.pi):
        R_state = st.evaluate("X1", rho=[1.0], theta=[theta], zeta=[0.0]).X1.values.item()
        R_par = sum(v * np.cos(m * theta) for (m, n), v in st.parameters["X1_b_cos"].items())
        R_in = sum(v * np.cos(m * theta) for (m, n), v in p["X1_b_cos"].items())
        assert abs(R_state - R_in) > 1e-6  # the boundary did move
        assert R_par == pytest.approx(R_state, abs=1e-12)


def test_newton_boundary_force_excludes_spectral_condensation(tmp_path):
    """The boundary force used by the Newton update is the pressure balance: it does not depend on the weight of the
    spectral-condensation term, whose normal part would bias the converged shape."""
    from gvec.core.minimize import MinimizationState

    def forces(epsilon, rundir):
        p = _params()
        p["FreeBoundary"] = True
        for q in ("X1", "X2", "LA"):
            p[f"{q}_mn_max"] = [2, 0]
        p["X1_b_cos"][(2, 0)] = 0.03  # m = 2 content: the condensation term is nonzero
        p["X2_b_sin"][(2, 0)] = -0.02
        p["free_boundary_epsilon_spectral"] = epsilon
        rundir.mkdir()
        st = MinimizationState.new(p, rundir=rundir)
        t_n, z_n = st.t_n, st.z_n
        st.set_Bext_squared(np.full(t_n * z_n, 0.3))
        return st._eval_boundary_forces()

    f0 = forces(0.0, tmp_path / "a")
    f1 = forces(0.5, tmp_path / "b")
    assert np.abs(f0).max() > 0
    np.testing.assert_allclose(f1, f0, rtol=0, atol=1e-12)
