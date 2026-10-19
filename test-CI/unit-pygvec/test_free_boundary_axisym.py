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


class _ToyBoundaryState:
    """Stand-in for the interior equilibrium of a Newton free-boundary run: the unknowns are the boundary
    coefficients y themselves, kept here, with checkpoint/restore."""

    def __init__(self, y):
        self.y = y
        self._saved = y.copy()

    def reset_descent(self):
        pass

    def solve(self, **kwargs):
        pass

    def _checkpoint_dofs(self):
        self._saved = self.y.copy()

    def _restore_dofs(self):
        self.y = self._saved.copy()

    def get_iterations(self):
        return 0


def _toy_newton(monkeypatch, y_start, y_ref, stiffness, curvature, **fbp):
    """Run FreeBoundaryRun._run_newton on the residual g(y) = stiffness * e + curvature * e**2, e = y - y_ref, whose
    zero is the reference state. The quadratic term makes the finite-difference Jacobian (step 1e-4) of the
    soft mode stiff by 40 %, so the steps shrink geometrically while |g| is still far above zero."""
    from gvec.core.run import free_boundary as fb

    k = len(y_ref)
    run = fb.FreeBoundaryRun.__new__(fb.FreeBoundaryRun)
    run.state = _ToyBoundaryState(np.array(y_start, dtype=float))
    run._state_parameters = {"free_boundary": dict(newton_workers=1, newton_max_step=0.05, newton_fd_step=1e-4, **fbp)}
    run.free_boundary_target = 1.0
    run._update_diagnostics = lambda *args: None
    monkeypatch.setattr(fb, "_normal_displacement_basis", lambda state: (np.eye(k), np.ones(k)))

    def residual(digits):
        e = run.state.y - y_ref
        return stiffness * e + curvature * e**2, ()

    def move(pert, blend_depth, minimize_tol, maxIter):
        run.state.y = run.state._saved + pert  # relative to the checkpointed base state
        return True

    run._fb_residual = residual
    run._fb_move_and_solve = move
    run._run_newton(100, 0.0, 1e-8, 1000, 8, -1.0)
    return run


def test_newton_small_step_does_not_stop_above_gtol(monkeypatch):
    """A small step is not convergence: with newton_gtol the update continues, with a rebuilt Jacobian, until |g|
    is below the level the problem reaches, and an offset start returns to the reference state."""
    y_ref = np.array([0.30, -0.10, 0.02])
    stiffness = np.array([2e4, 1e4, 50.0])
    y_start = y_ref + np.array([0.0, 0.0, 2e-3])
    kw = dict(newton_xtol=1e-4, newton_rtol=0.0)
    legacy = _toy_newton(monkeypatch, y_start, y_ref, stiffness, 2e5, **kw)
    fixed = _toy_newton(monkeypatch, y_start, y_ref, stiffness, 2e5, newton_gtol=1e-7, **kw)
    assert np.linalg.norm(legacy.state.y - y_ref) > 1e-7  # stops on the step size, far from the reference
    assert np.linalg.norm(fixed.state.y - y_ref) < 1e-8
    assert fixed.gnorm < 1e-7


def test_saved_parameters_do_not_repeat_boundary_perturbation(tmp_path):
    """The parameter file saved by a free-boundary state holds the current boundary, which already contains the
    boundary perturbation of the parameters: a state started from that file has the same boundary."""
    from gvec.core.minimize import MinimizationState

    p = _params()
    p.update(ProjectName="SP", FreeBoundary=True, boundary_perturb=True, boundary_perturb_type="cosm")
    p["X1pert_b_cos"] = {(0, 0): 0.01}
    (tmp_path / "a").mkdir()
    st = MinimizationState.new(p, rundir=tmp_path / "a")
    x0 = st._get_boundary_dofs()
    st.save()
    (tmp_path / "b").mkdir()
    st2 = MinimizationState.new(gvec.util.read_parameters(st.parameterfile), rundir=tmp_path / "b")
    np.testing.assert_allclose(st2._get_boundary_dofs(), x0, rtol=0, atol=1e-14)


def test_save_and_restart_reproduce_state(tmp_path, monkeypatch):
    """save() writes the state file into the run directory it reports (also when the working directory is
    elsewhere), and a state restarted from the saved parameter and state files has the same boundary and the same
    boundary force."""
    from gvec.core.minimize import MinimizationState

    monkeypatch.chdir(tmp_path)
    p = _params()
    p.update(ProjectName="RS", FreeBoundary=True, init_LA=False)  # init_LA=T would recompute lambda on restart
    p["pres"] = {"type": "polynomial", "coefs": [1e4, -1e4]}
    (tmp_path / "a").mkdir()
    st = MinimizationState.new(p, rundir=tmp_path / "a")
    B2 = np.linspace(1.0, 2.0, st.t_n * st.z_n)
    st.set_Bext_squared(B2)
    st.solve(logfinaliter=False, maxIter=300, minimize_tol=1e-8)
    pert = np.zeros_like(st._get_boundary_dofs())
    pert[0] = 0.003
    assert st._apply_boundary_perturbation(pert, 0.6)
    x0, f0 = st._get_boundary_dofs(), st._eval_boundary_forces()
    st.save()
    assert st.statefile.parent == tmp_path / "a" and st.statefile.exists()
    (tmp_path / "b").mkdir()
    st2 = MinimizationState.new(gvec.util.read_parameters(st.parameterfile), rundir=tmp_path / "b", statefile=st.statefile)
    st2.set_Bext_squared(B2)
    np.testing.assert_allclose(st2._get_boundary_dofs(), x0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(st2._eval_boundary_forces(), f0, rtol=0, atol=1e-10 * np.abs(f0).max())


def test_free_boundary_stage_does_not_inherit_convergence_norms(tmp_path):
    """Each free-boundary stage starts its gtol/xtol/ftol statistics afresh: a stage must not stop before its first
    update on the gradient norm of the previous stage."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    # gtol is met by the first measured gradient norm, so every stage stops after exactly one update
    p["free_boundary"] = dict(updates=4, tolerance=1e-12, dt=1e-3, force_ratio_limit=0.0, gtol=1e30)
    p["stages"] = [{}, {}]
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    stages = sorted(run.diagnostics_stage.children)
    # entry 0 of each stage is the state before its first update
    entries = [np.isfinite(run.diagnostics_stage[s].to_dataset().free_boundary_target.values).sum() for s in stages]
    assert entries == [2, 2]


def test_free_boundary_descent_resets_minimizer_at_min_dt(tmp_path, monkeypatch):
    """When the step-size heuristic holds dt at min_dt, the free-boundary minimizer is reset."""
    pytest.importorskip("fbtools")
    from gvec.core.minimize import MinimizationState

    calls = []
    update = MinimizationState._free_boundary_update

    def record(self, *args, **kwargs):
        calls.append((kwargs["dt"], kwargs["reset_minimizer"]))
        return update(self, *args, **kwargs)

    monkeypatch.setattr(MinimizationState, "_free_boundary_update", record)
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    # initial dt below min_dt: the first update is taken at dt = min_dt
    p["free_boundary"] = dict(updates=2, tolerance=1e-12, dt=1e-4, min_dt=2e-4, force_ratio_limit=0.0, gtol=0.0)
    gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    assert calls[0] == (pytest.approx(2e-4), True)


def test_B2_exterior_counterclockwise_zeta(tmp_path):
    """For a counter-clockwise zeta (G-frame of a circular axis traversed with +phi), the poloidal circulation of
    the exterior plasma field is mu_0 I_tor, with the sign of GVEC's I_tor (the circulation of B along theta)."""
    pytest.importorskip("fbtools")
    from gvec.core.run.free_boundary import _get_B2_fp

    nz = 9
    z = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    axis = dict(
        nzeta=nz,
        zetafull=z,
        xyz=np.array([R0 * np.cos(z), R0 * np.sin(z), 0 * z]),
        Nxyz=np.array([np.cos(z), np.sin(z), 0 * z]),
        Bxyz=np.array([0 * z, 0 * z, -1 + 0 * z]),  # (N, B, dX0/dzeta) right-handed
    )
    with gvec.util.chdir(tmp_path):
        gvec.gframe.write_Gframe_ncfile("ccw-Gframe.nc", {"nfp": 1, "axis": axis})
    p = _params()
    p.update(which_hmap=21, hmap_ncfile=str(tmp_path / "ccw-Gframe.nc"))
    p["X1_b_cos"] = {(1, 0): A0}
    p["X1_a_cos"] = {(0, 0): 0.0}
    st = gvec.State.new(p)
    assert not st.clockwise_zeta

    nt, nz = st.get_integration_points()[2], st.get_integration_points()[4]
    loop = Coil(np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]).T * 20.0, 0.0)
    theta = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    zeta = (np.arange(nz) + 0.5) * 2 * np.pi / nz
    ev = st.evaluate("e_theta", "B", "I_tor", rho=1.0, theta=theta, zeta=zeta).isel(rad=0)
    e_theta = ev.e_theta.transpose("xyz", "tor", "pol").values
    B_in = ev.B.transpose("xyz", "tor", "pol").values
    # GVEC's I_tor is the circulation of B along theta
    i_tor = ev.I_tor.values.ravel()[0]
    assert (B_in * e_theta).sum(axis=(0, 2))[0] * 2 * np.pi / nt / mu_0 == pytest.approx(i_tor, rel=1e-6)

    _, B_plas, _, _ = _get_B2_fp(st, CoilSet([loop]), nt, nz, i_tor, True, 8)
    circulation = (B_plas * e_theta).sum(axis=(0, 2)) * 2 * np.pi / nt
    np.testing.assert_allclose(circulation, mu_0 * i_tor, rtol=1e-3)


def test_newton_parallel_fd_jacobian_matches_serial(tmp_path):
    """The FD Jacobians of the Newton update computed by the worker pool (dynamically scheduled columns, workers kept
    for the second Jacobian from a new base state) equal the serial ones, up to the interior tolerance."""
    pytest.importorskip("fbtools")
    runs = {}
    for workers in (1, 2):
        p = _params()
        for q in ("X1", "X2", "LA"):
            p[f"{q}_mn_max"] = [2, 0]
        p["X1_b_cos"][(2, 0)] = 0.02
        p["totalIter"] = 200000
        p["minimize_tol"] = 1e-9
        p["free_boundary"] = dict(
            updates=4, tolerance=1e-14, method="newton", newton_workers=workers, newton_fd_step=1e-4,
            newton_rtol=0.0, newton_xtol=0.0, newton_max_step=0.01, ramp_force_tol=False,
        )
        runs[workers] = gvec.run(p, runpath=tmp_path / f"w{workers}", coil_set=_axis_wire(2e6))
    J1, J2 = runs[1]._newton_last_fd_jacobian, runs[2]._newton_last_fd_jacobian
    assert J1.shape == J2.shape and J1.shape[0] >= 3
    np.testing.assert_allclose(J2, J1, rtol=0, atol=1e-4 * np.abs(J1).max())
    h1, h2 = runs[1].newton_history, runs[2].newton_history
    assert len(h1) == len(h2) == 3
    assert h2[-1]["jacobians"] == 2  # the second FD Jacobian reuses the worker pool from a new base state
    np.testing.assert_allclose(h2[-1]["residual"], h1[-1]["residual"], rtol=1e-3)
