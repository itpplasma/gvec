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


@pytest.fixture
def state(tmp_path, monkeypatch):
    # function scope: pygvec holds one global Fortran state, which any other State.new or run in this module replaces
    monkeypatch.chdir(tmp_path)  # State.new writes parameter.ini to the working directory
    return gvec.State.new(_params())


def test_axisymmetric_state_has_one_toroidal_point(state):
    assert state.get_integration_points()[4] == 1


def test_winding_number_axisymmetric(state):
    # zeta = -phi for which_hmap=1
    assert state.winding_nr == pytest.approx(-1.0, abs=1e-12)
    assert state.clockwise_zeta


def test_midplane_rays_ignore_poloidal_parameterization(state):
    from free_boundary_oracles import midplane_ray_radii

    class Reparameterized:
        parameters = state.parameters

        def evaluate(self, *args, theta, **kwargs):
            theta = np.asarray(theta)
            return state.evaluate(*args, theta=theta + 0.37 + 0.2 * np.sin(theta), **kwargs)

    expected = [[R0 - A0, R0 + A0]] * 2
    for surface in (state, Reparameterized()):
        np.testing.assert_allclose(
            midplane_ray_radii(surface, [0.0, np.pi]), expected, rtol=0, atol=1e-12
        )


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
    loop = Coil(
        np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]).T * 20.0, 0.0
    )
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


def test_free_boundary_resolution_proposal_after_newton_stalls_at_update_0(tmp_path):
    """Newton that makes no step leaves only update 0 in the diagnostics; the resolution proposal at the end of
    the stage must use that column, not the (still NaN) last one."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 2000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(
        updates=4,
        tolerance=1e-12,
        method="newton",
        newton_max_backtracks=0,  # no trial step is ever accepted
        force_ratio_limit=2.0,  # > 1: always ask for the resolution proposal
    )
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    fb = run.diagnostics_stage["stage_1"].force_balance
    assert fb.isel(updates=0).notnull().all()
    assert fb.isel(updates=-1).isnull().all()


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


def test_boundary_checkpoint_restores_current_fit(tmp_path):
    """A rejected current-controlled trial must restore its profile as well as geometry."""
    from scipy.interpolate import make_interp_spline

    state = _minimization_state(tmp_path)
    rho = np.linspace(0.0, 1.0, 9)
    profile = make_interp_spline(rho**2, 0.4 + 0.1 * rho**2, k=5)
    state.iota = profile
    initial = state.evaluate("iota", "B", rho=rho[1:], theta=0.3, zeta=0.0)
    state._checkpoint_dofs()
    state.iota = make_interp_spline(rho**2, 0.7 - 0.2 * rho**2, k=5)
    state._restore_dofs()
    restored = state.evaluate("iota", "B", rho=rho[1:], theta=0.3, zeta=0.0)
    np.testing.assert_allclose(state.iota(rho**2), profile(rho**2), rtol=0, atol=1e-14)
    np.testing.assert_allclose(restored.iota, initial.iota, rtol=0, atol=1e-14)
    np.testing.assert_allclose(restored.B, initial.B, rtol=0, atol=1e-14)


def test_update_boundary_returns_the_used_stepsize(tmp_path):
    """pygvec's update_boundary returns the free-boundary step-size (it used to return an undefined value)."""
    state = _minimization_state(tmp_path)
    state.set_Bext_squared(np.full(state.t_n * state.z_n, 0.3))
    state.reset_descent()
    dt = 3e-3
    dt_out = state.update_boundary(dt, True)
    assert dt_out == pytest.approx(state.dt_free, rel=0, abs=0)
    assert 0.0 < dt_out <= dt


@pytest.mark.parametrize("vertical", [False, True])
def test_normal_displacement_basis_circle(tmp_path, vertical):
    """All allowed DOFs' normal traces survive the gauge, including rigid shifts.

    The analytic circle normal is (cos(theta), sin(theta)); this oracle also
    checks the highest product harmonic, absent from the old X1-only basis.
    """
    from gvec.core.minimize import MinimizationState
    from gvec.core.run.free_boundary import _boundary_mode_list, _normal_displacement_basis

    p = _params()
    p["FreeBoundary"] = True
    for q in ("X1", "X2", "LA"):
        p[f"{q}_mn_max"] = [2, 0]
    if vertical:
        p.update(X2_sin_cos="_sincos_", X2_b_cos={(0, 0): 0.0})
    state = MinimizationState.new(p, rundir=tmp_path)
    modes = _boundary_mode_list(state)
    P, _ = _normal_displacement_basis(state)
    theta = np.arange(128) * 2 * np.pi / 128
    A = np.column_stack(
        [
            (np.sin if sc == "s" else np.cos)(m * theta)
            * (np.cos(theta) if var == "X1" else np.sin(theta))
            for var, sc, m, _ in modes
        ]
    )
    assert P.shape[1] == (7 if vertical else 4)
    np.testing.assert_allclose(P.T @ P, np.eye(P.shape[1]), atol=1e-12)
    np.testing.assert_allclose(A @ P @ np.linalg.lstsq(A @ P, A, rcond=1e-10)[0], A, atol=1e-12)
    for var in ("X1", "X2") if vertical else ("X1",):
        shift = np.zeros(len(modes))
        shift[modes.index((var, "c", 0, 0))] = 1.0
        np.testing.assert_allclose(P @ P.T @ shift, shift, atol=1e-12)
    # The one exact relabelling null is (-1 + cos(2t), sin(2t)).
    # Its derivative-metric gauge is dR_cos2 + dZ_sin2 = 0; translations are free.
    np.testing.assert_allclose(
        P[modes.index(("X1", "c", 2, 0))] + P[modes.index(("X2", "s", 2, 0))],
        0,
        atol=1e-12,
    )


def test_free_boundary_descent_does_not_stop_before_full_window(tmp_path):
    """ftol/xtol must not stop the descent loop from the first update's two-entry moving average."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(
        updates=6, tolerance=1e-12, dt=1e-3, force_ratio_limit=0.0, gtol=0.0
    )
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    ds = run.diagnostics_stage[list(run.diagnostics_stage.children)[-1]].to_dataset()
    assert np.isfinite(ds.free_boundary_target.values).sum() == 6


def test_normal_displacement_basis_3d(tmp_path):
    """A helical boundary retains all representable normal traces with the right nfp."""
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

    A = np.column_stack(
        [
            (
                field(np.eye(len(modes))[:, i], "X1") * nR
                + field(np.eye(len(modes))[:, i], "X2") * nZ
            ).ravel()
            for i in range(len(modes))
        ]
    )
    AP = A @ P
    np.testing.assert_allclose(AP @ np.linalg.lstsq(AP, A, rcond=1e-10)[0], A, atol=1e-10)
    phi = np.cos(TH - 3 * ZE)
    coefs = P @ np.linalg.lstsq(AP, phi.ravel(), rcond=1e-10)[0]
    err = np.abs(field(coefs, "X1") * nR + field(coefs, "X2") * nZ - phi).max()
    assert (
        err < 0.05
    )  # truncation of phi * n at m <= 4, |n| <= 2; a wrong zeta frequency gives O(1)


def test_free_boundary_final_parameters_hold_final_boundary(tmp_path):
    """The final parameter file of a free-boundary run describes the moved boundary, not the input one."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    p["free_boundary"] = dict(
        updates=4, tolerance=1e-12, dt=1e-1, force_ratio_limit=0.0, gtol=0.0, ftol=0.0, xtol=0.0
    )
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
        self.iterations = 0

    def reset_descent(self):
        pass

    def solve(self, **kwargs):
        self.iterations += kwargs["maxIter"]  # worst case: every solve uses its whole allowance

    def _checkpoint_dofs(self):
        self._saved = self.y.copy()

    def _restore_dofs(self):
        self.y = self._saved.copy()

    def get_iterations(self):
        return self.iterations


def _toy_newton(
    monkeypatch,
    y_start,
    y_ref,
    stiffness,
    curvature,
    maxIter=1000,
    totaliter=None,
    basis=None,
    residual=None,
    **fbp,
):
    """Run Newton on g(y) = stiffness * e + curvature * e**2, e = y - y_ref.
    The quadratic soft mode needs several steps; a small step alone can leave
    the residual far above its target."""
    from gvec.core.run import free_boundary as fb

    k = len(y_ref)
    run = fb.FreeBoundaryRun.__new__(fb.FreeBoundaryRun)
    run.state = _ToyBoundaryState(np.array(y_start, dtype=float))
    run._state_parameters = {
        "free_boundary": dict(
            newton_workers=1, newton_max_step=0.05, newton_fd_step=1e-4, **fbp
        )
    }
    run.free_boundary_target = 1.0
    run._update_diagnostics = lambda *args: None
    if basis is None:
        basis = lambda y: np.eye(k)  # noqa: E731

    def normal_basis(state):
        P = basis(state.y)
        return P, np.ones(P.shape[1])

    monkeypatch.setattr(fb, "_normal_displacement_basis", normal_basis)

    def toy_residual(digits):
        e = run.state.y - y_ref
        if residual is not None:
            return residual(run.state.y), ()
        return stiffness * e + curvature * e**2, ()

    def move(pert, blend_depth, minimize_tol, maxIter):
        run.state.y = run.state._saved + pert  # relative to the checkpointed base state
        run.state.solve(maxIter=maxIter)
        return True

    run._fb_residual = toy_residual
    run._fb_move_and_solve = move
    run._run_newton(100, 0.0, 1e-8, maxIter, 8, -1.0, totaliter=totaliter)
    return run


def test_newton_damping_preserves_unforced_direction(monkeypatch):
    """Diagonal LM damping cannot induce a displacement in a null force direction."""
    run = _toy_newton(
        monkeypatch,
        [0.01, 0.0],
        np.zeros(2),
        np.array([1.0, 0.0]),
        0.0,
        newton_lm=1.0,
    )
    assert abs(run.state.y[0]) < 1e-6
    assert run.state.y[1] == pytest.approx(0.0, rel=0, abs=1e-14)


def test_newton_already_converged_restart(monkeypatch):
    """A root needs no trial decrease or Jacobian, including at exactly zero force."""
    run = _toy_newton(monkeypatch, np.zeros(2), np.zeros(2), np.ones(2), 0.0, newton_gtol=1e-8)
    assert run.gnorm == 0.0
    assert run.newton_history == []
    assert run.newton_jacobian is None


def test_newton_small_step_does_not_stop_above_gtol(monkeypatch):
    """A small step is not convergence: with newton_gtol the update continues, with a rebuilt Jacobian, until |g|
    is below the level the problem reaches, and an offset start returns to the reference state."""
    y_ref = np.array([0.30, -0.10, 0.02])
    stiffness = np.array([2e4, 1e4, 50.0])
    y_start = y_ref + np.array([0.0, 0.0, 2e-3])
    kw = dict(newton_xtol=1e-4, newton_rtol=0.0)
    legacy = _toy_newton(monkeypatch, y_start, y_ref, stiffness, 2e5, **kw)
    fixed = _toy_newton(monkeypatch, y_start, y_ref, stiffness, 2e5, newton_gtol=1e-7, **kw)
    assert (
        np.linalg.norm(legacy.state.y - y_ref) > 1e-7
    )  # stops on the step size, far from the reference
    assert np.linalg.norm(fixed.state.y - y_ref) < 1e-8
    assert fixed.gnorm < 1e-7


def test_newton_respects_totaliter_budget(monkeypatch):
    """The Newton path stops when the totalIter budget is used up (like the descent path), also within the
    finite-difference Jacobian, and limits each interior solve to the iterations left."""
    y_ref = np.array([0.30, -0.10, 0.02])
    y_start = y_ref + np.array([1e-3, 0.0, 2e-3])
    stiffness = np.array([2e4, 1e4, 50.0])
    free = _toy_newton(monkeypatch, y_start, y_ref, stiffness, 0.0, maxIter=10)
    assert free.state.iterations > 25  # unlimited: base solve, Jacobian and steps
    for budget in (0, 5, 25, 45):
        run = _toy_newton(
            monkeypatch, y_start, y_ref, stiffness, 0.0, maxIter=10, totaliter=budget
        )
        assert run.state.iterations <= budget
        # the boundary is left at the last accepted state, not in a half-finished trial
        np.testing.assert_array_equal(run.state.y, run.state._saved)
    # Base solve (10), two residual evaluations (20), then the remaining budget (5).
    assert (
        _toy_newton(
            monkeypatch, y_start, y_ref, stiffness, 0.0, maxIter=10, totaliter=35
        ).state.iterations
        == 35
    )


def test_newton_jacobian_keeps_the_solve_basis_fixed(monkeypatch):
    """The projection and coordinates stay fixed throughout a Newton solve, even when
    the geometric normal basis would rotate after moving the boundary."""
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    K = Q.T @ np.diag([2e4, 1e3, 50.0]) @ Q
    y_ref = np.array([0.30, -0.10, 0.02])
    y_start = y_ref + np.array([0.01, -0.02, 0.015])

    def basis(y):  # rotation about the first axis by an angle that depends on the boundary
        c, s = np.cos(15 * y[1]), np.sin(15 * y[1])
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    run = _toy_newton(
        monkeypatch,
        y_start,
        y_ref,
        None,
        None,
        basis=basis,
        residual=lambda y: K @ (y - y_ref),
        newton_xtol=0.0,
        newton_rtol=1e-6,
        newton_noise_step=0.0,
    )
    P0, P1 = basis(y_start), basis(run.state.y)
    assert np.abs(P1 - P0).max() > 0.1  # recomputing the basis would change it
    J = run.newton_jacobian
    np.testing.assert_allclose(J, P0.T @ K @ P0, rtol=1e-6, atol=1e-6 * np.abs(K).max())
    assert np.linalg.norm(run.state.y - y_ref) < 1e-6


def test_newton_fixed_projection_with_nonzero_full_residual(monkeypatch, record_property):
    """A residual outside the solve space must not feed back through a moving P.

    The independent central difference is of g(y), not just of the full force.
    """
    start = np.array([0.01, -0.02, 0.0])
    reference = np.zeros(3)
    K = np.diag([2e4, -50.0, 1.0])  # the vertical direction is indefinite
    calls = []

    def basis(x):
        calls.append(x.copy())
        angle = 20 * x[0]
        return np.array([[np.cos(angle), 0.0], [0.0, 1.0], [np.sin(angle), 0.0]])

    P = basis(start)
    residual = lambda x: K @ x + 3.0 * np.cross(P[:, 0], P[:, 1])  # noqa: E731
    calls.clear()
    run = _toy_newton(
        monkeypatch,
        start,
        reference,
        None,
        None,
        basis=basis,
        residual=residual,
        newton_gtol=1e-9,
        newton_xtol=0.0,
        newton_rtol=0.0,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], start)
    assert np.linalg.norm(residual(run.state.y)) > 1.0
    assert np.linalg.norm(P.T @ residual(run.state.y)) < 1e-9
    epsilon = 1e-6
    dg = np.column_stack(
        [
            (
                P.T @ residual(run.state.y + epsilon * p)
                - P.T @ residual(run.state.y - epsilon * p)
            )
            / (2 * epsilon)
            for p in P.T
        ]
    )
    record_property(
        "dg_relative_error", np.linalg.norm(run.newton_jacobian - dg) / np.linalg.norm(dg)
    )
    np.testing.assert_allclose(run.newton_jacobian, dg, rtol=1e-8, atol=1e-8)


def test_change_jacobian_basis_rectangular():
    """For P with more rows than columns the new Jacobian is the one of the restriction to span(P_old)."""
    from gvec.core.run.free_boundary import _change_jacobian_basis

    rng = np.random.default_rng(1)
    P_old = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    A = rng.normal(size=(5, 5))
    R = A.T @ A
    J = P_old.T @ R @ P_old
    P_new = P_old @ np.linalg.qr(rng.normal(size=(3, 3)))[0]  # same span, rotated
    np.testing.assert_allclose(
        _change_jacobian_basis(J, P_old, P_new), P_new.T @ R @ P_new, atol=1e-10
    )


def test_saved_parameters_do_not_repeat_boundary_perturbation(tmp_path):
    """The parameter file saved by a free-boundary state holds the current boundary, which already contains the
    boundary perturbation of the parameters: a state started from that file has the same boundary."""
    from gvec.core.minimize import MinimizationState

    p = _params()
    p.update(
        ProjectName="SP", FreeBoundary=True, boundary_perturb=True, boundary_perturb_type="cosm"
    )
    p["X1pert_b_cos"] = {(0, 0): 0.01}
    (tmp_path / "a").mkdir()
    st = MinimizationState.new(p, rundir=tmp_path / "a")
    x0 = st._get_boundary_dofs()
    st.save()
    (tmp_path / "b").mkdir()
    st2 = MinimizationState.new(
        gvec.util.read_parameters(st.parameterfile), rundir=tmp_path / "b"
    )
    np.testing.assert_allclose(st2._get_boundary_dofs(), x0, rtol=0, atol=1e-14)


def test_save_and_restart_reproduce_state(tmp_path, monkeypatch):
    """save() writes the state file into the run directory it reports (also when the working directory is
    elsewhere), and a state restarted from the saved parameter and state files has the same boundary and the same
    boundary force."""
    from gvec.core.minimize import MinimizationState

    monkeypatch.chdir(tmp_path)
    p = _params()
    p.update(
        ProjectName="RS", FreeBoundary=True, init_LA=False
    )  # init_LA=T would recompute lambda on restart
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
    st2 = MinimizationState.new(
        gvec.util.read_parameters(st.parameterfile),
        rundir=tmp_path / "b",
        statefile=st.statefile,
    )
    st2.set_Bext_squared(B2)
    np.testing.assert_allclose(st2._get_boundary_dofs(), x0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        st2._eval_boundary_forces(), f0, rtol=0, atol=1e-10 * np.abs(f0).max()
    )


def test_free_boundary_stage_does_not_inherit_convergence_norms(tmp_path):
    """Each free-boundary stage starts its gtol/xtol/ftol statistics afresh: a stage must not stop before its first
    update on the gradient norm of the previous stage."""
    pytest.importorskip("fbtools")
    p = _params()
    p["totalIter"] = 20000
    p["minimize_tol"] = 1e-5
    # gtol is met by the first measured gradient norm, so every stage stops after exactly one update
    p["free_boundary"] = dict(
        updates=4, tolerance=1e-12, dt=1e-3, force_ratio_limit=0.0, gtol=1e30
    )
    p["stages"] = [{}, {}]
    run = gvec.run(p, runpath=tmp_path / "fb", coil_set=_axis_wire(1e6))
    stages = sorted(run.diagnostics_stage.children)
    # entry 0 of each stage is the state before its first update
    entries = [
        np.isfinite(run.diagnostics_stage[s].to_dataset().free_boundary_target.values).sum()
        for s in stages
    ]
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
    p["free_boundary"] = dict(
        updates=2, tolerance=1e-12, dt=1e-4, min_dt=2e-4, force_ratio_limit=0.0, gtol=0.0
    )
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
    with gvec.util.chdir(tmp_path):  # State.new writes parameter.ini to the working directory
        st = gvec.State.new(p)
    assert not st.clockwise_zeta

    nt, nz = st.get_integration_points()[2], st.get_integration_points()[4]
    loop = Coil(
        np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]).T * 20.0, 0.0
    )
    theta = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    zeta = (np.arange(nz) + 0.5) * 2 * np.pi / nz
    ev = st.evaluate("e_theta", "B", "I_tor", rho=1.0, theta=theta, zeta=zeta).isel(rad=0)
    e_theta = ev.e_theta.transpose("xyz", "tor", "pol").values
    B_in = ev.B.transpose("xyz", "tor", "pol").values
    # GVEC's I_tor is the circulation of B along theta
    i_tor = ev.I_tor.values.ravel()[0]
    assert (B_in * e_theta).sum(axis=(0, 2))[0] * 2 * np.pi / nt / mu_0 == pytest.approx(
        i_tor, rel=1e-6
    )

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
            updates=4,
            tolerance=1e-14,
            method="newton",
            newton_workers=workers,
            newton_fd_step=1e-4,
            newton_rtol=0.0,
            newton_xtol=0.0,
            newton_max_step=0.01,
            ramp_force_tol=False,
        )
        runs[workers] = gvec.run(p, runpath=tmp_path / f"w{workers}", coil_set=_axis_wire(2e6))
    J1, J2 = runs[1]._newton_last_fd_jacobian, runs[2]._newton_last_fd_jacobian
    assert J1.shape == J2.shape and J1.shape[0] >= 3
    np.testing.assert_allclose(J2, J1, rtol=0, atol=1e-4 * np.abs(J1).max())
    h1, h2 = runs[1].newton_history, runs[2].newton_history
    assert len(h1) == len(h2) == 3
    assert (
        h2[-1]["jacobians"] == 2
    )  # the second FD Jacobian reuses the worker pool from a new base state
    np.testing.assert_allclose(h2[-1]["residual"], h1[-1]["residual"], rtol=1e-3)
