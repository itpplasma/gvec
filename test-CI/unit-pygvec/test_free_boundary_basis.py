"""Normal-space coordinates must not inherit the BLAS singular-vector frame."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("rank", [0, 1, 7, 12])
def test_canonical_range_ignores_frame(rank):
    from gvec.core.run.free_boundary import _canonical_range_basis

    rng = np.random.default_rng(42)
    P = np.linalg.qr(rng.normal(size=(12, rank)))[0]
    rotation = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    canonical = _canonical_range_basis(P)
    np.testing.assert_allclose(canonical.T @ canonical, np.eye(rank), rtol=0, atol=1e-12)
    np.testing.assert_allclose(canonical @ canonical.T, P @ P.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        _canonical_range_basis(P @ rotation), canonical, rtol=0, atol=1e-12
    )


def test_canonical_range_tied_axes():
    from gvec.core.run.free_boundary import _canonical_range_basis

    # Equal projected axis lengths: DOF order and positive pivots fix both signs.
    P = np.array([[1, 0], [0, 1], [1, 0], [0, -1]]) / np.sqrt(2)
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(_canonical_range_basis(P @ rotation), P, rtol=0, atol=1e-12)


def test_normal_basis_and_newton_across_blas_kernels(tmp_path, record_property):
    results = []
    for kernel in ("native", "Nehalem"):
        work = tmp_path / kernel
        work.mkdir()
        env = dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        env.pop("OPENBLAS_CORETYPE", None)
        env.pop("DISPLAY", None)
        env["MPLBACKEND"] = "Agg"
        if kernel != "native":
            env["OPENBLAS_CORETYPE"] = kernel
        with (work / "output.log").open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(work)],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        assert completed.returncode == 0, (work / "output.log").read_text()[-12000:]
        results.append(np.load(work / "result.npz"))
    native, nehalem = results
    for key in ("basis", "scales", "iterates", "steps"):
        difference = float(np.max(np.abs(native[key] - nehalem[key])))
        record_property(f"{key}_kernel_max_difference", difference)
        np.testing.assert_allclose(native[key], nehalem[key], rtol=0, atol=1e-12, err_msg=key)
    np.testing.assert_array_equal(native["decisions"], nehalem["decisions"])
    assert 3 <= len(native["steps"]) < 30
    np.testing.assert_allclose(native["steps"][0], 0.01, rtol=0, atol=1e-14)
    assert np.max(native["steps"]) <= 0.01 + 1e-14
    np.testing.assert_allclose(native["iterates"][-1], 0.0, rtol=0, atol=1e-11)


def _kernel_case(work):
    from gvec.core.minimize import MinimizationState
    from gvec.core.run import free_boundary as fb
    from test_free_boundary_axisym import _params, _ToyBoundaryState

    # Circular 3D boundary: many exactly degenerate singular values and a
    # tangential nullspace. Independent of coil/BIEST numerical errors.
    p = _params()
    p.update(nfp=3, FreeBoundary=True)
    for var in ("X1", "X2", "LA"):
        p[f"{var}_mn_max"] = [2, 2]
    state = MinimizationState.new(p, rundir=work)
    basis, scales = fb._normal_displacement_basis(state)
    assert basis.shape == (25, 18)

    # A manufactured, well-conditioned force has the independent root x=0.
    # Exercise the real FD, LM, clipping, backtracking and Broyden implementation
    # without amplifying interior/BIEST noise in a nearly unforced physical mode.
    run = fb.FreeBoundaryRun.__new__(fb.FreeBoundaryRun)
    run.state = _ToyBoundaryState(basis @ np.linspace(0.02, 0.04, basis.shape[1]))
    run._state_parameters = {
        "free_boundary": dict(
            newton_workers=1,
            newton_fd_step=1e-4,
            newton_max_step=0.01,
            newton_gtol=1e-11,
            newton_rtol=0.0,
            newton_xtol=0.0,
        )
    }
    run.free_boundary_target = 1.0
    iterates = [run.state.y.copy()]
    stiffness = np.linspace(1.0, 2.0, basis.shape[0])

    def residual(digits):
        x = run.state.y
        return stiffness * x + 10.0 * x**3, ()

    def move(pert, blend_depth, minimize_tol, maxIter):
        run.state.y = run.state._saved + pert
        run.state.solve(maxIter=maxIter)
        return True

    def diagnostics(*args):
        iterates.append(run.state.y.copy())

    run._fb_residual = residual
    run._fb_move_and_solve = move
    run._update_diagnostics = diagnostics
    fb._normal_displacement_basis = lambda state: (basis, scales)
    run._run_newton(30, 0.6, 1e-12, 1, 8, -1.0)
    assert run.gnorm < 1e-11
    np.savez(
        work / "result.npz",
        basis=basis,
        scales=scales,
        iterates=iterates,
        steps=[entry["step_max"] for entry in run.newton_history],
        decisions=[
            [entry[key] for key in ("update", "alpha", "jacobians")]
            for entry in run.newton_history
        ],
    )


if __name__ == "__main__":
    _kernel_case(Path(sys.argv[1]))
