"""Small fixed-boundary solves against exact non-axisymmetric equilibria."""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from nested_oracles import DELTA, field, surface_distance
from nested_initialization import axis, exact_state

MU0 = 4 * np.pi * 1e-7
INPUTS = Path(__file__).parents[1] / "data" / "nested-oracles"


def test_independent_integer_oracle():
    """Check this test's complex extension against FortSym-generated samples."""
    samples = np.loadtxt(INPUTS / "field-reference.txt")
    for tau in (0.0, 0.4):
        rows = samples[samples[:, 0] == tau]
        points = rows[:, 1:4]
        psi, pressure, magnetic = field(points, tau)
        np.testing.assert_allclose(
            np.column_stack((psi, pressure, magnetic)), rows[:, 4:], rtol=0, atol=3e-13
        )
        db, gp = np.empty((len(points), 3, 3)), np.empty_like(points)
        for k in range(3):
            shifted = points.astype(complex)
            shifted[:, k] += 1e-30j
            _, p, b = field(shifted, tau)
            db[:, :, k] = b.imag / 1e-30
            gp[:, k] = p.imag / 1e-30
        curl = np.column_stack(
            (db[:, 2, 1] - db[:, 1, 2], db[:, 0, 2] - db[:, 2, 0], db[:, 1, 0] - db[:, 0, 1])
        )
        np.testing.assert_allclose(np.trace(db, axis1=1, axis2=2), 0, atol=2e-12)
        np.testing.assert_allclose(np.cross(curl, magnetic), gp, rtol=0, atol=2e-12)
        assert np.max(surface_distance(points, np.ones(len(points)), tau)) < 1e-11


def measure(tau, resolution, work):
    import f90nml
    import gvec
    from gvec.core.minimize import MinimizationState

    name = "issan" if tau else "landreman"
    source = INPUTS / f"input.{name}"
    parameters = gvec.util.parameters_from_vmec(f90nml.read(source)["indata"], name)
    # Input theta is clockwise, phi is geometric. GVEC uses phi=-zeta.
    # Flip both angles, preserving positive Jacobians and iota=+2.
    parameters = gvec.util.flip_parameters_zeta(parameters)
    parameters = gvec.util.flip_parameters_theta(parameters)
    if tau:
        assert all(
            parameters[key] == "_sincos_" for key in ("X1_sin_cos", "X2_sin_cos", "LA_sin_cos")
        )
        assert max(abs(v) for v in parameters["X2_b_cos"].values()) > 1e-3
    parameters.update(
        X1_mn_max=[resolution, resolution],
        X2_mn_max=[resolution, resolution],
        LA_mn_max=[resolution, resolution],
        sgrid=dict(grid_type=0, nElems=resolution),
        totalIter=2000,
        maxIter=2000,
        fac_nyq=4,
        minimize_tol=1e-6,
        logIter=1000,
    )
    started = time.perf_counter()
    axis(parameters, tau)
    (work / "initial").mkdir()
    initial = MinimizationState.new(parameters, rundir=work / "initial")
    exact = exact_state(initial, tau)
    result = gvec.run(parameters, restartstate=exact, runpath=work / "solve", quiet=True)
    solve_seconds = time.perf_counter() - started
    labels_1d = np.array([0.25, 0.5, 0.75])
    angles = 2 * np.pi * (np.arange(12) + 0.37) / 12
    data = gvec.evaluate(
        result.state,
        "pos",
        "B",
        "J",
        "p",
        "dp_dr",
        "grad_rho",
        "iota",
        rho=np.sqrt(labels_1d),
        theta=angles,
        zeta=angles / result.state.nfp,
    )

    def vector(name):
        return np.asarray(data[name].transpose("rad", "pol", "tor", "xyz")).reshape(-1, 3)

    points, magnetic = vector("pos"), vector("B")
    template = data.pos.isel(xyz=0, drop=True)
    labels = np.broadcast_to(labels_1d[:, None, None], template.shape).ravel()
    psi, pressure, exact = field(points, tau)
    derivative = np.asarray(
        data.dp_dr.broadcast_like(template).transpose("rad", "pol", "tor")
    ).ravel()
    gradient = MU0 * vector("grad_rho") * derivative[:, None]
    force = np.cross(MU0 * vector("J"), magnetic) - gradient

    def rms(values):
        return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))

    metrics = dict(
        tau=tau,
        resolution=resolution,
        solve_seconds=solve_seconds,
        native_force=float(result.max_force),
        iterations=int(result.GVEC_iter_used),
        B_relative=rms(magnetic - exact) / rms(exact),
        Bmag_relative=float(
            np.sqrt(
                np.mean((np.linalg.norm(magnetic, axis=1) - np.linalg.norm(exact, axis=1)) ** 2)
            )
        )
        / rms(exact),
        label_error=float(np.max(np.abs(psi / DELTA - labels))),
        surface_distance=float(np.max(surface_distance(points, labels, tau))),
        iota_error=float(np.max(np.abs(np.asarray(data.iota) - 2))),
        force_relative=rms(force) / rms(gradient),
        claim="[V] exact-interior start; sampled fixed-point diagnostics",
    )
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("[V] " + json.dumps(metrics, sort_keys=True))
    return metrics


@pytest.mark.parametrize("tau", [0.0, 0.4], ids=["landreman", "issan"])
@pytest.mark.parametrize("resolution", [6, 8], ids=["m6", "m8"])
def test_exact_interior_fixed_boundary(tau, resolution, tmp_path):
    metrics = measure(tau, resolution, tmp_path)
    # Three times the gfortran-12.2 exact-start ladder values on this grid.
    # Force is the solver stopping criterion, not a last-digit version gate.
    observed = {0.0: {6: 3.789e-5, 8: 1.927e-6}, 0.4: {6: 1.479e-5, 8: 7.718e-7}}
    assert metrics["native_force"] <= 1e-6
    assert metrics["B_relative"] < 3 * observed[tau][resolution]
    assert metrics["Bmag_relative"] < 3 * observed[tau][resolution]
    assert metrics["surface_distance"] < 0.03 * np.sqrt(DELTA)
    assert metrics["iota_error"] < 1e-4
    assert metrics["force_relative"] < 0.05


if __name__ == "__main__":
    # One cluster mini-ladder point per task; a failed solve cannot hide
    # the evidence for the other symmetry or resolution.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--tau", type=float, choices=(0.0, 0.4), required=True)
    args = parser.parse_args()
    work = args.output.resolve()
    work.mkdir(parents=True)
    measure(args.tau, args.resolution, work)
