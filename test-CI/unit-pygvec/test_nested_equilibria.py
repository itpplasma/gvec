"""Small fixed-boundary solves against exact non-axisymmetric equilibria."""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from nested_oracles import DELTA, EPS, field, surface_distance

MU0 = 4 * np.pi * 1e-7
INPUTS = Path(__file__).parents[1] / "data" / "nested-oracles"


def measure(tau, resolution, work):
    import f90nml
    import gvec

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
        totalIter=20000,
        minimize_tol=1e-6,
        logIter=1000,
    )
    started = time.perf_counter()
    result = gvec.run(parameters, runpath=work / "solve", quiet=True)
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
        claim="candidate; sampled diagnostics; continuous bounds unavailable",
    )
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("[V] " + json.dumps(metrics, sort_keys=True))
    return metrics


@pytest.mark.parametrize("tau", [0.0, 0.4], ids=["landreman", "issan"])
def test_nested_fixed_boundary(tau, tmp_path):
    metrics = measure(tau, 6, tmp_path)
    assert metrics["native_force"] <= 1e-6
    assert metrics["B_relative"] < 0.01
    assert metrics["Bmag_relative"] < 0.01
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
