"""A manufactured quintic map folds between the six integration points."""

import numpy as np
import pytest
from scipy.interpolate import BSpline


def test_reject_fold_between_integration_points(tmp_path):
    import gvec
    from gvec.core.minimize import MinimizationState

    coarse = (np.polynomial.legendre.leggauss(6)[0] + 1) / 2
    dense = (np.polynomial.legendre.leggauss(12)[0] + 1) / 2
    center = dense[-3]
    width = 0.03

    # R=3+f(rho)cos(theta), Z=rho*sin(theta)/2; f' has a narrow negative well.
    # f'=(rho**2-center**2)**2-width**2, so the fold is known analytically.
    def radial(rho):
        return rho**5 / 5 - 2 * center**2 * rho**3 / 3 + (center**4 - width**2) * rho

    assert np.min((coarse**2 - center**2) ** 2 - width**2) > 0
    assert np.min((dense**2 - center**2) ** 2 - width**2) < 0
    parameters = {
        "ProjectName": "offgrid",
        "nfp": 1,
        "which_hmap": 1,
        "PhiEdge": -1.0,
        "iota": {"type": "polynomial", "coefs": [0.5]},
        "pres": {"type": "polynomial", "coefs": [0.0]},
        "X1_b_cos": {(0, 0): 3.0, (1, 0): radial(1.0)},
        "X2_b_sin": {(1, 0): 0.5},
        "X1_a_cos": {(0, 0): 3.0},
        "X1_mn_max": [1, 0],
        "X2_mn_max": [1, 0],
        "LA_mn_max": [1, 0],
        "sgrid_nElems": 1,
        "X1X2_deg": 5,
        "LA_deg": 5,
        "degGP": 5,
        "fac_nyq": 64,
        "init_LA": False,
        "MinimizerType": 0,
    }
    initial = MinimizationState.new(parameters, rundir=tmp_path)
    initial.save()
    initial.unbind()
    lines = initial.statefile.read_text().splitlines()
    knots = np.r_[np.zeros(6), np.ones(6)]
    rho = np.linspace(0, 1, 6)
    coefficients = np.linalg.solve(BSpline.design_matrix(rho, knots, 5).toarray(), radial(rho))
    start = next(i for i, line in enumerate(lines) if line.startswith("## X1:")) + 1
    count = int(lines[9].split(",")[3])
    for i in range(start, start + count):
        entry = np.fromstring(lines[i], sep=",")
        if tuple(entry[:2]) == (1, 0):
            lines[i] = "1,0," + ",".join(f"{value:.16e}" for value in coefficients)
    folded = tmp_path / "folded.dat"
    folded.write_text("\n".join(lines) + "\n")
    state = MinimizationState(initial.parameterfile, folded)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    assert float(state.evaluate("Jac", rho=coarse, theta=theta, zeta=[0.0]).Jac.min()) > 0
    assert float(state.evaluate("Jac", rho=dense, theta=theta, zeta=[0.0]).Jac.min()) < 0
    # An effectively zero step cannot repair this already folded map. It must
    # exhaust rejections without accepting even one trial. The old gate accepts it.
    with pytest.raises(gvec.errors.InitializationError, match="repeated timestep reductions"):
        state._minimize(dt=1e-30, maxIter=1, minimize_tol=1e-30, logfinaliter=False)
    assert state.get_iterations() == 0
    assert "off-grid trial Jacobian min/max" in state.stdout
