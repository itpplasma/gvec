"""A rejected trial must preserve the accepted map, including after iteration 100."""

import numpy as np


def test_retry_negative_jacobian_after_iteration_100(tmp_path):
    from gvec.core.minimize import MinimizationState

    parameters = {
        "ProjectName": "backtrack",
        "nfp": 1,
        "which_hmap": 1,
        "PhiEdge": -1.0,
        "iota": {"type": "polynomial", "coefs": [0.5]},
        "pres": {"type": "polynomial", "coefs": [0.0]},
        "X1_b_cos": {(0, 0): 3.0, (1, 0): 0.5},
        "X2_b_sin": {(1, 0): 0.5},
        "X1_a_cos": {(0, 0): 3.0},
        "X1_mn_max": [2, 0],
        "X2_mn_max": [2, 0],
        "LA_mn_max": [2, 0],
        "sgrid_nElems": 2,
        "X1X2_deg": 3,
        "LA_deg": 3,
        "logIter": 100,
    }
    state = MinimizationState.new(parameters, rundir=tmp_path)
    state._minimize(maxIter=101, minimize_tol=1e-30, logfinaliter=False)
    before = state.get_iterations()
    assert before == 101
    state.reset_LA()
    state.reset_descent()
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    boundary = state.evaluate("pos", rho=[1.0], theta=theta, zeta=[0.0]).pos.copy()

    # Force an inadmissible prediction on a well-defined accepted state.
    state._minimize(dt=1000.0, maxIter=1, minimize_tol=1e-30, logfinaliter=False)

    assert state.get_iterations() == before + 1
    assert state.dt_fixed < 1000.0
    assert "retry from last accepted state" in state.stdout
    result = state.evaluate("pos", "Jac", rho=[0.2, 0.5, 1.0], theta=theta, zeta=[0.0])
    assert np.isfinite(result.Jac).all()
    assert (result.Jac > 0).all()
    np.testing.assert_allclose(result.pos.isel(rad=-1), boundary.isel(rad=0), atol=1e-12)
