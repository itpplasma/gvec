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
