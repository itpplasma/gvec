try:
    import numpy as np
    import gvec
except ImportError:
    pass  # tests will be skipped via the check_imports fixture


def test_sincos_boozer_circular_torus(tmp_path):
    """Rotated logical angle requires cosine lambda; toroidal B has analytic 1/R."""
    R0, a, delta = 3.0, 0.6, 0.7
    params = dict(
        ProjectName="circular",
        nfp=3,
        which_hmap=1,
        PhiEdge=1.0,
        iota_type="polynomial",
        iota_coefs=[0.0],
        pres_type="polynomial",
        pres_coefs=[0.0],
        sgrid_nElems=4,
        X1X2_deg=3,
        LA_deg=3,
        maxIter=0,
        init_LA=False,
        X1_mn_max=[12, 0],
        X2_mn_max=[12, 0],
        LA_mn_max=[12, 0],
        X1_sin_cos="_sincos_",
        X2_sin_cos="_sincos_",
        LA_sin_cos="_sincos_",
        X1_a_cos={(0, 0): R0},
        X1_b_cos={(0, 0): R0, (1, 0): a * np.cos(delta)},
        X1_b_sin={(1, 0): a * np.sin(delta)},
        X2_b_cos={(1, 0): -a * np.sin(delta)},
        X2_b_sin={(1, 0): a * np.cos(delta)},
    )
    st = gvec.State.new(params, rundir=tmp_path)
    rho = np.array([0.4, 0.8])
    theta = np.arange(256) * 2 * np.pi / 256
    pot = st.get_boozer(rho, MNfactor=1, sincos="sincos")
    tz = np.array([theta, np.zeros_like(theta)])
    la = st.evaluate_boozer_list_tz_all(pot, "LA", [0, 1], tz)
    R = R0 + a * rho[:, None] * np.cos(theta - delta)
    expected = np.sqrt(R0**2 - (a * rho[:, None]) ** 2) / R - 1
    np.testing.assert_allclose(la[1], expected, atol=1e-10)
    # Fourier quadrature of d(lambda)/dtheta contains both parities.
    assert np.max(np.abs(la[0][:, 0])) > 0.01
    ev = gvec.evaluate_sfl(
        st,
        "mod_B",
        "X1",
        rho=rho,
        theta=128,
        zeta=4,
        sfl="boozer",
        MNfactor=1,
        sincos="sincos",
        radial_derivative=False,
    )
    value = (ev.mod_B * ev.X1).transpose("rad", "pol", "tor").values
    np.testing.assert_allclose(
        value, value[:, :1, :1] * np.ones_like(value), rtol=1e-9, atol=1e-10
    )
