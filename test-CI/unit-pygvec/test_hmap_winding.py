"""Check the winding-dependent G-frame rotation against an analytic torus knot."""

from pathlib import Path

import numpy as np
import helpers


def test_cinquefoil_frame(tmp_path, util):
    from gvec import State

    source = Path(__file__).parents[1] / "examples" / "cinquefoil" / "parameter.ini"
    paramfile = tmp_path / "parameter.ini"
    util.adapt_parameter_file(source, paramfile, testlevel=-1, MaxIter=1)
    helpers.generate_cinquefoil(tmp_path / "cinquefoil.nc")
    state = State(paramfile)
    zeta = np.linspace(0, 2 * np.pi, 71, endpoint=False)
    zero = np.zeros_like(zeta)
    pos, normal, binormal, tangent = state.evaluate_hmap_only(zero, zero, zeta)
    *_, second = state.evaluate_hmap_derivs(zero, zero, zeta)

    expected_pos = np.array(
        [
            np.cos(2 * zeta) + 0.1 * np.cos(3 * zeta),
            np.sin(2 * zeta) - 0.1 * np.sin(3 * zeta),
            0.2 * np.sin(5 * zeta),
        ]
    )
    expected_tangent = np.array(
        [
            -2 * np.sin(2 * zeta) - 0.3 * np.sin(3 * zeta),
            2 * np.cos(2 * zeta) - 0.3 * np.cos(3 * zeta),
            np.cos(5 * zeta),
        ]
    )
    expected_second = np.array(
        [
            -4 * np.cos(2 * zeta) - 0.9 * np.cos(3 * zeta),
            -4 * np.sin(2 * zeta) + 0.9 * np.sin(3 * zeta),
            -5 * np.sin(5 * zeta),
        ]
    )
    for order, (actual, expected) in enumerate(
        [
            (pos, expected_pos),
            (tangent, expected_tangent),
            (second, expected_second),
        ]
    ):
        # Differentiation amplifies DFT roundoff by the highest retained frequency.
        max_frequency = 5 * ((21 - 1) // 2)
        atol = 64 * np.finfo(float).eps * max_frequency**order
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)

    # Also exercise the first and second derivatives of the normal and binormal.
    x1, x2, step = zero + 0.03, zero - 0.02, 1e-5
    pos, _, _, tangent = state.evaluate_hmap_only(x1, x2, zeta)
    *_, second = state.evaluate_hmap_derivs(x1, x2, zeta)
    pos_plus, _, _, tangent_plus = state.evaluate_hmap_only(x1, x2, zeta + step)
    pos_minus, _, _, tangent_minus = state.evaluate_hmap_only(x1, x2, zeta - step)
    np.testing.assert_allclose(tangent, (pos_plus - pos_minus) / (2 * step), atol=1e-8)
    np.testing.assert_allclose(second, (tangent_plus - tangent_minus) / (2 * step), atol=1e-7)
