import pytest

try:
    import numpy as np

    import gvec
except ImportError:
    pass  # tests will be skipped via the check_imports fixture


@pytest.fixture
def parameters():
    return {
        "nfp": 1,
        "sgrid_nElems": 2,
        "X1X2_deg": 3,
        "LA_deg": 3,
        "init_average_axis": True,
        "init_LA": False,
        "X1_b_cos": {(0, 0): 5.0, (1, 0): 0.9},
        "iota": {"type": "polynomial", "coefs": [0.4]},
        "pres": {"type": "polynomial", "coefs": [0.0]},
    }


@pytest.mark.parametrize("minor_radius", [0.9, 0.0], ids=["line", "point"])
def test_init_average_axis_zero_area(parameters, minor_radius, tmp_path):
    parameters["X1_b_cos"][(1, 0)] = minor_radius
    state = gvec.State.new(parameters, rundir=tmp_path)
    try:
        with pytest.raises(
            gvec.errors.InitializationError,
            match="init_average_axis: boundary has zero signed area",
        ):
            state.bind(force=True)
    finally:
        if gvec.core.state._binding.bound_state is state:
            state.unbind()


def test_init_average_axis_ellipse(parameters, tmp_path):
    parameters["X2_b_sin"] = {(1, 0): 0.6}
    state = gvec.State.new(parameters, rundir=tmp_path)
    try:
        for quantity, center in [("X1", 5.0), ("X2", 0.0)]:
            values = state.evaluate_base_tens(
                quantity, None, [0.0, 0.5, 1.0], [0.0, 1.0], [0.0]
            )
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values[0], center, atol=1e-12)
    finally:
        if gvec.core.state._binding.bound_state is state:
            state.unbind()
