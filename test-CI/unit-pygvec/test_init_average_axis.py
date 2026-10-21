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
            match="init_average_axis: boundary area is zero or not resolvable",
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


def test_init_average_axis_near_collapsed(parameters, tmp_path):
    # The vertical offset makes the shoelace terms cancel at roundoff scale.
    parameters.update(
        X1_mn_max=(1, 0),
        X2_mn_max=(1, 0),
        LA_mn_max=(1, 0),
        fac_nyq=8,
        X2_sin_cos="_sincos_",
        X2_b_cos={(0, 0): 2.0},
        X2_b_sin={(1, 0): 1e-14},
    )
    state = gvec.State.new(parameters, rundir=tmp_path)
    try:
        with pytest.raises(
            gvec.errors.InitializationError,
            match="init_average_axis: boundary area is zero or not resolvable.*IntInfo= *1",
        ):
            state.bind(force=True)
    finally:
        if gvec.core.state._binding.bound_state is state:
            state.unbind()


@pytest.mark.parametrize(
    "major_radius, radial_radius, vertical_radius, vertical_center",
    [
        (5.0, 0.9, 0.9e-6, 0.0),
        (5.0, 0.9, 1e-6, 2.0),
        (5.0, 0.9e-3, 0.9, 0.0),
        (5.0, 1e-6, 0.6e-6, 0.0),
        (1e6, 0.9, 0.6, 0.0),
        (5e-9, 0.9e-9, 0.6e-9, 0.0),
        (5e9, 0.9e9, 0.6e9, 0.0),
    ],
    ids=[
        "thin",
        "shifted-thin",
        "elongated",
        "small-minor-radius",
        "large-major-radius",
        "small-scale",
        "large-scale",
    ],
)
def test_init_average_axis_shaped_ellipse(
    parameters, tmp_path, major_radius, radial_radius, vertical_radius, vertical_center
):
    parameters["X1_b_cos"] = {(0, 0): major_radius, (1, 0): radial_radius}
    parameters["X2_sin_cos"] = "_sincos_"
    parameters["X2_b_cos"] = {(0, 0): vertical_center}
    parameters["X2_b_sin"] = {(1, 0): vertical_radius}
    # Centroid roundoff scales with the coordinates and vertical elongation:
    # the vertical moment is O(R*b**2), while the area is O(a*b).
    atol = (
        64
        * np.finfo(float).eps
        * max(abs(major_radius), radial_radius, vertical_radius, abs(vertical_center))
        * max(1.0, vertical_radius / radial_radius)
    )
    state = gvec.State.new(parameters, rundir=tmp_path)
    try:
        for quantity, center in [("X1", major_radius), ("X2", vertical_center)]:
            values = state.evaluate_base_tens(
                quantity, None, [0.0, 0.5, 1.0], [0.0, 1.0], [0.0]
            )
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values[0], center, rtol=1e-8, atol=atol)
    finally:
        if gvec.core.state._binding.bound_state is state:
            state.unbind()


def test_init_average_axis_clockwise_ellipse(parameters, tmp_path):
    parameters["X2_b_sin"] = {(1, 0): -0.6}
    state = gvec.State.new(parameters, rundir=tmp_path)
    try:
        # The area guard must accept either sign. The later Jacobian check still
        # rejects a left-handed coordinate system, after constructing the axis.
        with pytest.raises(gvec.errors.InitializationError, match="LEFT-HANDED COORDINATES"):
            state.bind(force=True)
    finally:
        if gvec.core.state._binding.bound_state is state:
            state.unbind()
