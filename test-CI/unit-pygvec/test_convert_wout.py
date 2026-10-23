from pathlib import Path

import netCDF4
import numpy as np
import pytest

import gvec
from gvec.scripts.convert_wout import convert_vmec_wout


@pytest.mark.parametrize("signgs", ["neg", "pos"])
@pytest.mark.parametrize("asymmetric", ["F", "T"])
def test_converted_wout_state_can_be_reloaded(tmp_path, signgs, asymmetric):
    source = (
        Path(__file__).parents[1] / "data" / f"wout_test-signgs-{signgs}-asym{asymmetric}.nc"
    )
    convert_vmec_wout(source, tmp_path)
    (statefile,) = tmp_path.glob("*State*.dat")
    converted = gvec.load_state(tmp_path / "parameter.ini", statefile)
    try:
        position = np.asarray(
            converted.evaluate("pos", rho=1.0, theta=0.0, zeta=0.0).pos
        ).reshape(3)
        # At the boundary and zero angles all sine modes vanish. The original
        # VMEC Fourier coefficients give an independent geometric readback.
        with netCDF4.Dataset(source) as wout:
            radius = np.sum(wout["rmnc"][-1, :])
            height = np.sum(wout["zmnc"][-1, :]) if "zmnc" in wout.variables else 0.0
            assert converted.nfp == int(wout["nfp"][:])
        np.testing.assert_allclose(position, [radius, 0.0, height], rtol=1e-10, atol=1e-12)
    finally:
        converted.unbind()
