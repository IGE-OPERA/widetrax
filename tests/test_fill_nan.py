"""fill_nan: dual-path pyinterp behaviour on synthetic swaths."""
import numpy as np
import pytest

pytest.importorskip("pyinterp")

import xarray as xr

from widetrax.DataPreprocessing import (
    _gauss_seidel_fill,
    _gauss_seidel_inplace_api,
    fill_nan,
)


def _swath_dataset(lon_1d, lat_1d, values, varname="ssha"):
    ny, nx = values.shape
    lon2 = np.tile(lon_1d[None, :], (ny, 1))
    lat2 = np.repeat(lat_1d[:, None], nx, axis=1)
    return xr.Dataset(
        {varname: (("num_lines", "num_pixels"), values)},
        coords={
            "latitude": (("num_lines", "num_pixels"), lat2),
            "longitude": (("num_lines", "num_pixels"), lon2),
        },
    )


def test_api_detection_returns_bool():
    # On the installed pyinterp either flavour is fine; it must just be a bool.
    assert isinstance(_gauss_seidel_inplace_api(), bool)


@pytest.mark.parametrize("descending", [False, True])
def test_linear_field_recovered_exactly(descending):
    # Gauss-Seidel recovers linear fields exactly. A descending x axis forces
    # the column permutation (sorted axis -> fill -> restore) through its
    # paces; output must stay aligned with the caller's coordinates.
    # nx < 34 so the Nadir middle-column NaN-ing is not triggered here.
    ny, nx = 30, 30
    lon = np.linspace(300.0, 310.0, nx)
    if descending:
        lon = lon[::-1]
    lat = np.linspace(-40.0, -30.0, ny)
    truth = 0.1 * np.repeat(lat[:, None], nx, axis=1) + 0.01 * np.tile(lon[None, :], (ny, 1))

    values = truth.copy()
    values[10:20, 15:25] = np.nan  # interior hole

    ds = _swath_dataset(lon, lat, values)
    has_converged, filled = fill_nan({0: ds})

    assert set(filled.keys()) == {0}
    out = filled[0]["ssha"].values
    assert np.isfinite(out).all()
    # fill_nan runs at the default epsilon=1e-4: the linear structure comes
    # back to within that tolerance everywhere (max error ~4e-4 in the hole).
    np.testing.assert_allclose(out, truth, atol=1e-3)
    assert has_converged is True


def test_all_nan_column_restored():
    # Fully-NaN columns must come back NaN (historical contract), while
    # recoverable holes next to them are filled. This includes the two
    # Nadir middle columns (33:34) that fill_nan NaNs on purpose when the
    # swath is at least 34 pixels wide.
    ny, nx = 20, 50
    lon = np.linspace(300.0, 310.0, nx)
    lat = np.linspace(-40.0, -30.0, ny)
    truth = 0.1 * lat[:, None] + 0.01 * lon[None, :]

    values = truth.copy()
    values[:, 5] = np.nan          # fully-NaN column -> must stay NaN
    values[8:12, 20:30] = np.nan   # recoverable hole

    ds = _swath_dataset(lon, lat, values)
    has_converged, filled = fill_nan({0: ds})

    out = filled[0]["ssha"].values
    assert np.isnan(out[:, 5]).all()      # explicitly all-NaN column
    assert np.isnan(out[:, 33:35]).all() # NaN'd Nadir columns -> all-NaN -> restored

    keep = np.ones_like(truth, dtype=bool)
    keep[:, 5] = False
    keep[:, 33:35] = False
    np.testing.assert_allclose(out[keep], truth[keep], atol=1e-3)


def test_gauss_seidel_fill_direct_descending_x():
    # The helper must realign columns: a descending x axis with a linear field
    # is recovered AND returned in the original column order.
    ny, nx = 12, 15
    lon = np.linspace(310.0, 300.0, nx)  # descending
    lat = np.linspace(-40.0, -30.0, ny)
    truth = 0.5 * np.repeat(lat[:, None], nx, axis=1) + 0.02 * np.tile(lon[None, :], (ny, 1))

    values = truth.copy()
    values[4:8, 6:10] = np.nan

    lon2 = np.tile(lon[None, :], (ny, 1))
    lat2 = np.repeat(lat[:, None], nx, axis=1)

    # tight epsilon -> linear field recovered to (near) machine precision
    has_converged, filled = _gauss_seidel_fill(values, lon2, lat2, epsilon=1e-8)
    np.testing.assert_allclose(filled, truth, atol=1e-6)
    assert has_converged is True
