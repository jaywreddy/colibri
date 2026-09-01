"""Diffraction LUT physics — pinned against closed-form anchors.

The whole point of moving this out of GLSL is that it becomes checkable, so
these tests assert PHYSICS, not just shape: the photopic curve's peak, the
equal-energy white point, the square-wave order series, the grating equation
itself, and the monotone wavelength sweep the table has to produce for the
preview to shimmer correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.diffraction import (
    DEFAULT_SIZE,
    DEFAULT_U_MAX_UM,
    bake_lut,
    cie_xyz,
    lamellar_order_efficiency,
    lut_payload,
    xyz_to_linear_srgb,
)


# --- CIE fit ----------------------------------------------------------------


def test_ybar_is_the_photopic_curve():
    """ybar IS the luminous efficiency function: it peaks at 555 nm at 1.0.
    This single anchor catches a mistyped coefficient in the y lobes."""
    nm = np.arange(380.0, 731.0, 1.0)
    y = cie_xyz(nm)[:, 1]
    assert y.max() == pytest.approx(1.0, abs=0.02)
    assert nm[int(y.argmax())] == pytest.approx(555.0, abs=3.0)


def test_xbar_has_its_two_lobes_and_zbar_one():
    nm = np.arange(380.0, 731.0, 1.0)
    xyz = cie_xyz(nm)
    x, z = xyz[:, 0], xyz[:, 2]
    # xbar is bimodal: a small blue lobe near 442 and the main red lobe near 600.
    assert nm[int(x.argmax())] == pytest.approx(600.0, abs=6.0)
    blue = x[(nm > 420) & (nm < 470)]
    assert blue.max() > 0.3
    # zbar is a single blue lobe near 445.
    assert nm[int(z.argmax())] == pytest.approx(445.0, abs=8.0)
    # Everything decays at the band edges.
    assert xyz[0].max() < 0.05 and xyz[-1].max() < 0.05


def test_equal_energy_spectrum_is_neutral():
    """An equal-energy spectrum must land on the achromatic point x=y=1/3 —
    the check that pins xbar/ybar/zbar RELATIVE to each other."""
    nm = np.arange(380.0, 731.0, 1.0)
    xyz = cie_xyz(nm).sum(axis=0)
    s = xyz.sum()
    assert xyz[0] / s == pytest.approx(1 / 3, abs=0.02)
    assert xyz[1] / s == pytest.approx(1 / 3, abs=0.02)


def test_xyz_to_srgb_maps_equal_energy_to_near_neutral_rgb():
    nm = np.arange(380.0, 731.0, 1.0)
    rgb = xyz_to_linear_srgb(cie_xyz(nm).sum(axis=0))
    assert rgb.min() > 0.0
    # Equal-energy is not D65, so allow a modest tint but require no wild cast.
    assert rgb.max() / rgb.min() < 1.6


# --- grating orders ---------------------------------------------------------


def test_fifty_percent_duty_kills_even_orders():
    """The signature of a 50% mark-space grating."""
    for m in (2, 4, 6, 8):
        assert lamellar_order_efficiency(m, 0.5) == pytest.approx(0.0, abs=1e-12)
    for m in (1, 3, 5):
        assert lamellar_order_efficiency(m, 0.5) > 0.0


def test_order_efficiencies_match_the_square_wave_series():
    """eta_m = (c*sinc(m*c))^2 — closed form, checked at duty 0.5 where the odd
    orders are (1/(m*pi))^2."""
    for m in (1, 3, 5, 7):
        expected = (1.0 / (m * np.pi)) ** 2
        assert lamellar_order_efficiency(m, 0.5) == pytest.approx(expected, rel=1e-9)
    # ...and they fall off as 1/m^2.
    assert lamellar_order_efficiency(3, 0.5) == pytest.approx(
        lamellar_order_efficiency(1, 0.5) / 9.0, rel=1e-9
    )


def test_non_half_duty_revives_even_orders():
    assert lamellar_order_efficiency(2, 0.25) > 0.0


# --- the table --------------------------------------------------------------


def test_lut_shape_and_range():
    lut = bake_lut()
    assert lut.shape == (DEFAULT_SIZE, 3)
    assert lut.min() >= 0.0
    assert lut.max() == pytest.approx(1.0)


def test_lut_is_black_below_the_first_visible_wavelength():
    """u < 380 nm cannot put ANY order in the visible band (order m only makes
    lambda SHORTER), so the table must start black — the rainbow appears only
    once the geometry actually satisfies the grating equation."""
    lut = bake_lut()
    u = np.linspace(0.0, DEFAULT_U_MAX_UM, DEFAULT_SIZE)
    dark = lut[u < 0.37]
    assert dark.max() == pytest.approx(0.0, abs=1e-9)


def test_first_order_sweeps_blue_to_red_with_u():
    """Between u = 0.40 and 0.70 um only m=1 is visible, so the table must run
    blue -> red exactly as the grating equation says (lambda_1 = u)."""
    lut = bake_lut()
    u = np.linspace(0.0, DEFAULT_U_MAX_UM, DEFAULT_SIZE)

    def at(target_um: float) -> np.ndarray:
        return lut[int(np.argmin(np.abs(u - target_um)))]

    blue, green, red = at(0.45), at(0.53), at(0.65)
    assert blue[2] > blue[0], "450 nm must be blue-dominant"
    assert green[1] > green[0] and green[1] > green[2], "530 nm must be green-dominant"
    assert red[0] > red[2], "650 nm must be red-dominant"


def test_grating_equation_places_the_peak():
    """The physical anchor: with the fab accent grating (4.4 um) the geometric
    term (V.g + L.g) at which 550 nm green appears is lambda/d = 0.125. The
    table must be green there — i.e. the preview's colour is tied to the REAL
    fabricated pitch, which is the decoupling this whole module fixes."""
    d_um = 4.4
    lut = bake_lut()
    u_axis = np.linspace(0.0, DEFAULT_U_MAX_UM, DEFAULT_SIZE)
    geom = 0.550 / d_um                       # (V.g + L.g) for 550 nm at m=1
    u = d_um * geom                           # -> 0.550 um by construction
    rgb = lut[int(np.argmin(np.abs(u_axis - u)))]
    assert rgb[1] >= rgb[0] and rgb[1] > rgb[2]


def test_high_u_washes_toward_pastel_and_is_far_dimmer():
    """Far from specular the visible band is fed only by HIGH orders, whose
    energy falls as eta_m/m ~ 1/m^3 — so the light there is real but ~1e-3 of
    the first-order flash, and desaturated because several orders overlap.
    (Measured: order 13 is ~2200x weaker than order 1, exactly 13^3.) This is
    why the accent reads as a sharp spectral flash through the first-order
    angle rather than a broad rainbow wash — the real behaviour of a 4.4 um
    grating, and the thing the old tuned hue ramp got wrong."""
    lut = bake_lut()
    u = np.linspace(0.0, DEFAULT_U_MAX_UM, DEFAULT_SIZE)
    high = lut[u > 6.0]
    assert high.max() > 0.0
    assert high.max() < 0.01
    sat = (high.max(axis=1) - high.min(axis=1)) / np.maximum(high.max(axis=1), 1e-6)
    low = lut[(u > 0.40) & (u < 0.70)]
    sat_low = (low.max(axis=1) - low.min(axis=1)) / np.maximum(low.max(axis=1), 1e-6)
    assert sat.mean() < sat_low.mean()


def test_duty_changes_the_table():
    """A different fabricated duty must produce a different table (even orders
    return) — proving duty is a real input, not decoration."""
    a = bake_lut(duty=0.5)
    b = bake_lut(duty=0.25)
    assert not np.allclose(a, b)


def test_payload_is_json_shaped_and_linear():
    p = lut_payload(size=64)
    assert p["size"] == 64
    assert p["color_space"] == "linear-srgb"
    assert len(p["rgb"]) == 64 * 3
    assert all(isinstance(v, float) for v in p["rgb"][:12])
    assert max(p["rgb"]) == pytest.approx(1.0, abs=1e-4)
