"""Physics sanity checks on the generator math and Tier 2/3 simulators.

These are not ground-truth verifications — they check that simple closed-form
predictions match the code path within a loose tolerance:

  * Cafetero iridescence first-order diffraction angle: theta = asin(lambda / period)
  * Emerald-facet moiré magnification: m = d_front / (d_front - d_back)
  * Tairona-Talbot self-imaging distance: z_T = 2 * period^2 * n / lambda
  * Fraunhofer FFT of a single grating produces discrete orders at expected
    angular positions (peaks in |FFT|).
"""
from __future__ import annotations

import math

import numpy as np

import app.patterns.artistic  # noqa: F401
from app.patterns.base import registry


def test_cafetero_iridescence_first_order_matches_closed_form():
    cls = registry["cafetero-iridescence"]
    period = 4.0
    gp = cls.generate(period_um=period, duty=0.5)
    expected = math.degrees(math.asin(0.55 / period))
    got = float(gp.extra["first_order_green_deg"])
    assert abs(got - expected) < 1e-3, f"{got} vs {expected}"


def test_emerald_facet_moire_reports_expected_magnification():
    cls = registry["emerald-facet-moire"]
    d_front = 40.0
    d_back = 41.0
    gp = cls.generate(period_front_um=d_front, period_back_um=d_back)
    # Magnification formula for double-sided magnifier: m = d_front / (d_front - d_back)
    expected_abs = abs(d_front / (d_front - d_back))
    got = float(gp.extra["expected_magnification"])
    assert abs(abs(got) - expected_abs) / expected_abs < 0.05


def test_talbot_distance_scaling_with_index():
    cls = registry["tairona-talbot"]
    period = 20.0
    lam = 0.55
    gp = cls.generate(period_um=period, wavelength_um=lam)
    n = 1.46
    expected_zT = 2.0 * period**2 * n / lam
    got = float(gp.extra["talbot_distance_um"])
    assert abs(got - expected_zT) / expected_zT < 0.01


def test_fft_of_grating_shows_peaks_at_expected_orders():
    """Compute FFT of a 4 μm-period grating at 1 μm pitch and check that
    the dominant non-DC peak falls at the first-order k vector."""
    period_um = 4.0
    pitch_um = 0.5
    n_samples = 2048
    # Build a 1-D square wave of the grating
    x = np.arange(n_samples) * pitch_um
    mask = ((x % period_um) < (period_um * 0.5)).astype(np.float32)
    mask -= mask.mean()  # remove DC so the first order dominates
    F = np.abs(np.fft.fftshift(np.fft.fft(mask)))
    kx = np.fft.fftshift(np.fft.fftfreq(n_samples, d=pitch_um))  # cycles/μm
    # Predicted first-order frequency
    predicted = 1.0 / period_um  # cycles/μm
    # Find peak within the positive-frequency half, excluding DC
    mask_pos = kx > 0.01
    idx = np.argmax(F * mask_pos)
    got = abs(kx[idx])
    assert abs(got - predicted) / predicted < 0.01, f"{got} vs {predicted}"


def test_angular_spectrum_preserves_energy_in_free_space(tmp_path):
    """Free-space propagation via angular spectrum should conserve total
    intensity to within a few percent (no absorbing medium, no aperture)."""
    from app.sim.angular_spectrum import _propagate

    n_px = 128
    dx = 1.0  # μm
    lam = 0.55
    field = np.zeros((n_px, n_px), dtype=np.complex64)
    # Gaussian beam centered
    ys, xs = np.indices(field.shape)
    cy, cx = n_px / 2, n_px / 2
    r2 = (ys - cy) ** 2 + (xs - cx) ** 2
    field[:] = np.exp(-r2 / (2 * 10.0**2)).astype(np.complex64)

    e0 = float(np.sum(np.abs(field) ** 2))
    out = _propagate(field, dx, z_um=100.0, lam_um=lam, n=1.0)
    e1 = float(np.sum(np.abs(out) ** 2))
    # Allow 5 percent drift (numerical edge effects)
    assert abs(e1 - e0) / e0 < 0.05, f"energy drifted {e0} -> {e1}"
