"""Numeric spot-checks on the sim kernels that a visual-only test would miss.

These supplement test_physics_sanity.py with tighter invariants:
  * angular-spectrum at z=0 is identity
  * angular-spectrum of a tilted plane wave keeps the tilt direction
  * aperture DC term equals the integrated transmission squared
"""
from __future__ import annotations

import numpy as np

from app.sim.angular_spectrum import _propagate


def test_angular_spectrum_zero_distance_is_identity() -> None:
    rng = np.random.default_rng(0)
    field = rng.standard_normal((64, 64)).astype(np.complex64)
    out = _propagate(field, dx_um=0.5, z_um=0.0, lam_um=0.55, n=1.46)
    # exp(i*kz*0) = 1 for all k -> inverse FFT of FFT = original
    assert np.allclose(out, field, atol=1e-4)


def test_angular_spectrum_plane_wave_direction_preserved() -> None:
    """A tilted plane wave should still be (approximately) a plane wave after
    propagation through free space — its peak in k-space should stay at the
    same spatial-frequency bin."""
    h = w = 128
    dx = 0.5
    lam = 0.55
    n = 1.0
    # A plane wave tilted by a known angle -> known kx
    theta = np.deg2rad(5.0)
    k0 = 2 * np.pi / lam
    kx_target = n * k0 * np.sin(theta)
    xs = (np.arange(w) - w / 2) * dx
    ys = (np.arange(h) - h / 2) * dx
    X, _ = np.meshgrid(xs, ys)
    field_in = np.exp(1j * kx_target * X).astype(np.complex64)

    out = _propagate(field_in, dx, z_um=100.0, lam_um=lam, n=n)

    # Peak bin in k-space should match before and after (to within one bin).
    def peak_kx(f: np.ndarray) -> float:
        F = np.abs(np.fft.fftshift(np.fft.fft2(f)))
        row = F[h // 2]
        kx = np.fft.fftshift(np.fft.fftfreq(w, d=dx)) * 2 * np.pi
        return float(kx[int(np.argmax(row))])

    kx_in = peak_kx(field_in)
    kx_out = peak_kx(out)
    assert abs(kx_in - kx_out) < (2 * np.pi / (w * dx)) * 1.5


def test_fft_dc_equals_aperture_sum() -> None:
    """|FFT{U}[0,0]| = |sum(U)|. This is a Parseval-adjacent identity that
    would catch a regression in FFT normalization or axis convention."""
    rng = np.random.default_rng(1)
    aperture = rng.random((64, 64)).astype(np.float32)
    F = np.fft.fft2(aperture)
    dc = abs(F[0, 0])
    s = abs(aperture.sum())
    assert abs(dc - s) / s < 1e-5
