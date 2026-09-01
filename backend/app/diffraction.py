"""Diffraction physics for the rainbow accent — baked to a lookup table.

WHY THIS EXISTS. The preview's spectral sheen used to be a hand-tuned hue ramp
inside ``plate.frag``: it hardcoded its own 45 deg direction and a magic
``hue = fract(proj * 1.6 + ...)``, and it was never given the fabricated
grating's period at all. Changing the fab accent grating from 4.4 um to any
other pitch produced a BIT-IDENTICAL preview — the on-screen rainbow could not
tell you anything true about the mask. This module replaces that with real
physics, computed here (where it is testable against closed forms) and shipped
to the shader as DATA, so the GLSL has no physics left to get wrong.

WHY A LOOKUP TABLE. Diffraction is wave optics; a rasterizer solves geometric
optics, so unlike the parallax family (which emerges from two real planes and
perspective) a rainbow can never emerge from scene geometry. The real-time
literature converges on precomputing a diffraction LUT and sampling it in the
shader — Stam's Diffraction Shaders (1999) for the general BRDF, Toisoul &
Ghosh (ACM TOG 2017) for the practical "single precomputed lookup table"
formulation, Dhillon et al. for the precomputed-table reformulation. Those
methods spend their machinery ESTIMATING an unknown microstructure (windowed
FFTs of a measured height field). We are in the easy case: we designed the
grating, so its pitch, duty and orientation are known exactly and it is a
binary lamellar amplitude grating whose Fourier series is closed-form. No FFT
is needed and the table collapses to ONE DIMENSION.

THE PARAMETERIZATION (the reason this is cheap). For a grating with unit
in-plane grating vector ``g`` and period ``d``, momentum conservation along the
surface gives, for view direction V and light direction L (both unit, pointing
away from the surface):

    d * ( V.g + L.g ) = m * lambda                                        (1)

Define the scalar OPTICAL PATH TERM ``u = d * (V.g + L.g)`` (um). Then order m
lands in the eye at wavelength ``lambda_m = u / m``. Every geometric quantity
-- pitch, orientation, view, light -- enters ONLY through u, and the order
efficiencies depend only on the duty cycle. So a single 1-D table indexed by u
serves EVERY pitch: the shader computes u from the real fabricated period and
does one texture fetch. (This is the same "optical path difference as the
common index" trick the texture-based iridescence literature uses.)

THE SPECTRUM AT u. For a lamellar amplitude grating of duty c the order
amplitudes are the Fourier coefficients of a square wave, so

    eta_m = ( c * sinc(m*c) )^2 ,   sinc(x) = sin(pi x)/(pi x)            (2)

which correctly kills the even orders at c = 0.5 and falls off as 1/m^2. The
angular dispersion of order m is dtheta/dlambda = m / (d cos theta), so a fixed
solid angle collects a spectral bandwidth proportional to 1/m -- the standard
dispersion Jacobian. Summing the visible orders at u:

    S(u) = sum_m  (eta_m / m) * CMF(lambda_m) * I(lambda_m)               (3)

with CMF the CIE 1931 2-deg colour matching functions and I the illuminant
(equal-energy here -- the neutral default; the shader tints by the scene
illuminant afterwards). XYZ is then converted to LINEAR sRGB.

VALIDITY. Scalar (Kirchhoff) theory is appropriate while d >> lambda; the
accent grating is 4.4 um against ~0.55 um light, i.e. d/lambda ~ 8, comfortably
in range. Rigorous coupled-wave (RCWA) would only be needed near d ~ lambda,
which is not where this grating lives.

Everything here is pure NumPy and unit-tested against closed-form anchors --
see ``tests/test_diffraction.py``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Visible band actually integrated. Outside this the CMFs are ~0 anyway.
VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 730.0

# Default table extent. u = d*(V.g + L.g) and |V.g + L.g| <= 2, so a 4.4 um
# grating tops out near 8.8 um; 10 um covers it with headroom. 1024 entries
# gives ~10 nm of wavelength resolution at first order, which linear texture
# filtering smooths into a continuous sweep.
DEFAULT_U_MAX_UM = 10.0
DEFAULT_SIZE = 1024
# Orders summed. At the top of the u range the visible band is fed by m ~ 13,
# so 16 covers everything the table can show.
DEFAULT_ORDERS = 16


# --------------------------------------------------------------------------- #
#  CIE 1931 2-degree colour matching functions                                 #
# --------------------------------------------------------------------------- #
# Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ Colour
# Matching Functions" (JCGT 2013): multi-lobe piecewise-Gaussian fits, max error
# ~1% of peak. Used instead of a 471-row table because a closed form cannot be
# mistyped in a way the tests below do not catch -- ybar is the photopic
# luminosity curve, so ybar(555 nm) == 1.0 pins it, and the equal-energy white
# point pins the xbar/zbar balance.


def _piecewise_gauss(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    """Gaussian with a different sigma below and above the peak."""
    sigma = np.where(x < mu, s1, s2)
    t = (x - mu) / sigma
    return np.exp(-0.5 * t * t)


def cie_xyz(nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2-deg (xbar, ybar, zbar) for wavelengths in nm. Shape (..., 3)."""
    nm = np.asarray(nm, dtype=float)
    x = (
        1.056 * _piecewise_gauss(nm, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gauss(nm, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gauss(nm, 501.1, 20.4, 26.2)
    )
    y = (
        0.821 * _piecewise_gauss(nm, 568.8, 46.9, 40.5)
        + 0.286 * _piecewise_gauss(nm, 530.9, 16.3, 31.1)
    )
    z = (
        1.217 * _piecewise_gauss(nm, 437.0, 11.8, 36.0)
        + 0.681 * _piecewise_gauss(nm, 459.0, 26.0, 13.8)
    )
    return np.stack([x, y, z], axis=-1)


# CIE XYZ -> linear sRGB (IEC 61966-2-1 primaries, D65 white).
XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=float,
)


def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    """(..., 3) XYZ -> (..., 3) LINEAR sRGB, negatives clipped to 0.

    Clipping (rather than gamut-mapping) is deliberate: a single diffracted
    wavelength is monochromatic and sits far outside the sRGB triangle, so some
    negative component is expected. The clip desaturates it to the nearest
    representable colour, which is what any RGB renderer must do.
    """
    rgb = xyz @ XYZ_TO_LINEAR_SRGB.T
    return np.clip(rgb, 0.0, None)


# --------------------------------------------------------------------------- #
#  lamellar grating orders                                                     #
# --------------------------------------------------------------------------- #

def _sinc(x: np.ndarray | float) -> np.ndarray:
    """Normalized sinc, sin(pi x)/(pi x), with sinc(0) = 1."""
    return np.sinc(x)  # numpy's sinc IS the normalized one


def lamellar_order_efficiency(m: int, duty: float) -> float:
    """Diffraction efficiency of order ``m`` for a binary amplitude grating.

    The Fourier coefficient of a duty-``duty`` square wave at order m is
    ``duty * sinc(m*duty)``; the efficiency is its square (equation 2 in the
    module docstring). At duty 0.5 every EVEN order vanishes -- the classic
    signature of a 50% mark-space grating -- and the odd orders fall as 1/m^2.
    """
    if m == 0:
        return float(duty * duty)
    return float((duty * _sinc(m * duty)) ** 2)


# --------------------------------------------------------------------------- #
#  the table                                                                   #
# --------------------------------------------------------------------------- #

def bake_lut(
    *,
    duty: float = 0.5,
    u_max_um: float = DEFAULT_U_MAX_UM,
    size: int = DEFAULT_SIZE,
    orders: int = DEFAULT_ORDERS,
) -> np.ndarray:
    """Bake the 1-D diffraction colour table: ``(size, 3)`` LINEAR sRGB.

    Entry i is the colour seen when ``u = d*(V.g + L.g)`` equals
    ``i/(size-1) * u_max_um`` (um) -- see the module docstring. Normalized so
    the brightest entry is 1.0, because the absolute scale belongs to the
    renderer's exposure, not to the physics of which colour appears where.
    """
    if size < 2:
        raise ValueError(f"size must be >= 2 (got {size})")
    if not (0.0 < duty < 1.0):
        raise ValueError(f"duty must be in (0, 1) (got {duty})")

    u = np.linspace(0.0, u_max_um, size)          # um
    out = np.zeros((size, 3), dtype=float)

    for m in range(1, orders + 1):
        eta = lamellar_order_efficiency(m, duty)
        if eta <= 0.0:
            continue                               # even orders at duty 0.5
        lam_nm = (u / m) * 1000.0                  # lambda_m = u/m, um -> nm
        visible = (lam_nm >= VISIBLE_MIN_NM) & (lam_nm <= VISIBLE_MAX_NM)
        if not visible.any():
            continue
        xyz = cie_xyz(lam_nm)
        # Equal-energy illuminant, so I(lambda) is a constant and drops out of
        # the shape; the 1/m is the dispersion Jacobian (module docstring).
        out[visible] += (eta / m) * xyz[visible]

    rgb = xyz_to_linear_srgb(out)
    peak = float(rgb.max())
    if peak > 0.0:
        rgb /= peak
    return rgb


def lut_payload(
    *,
    duty: float = 0.5,
    u_max_um: float = DEFAULT_U_MAX_UM,
    size: int = DEFAULT_SIZE,
    orders: int = DEFAULT_ORDERS,
) -> dict[str, Any]:
    """JSON-able LUT for the renderer: linear-sRGB triples plus its index scale.

    Values are LINEAR (not sRGB-encoded) so the frontend can upload them to a
    float texture and the shader can add them straight into its linear-light
    accumulation -- no decode step, no 8-bit quantization of a gradient whose
    whole job is to sweep smoothly.
    """
    rgb = bake_lut(duty=duty, u_max_um=u_max_um, size=size, orders=orders)
    return {
        "kind": "diffraction_lut",
        "duty": duty,
        "u_max_um": u_max_um,
        "size": size,
        "orders": orders,
        "color_space": "linear-srgb",
        # Flat list keeps the payload compact and the upload a single memcpy.
        "rgb": [round(float(v), 5) for v in rgb.reshape(-1)],
    }
