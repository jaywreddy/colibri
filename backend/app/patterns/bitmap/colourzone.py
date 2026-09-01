"""Spectral colour on top of a halftone: sub-grate the gold bands, by region.

A halftone line screen renders tone by varying the WIDTH of a gold band. Fill
that band with a fine grating instead of solid metal and it diffracts, so the
region reads as spectral colour rather than gold. Doing that per REGION is how
you colour-code a picture without turning the whole thing into a rainbow.

What actually controls the hue
------------------------------
The diffracted wavelength at a given view is ``lambda_m = d*(V.g + L.g)/m``, so:

* PERIOD ``d`` is the hue knob, and a period RATIO between two zones is
  view-INDEPENDENT — both sweep together as the piece tilts and the separation
  between them stays put. That is why zones here are specified as a
  ``period_scale`` against a base rather than as an absolute colour: you are
  choosing an offset from the neighbours, which is a thing that survives the
  viewer moving.
* ANGLE also shifts the hue, through the projection ``V.g`` — but the ratio
  between two differently-angled zones is NOT stable: turning the piece in its
  own plane swings it (1.10 at one azimuth, 0.61 at another). Use angle to
  place a zone's sweep, never to hold a relationship between zones.
* DUTY sets the ORDER EFFICIENCIES (eta_m = (c*sinc(mc))^2), so it changes
  brightness and which orders exist. It is a saturation knob, not a hue knob.
* PHASE does nothing at all. Intensity is |FT|^2 and a lateral shift only moves
  the complex phase of each order, so a shifted amplitude grating is optically
  identical. (A phase grating — an etched DEPTH — would be a real colour
  control, because the etch depth is wavelength-dependent. That needs an etch
  step this build does not have.)

The two costs
-------------
1. A gratinged band reflects roughly ``duty`` of what a solid band does, so to
   hold the same tone it must be ``1/duty`` times wider. At a 50% sub-grating
   that caps the renderable tone at 50% — colour costs about a stop.
2. A band narrower than a couple of grating periods has no spectrum to give.
   Dark tones therefore cannot carry colour at all, which is a natural and
   rather flattering behaviour: highlights take the colour, shadows stay metal.

Both are reported per zone by :func:`screen_with_colour` rather than left to be
discovered on glass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# The DRC floor. Both the gold line and the gap of the sub-grating are bounded
# by it, so a duty-0.5 grating needs a period of at least twice this.
MIN_FEATURE_UM = 2.0

# A band with fewer than this many grating periods across it does not diffract
# into a clean spectrum — it is a couple of slits, not a grating.
MIN_PERIODS_PER_BAND = 2.0


@dataclass(frozen=True)
class ColourZone:
    """One colour-coded region.

    ``period_scale`` is the hue OFFSET from the base period, and is the
    parameter to reach for when zones need to differ from each other: it is the
    only control whose relationship between zones survives a change of view.
    """

    period_scale: float = 1.0
    duty: float = 0.5
    angle_deg: float | None = None
    """None follows the screen, rotated 90 degrees — the orientation that keeps
    the sub-grating from beating with the screen's own lines."""
    hold_tone: bool = True
    """Widen the band by 1/duty so the zone keeps its tone. Off lets the zone
    simply go darker, which preserves the band geometry exactly."""
    label: str = ""

    def __post_init__(self) -> None:
        if self.period_scale <= 0.0:
            raise ValueError(f"period_scale must be > 0 (got {self.period_scale})")
        if not 0.05 <= self.duty <= 0.95:
            raise ValueError(f"duty must be 0.05..0.95 (got {self.duty})")


def _axis(shape: tuple[int, int], cell_um: float, angle_deg: float) -> np.ndarray:
    """Projection onto the grating vector, in micrometres.

    Angle 0 gives VERTICAL gold lines (constant-x stripes), matching
    ``effects.gratings`` so masks from the two compose without a hidden rotation.
    """
    h, w = shape
    a = math.radians(angle_deg)
    return (
        (np.arange(w, dtype=np.float32) * (math.cos(a) * cell_um))[None, :]
        + (np.arange(h, dtype=np.float32) * (math.sin(a) * cell_um))[:, None]
    )


def zone_period_um(base_period_um: float, zone: ColourZone) -> float:
    return float(base_period_um) * float(zone.period_scale)


def tone_cap(zone: ColourZone) -> float:
    """Highest tone this zone can render while holding its brightness."""
    return float(zone.duty) if zone.hold_tone else 1.0


def min_tone_for_colour(
    period_um: float,
    screen_period_um: float,
    *,
    duty: float = 0.5,
    hold_tone: bool = True,
    min_periods: float = MIN_PERIODS_PER_BAND,
) -> float:
    """Dimmest tone whose gold band still spans ``min_periods`` of the grating."""
    need_um = min_periods * period_um
    band_per_tone = screen_period_um * (1.0 / duty if hold_tone else 1.0)
    if band_per_tone <= 0.0:
        return 1.0
    return float(min(1.0, need_um / band_per_tone))


def screen_with_colour(
    tone: np.ndarray,
    zones: Sequence[tuple[np.ndarray, ColourZone]] = (),
    *,
    cell_um: float,
    screen_period_um: float,
    tone_steps: int,
    screen_angle_deg: float = 0.0,
    base_period_um: float = 4.4,
    min_feature_um: float = MIN_FEATURE_UM,
    min_periods: float = MIN_PERIODS_PER_BAND,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Screen ``tone`` into a binary mask, sub-grating the bands inside zones.

    ``zones`` is a sequence of ``(bool mask, ColourZone)``. Later zones win
    where they overlap, so they can be layered like paint.

    Returns ``(mask, report)``. The report carries, per zone, the realized
    period and line width, whether that clears the litho floor, the tone cap,
    and the fraction of the zone whose band was too narrow to hold a spectrum —
    the things you would otherwise only find out on glass.
    """
    tone = np.clip(np.asarray(tone, dtype=np.float32), 0.0, 1.0)
    if tone.ndim != 2:
        raise ValueError(f"tone must be 2-D, got shape {tone.shape}")
    if screen_period_um <= 0.0 or cell_um <= 0.0:
        raise ValueError("screen_period_um and cell_um must be > 0")
    steps = max(2, int(tone_steps))

    # The screen itself: band height quantized to 1/steps of the period.
    u = _axis(tone.shape, cell_um, screen_angle_deg)
    band_frac = (u / float(screen_period_um)) % 1.0
    quantized = np.floor(tone * steps) / steps
    mask = band_frac < quantized

    report: dict[str, Any] = {
        "screen_period_um": float(screen_period_um),
        "tone_steps": steps,
        "zones": [],
    }

    for i, (where, zone) in enumerate(zones):
        where = np.asarray(where, dtype=bool)
        if where.shape != tone.shape:
            raise ValueError(f"zone {i} mask shape {where.shape} != tone {tone.shape}")

        period = zone_period_um(base_period_um, zone)
        line_um = period * zone.duty
        gap_um = period * (1.0 - zone.duty)
        angle = (screen_angle_deg + 90.0) if zone.angle_deg is None else zone.angle_deg

        # Widen the band so the zone keeps its tone despite the grating removing
        # half the metal; the cap is where widening runs out of period.
        cap = tone_cap(zone)
        z_tone = np.minimum(tone / zone.duty, 1.0) if zone.hold_tone else tone
        z_band = (u / float(screen_period_um)) % 1.0 < (np.floor(z_tone * steps) / steps)

        v = _axis(tone.shape, cell_um, angle)
        grating = ((v / period) % 1.0) < zone.duty

        mask = np.where(where, z_band & grating, mask)

        band_um = (np.floor(z_tone * steps) / steps) * screen_period_um
        too_narrow = where & (band_um < min_periods * period)
        n_zone = int(where.sum())
        report["zones"].append(
            {
                "index": i,
                "label": zone.label,
                "period_um": round(period, 4),
                "line_um": round(line_um, 4),
                "gap_um": round(gap_um, 4),
                "angle_deg": round(float(angle), 3),
                "period_scale": zone.period_scale,
                "duty": zone.duty,
                "clears_litho_floor": bool(
                    line_um >= min_feature_um - 1e-9 and gap_um >= min_feature_um - 1e-9
                ),
                "tone_cap": round(cap, 4),
                "min_tone_for_colour": round(
                    min_tone_for_colour(
                        period, screen_period_um,
                        duty=zone.duty, hold_tone=zone.hold_tone, min_periods=min_periods,
                    ),
                    4,
                ),
                "frac_band_too_narrow": round(
                    float(too_narrow.sum()) / n_zone if n_zone else 0.0, 4
                ),
                "frac_of_image": round(n_zone / float(tone.size), 4),
            }
        )

    report["all_zones_printable"] = all(z["clears_litho_floor"] for z in report["zones"])
    report["coverage"] = round(float(mask.mean()), 4)
    return mask, report


def min_base_period_um(
    n_steps: int,
    spread: float = 1.45,
    duty: float = 0.5,
    min_feature_um: float = MIN_FEATURE_UM,
) -> float:
    """Smallest base period whose WHOLE hue ladder still clears the litho floor.

    A ladder centred on the obvious 4.4 um accent period does not fit: its blue
    end lands at 0.83 x 4.4 = 3.65 um, i.e. 1.83 um lines, under the 2.0 um
    floor. The ladder has to be centred higher, and this says how much higher.
    """
    lo = min(hue_ladder(n_steps, spread))
    if duty <= 0.0 or lo <= 0.0:
        raise ValueError("duty and ladder scales must be > 0")
    return float(min_feature_um / (duty * lo))


def hue_ladder(n: int, spread: float = 1.45) -> list[float]:
    """``n`` period scales spanning the visible, as multipliers of a base.

    ``spread`` is the red/blue period ratio; 650/450 = 1.44 covers the visible
    band, so the default lands the ends at the ends of the spectrum whatever the
    base period happens to be.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    if n == 1:
        return [1.0]
    lo = spread ** -0.5
    return [round(lo * spread ** (i / (n - 1)), 4) for i in range(n)]
