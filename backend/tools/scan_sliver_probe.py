"""Measure the true minimum gold feature in the capybara scanimation BACK
layer at a FINE analysis raster, before/after the interleaver sliver clip.

Runs the standalone _build at a fine grid (slot >> analysis pixel) so the
measured minimum reflects real geometry, not raster quantisation. One process.

Run: uv run --extra dev python tools/scan_sliver_probe.py
"""
from __future__ import annotations

import numpy as np

from app.patterns.artistic import capybara_scanimation as cs
from app.patterns.effects import drc, gratings


def _runs_1d(line: np.ndarray) -> list[int]:
    idx = np.where(line)[0]
    if len(idx) == 0:
        return []
    return [len(r) for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)]


def true_min_features(mask: np.ndarray, cell_um: float) -> dict:
    """Measure REAL feature dimensions (not row-span artifacts): the minimum
    connected gold run in x (per row) and in y (per column). A sliver is a run
    shorter than the 2 µm floor in EITHER axis."""
    xruns, yruns = [], []
    for r in range(mask.shape[0]):
        xruns += _runs_1d(mask[r, :])
    for c in range(mask.shape[1]):
        yruns += _runs_1d(mask[:, c])
    xr = np.array(xruns) * cell_um if xruns else np.array([np.inf])
    yr = np.array(yruns) * cell_um if yruns else np.array([np.inf])
    n_sub_x = int((xr < 2.0).sum())
    n_sub_y = int((yr < 2.0).sum())
    return {
        "min_x_um": float(xr.min()), "min_y_um": float(yr.min()),
        "n_sub_x": n_sub_x, "n_sub_y": n_sub_y,
        "n_xruns": len(xruns), "n_yruns": len(yruns),
    }


def measure(mask: np.ndarray, cell_um: float, label: str) -> dict:
    t = true_min_features(mask, cell_um)
    print(f"  {label}: min_x={t['min_x_um']:.2f}µm (sub-floor x-runs={t['n_sub_x']}/{t['n_xruns']}) "
          f"min_y={t['min_y_um']:.2f}µm (sub-floor y-runs={t['n_sub_y']}/{t['n_yruns']})")
    return t


def main():
    # Fine analysis: slot = 60/4 = 15 µm; pick cell so slot is ~30 px (0.5 µm).
    frame_pitch = cs_frame = 60.0
    n_phases = 4
    slot_um = frame_pitch / n_phases  # 15 µm
    analysis_px = 0.5                 # µm/pixel -> 15 µm slot = 30 px
    # Fine pixel (0.5 µm) under the 400k lattice cap -> extent <= ~316 µm.
    # 300 µm = 5 frame pitches, fully representative of the periodic interleave;
    # 600x600 = 360k cells, one bounded pass.
    extent = 300.0
    n_grid = int(round(extent / analysis_px))  # 600 -> 360k cells
    print(f"analysis: extent={extent}µm n_grid={n_grid} cell={extent/n_grid:.3f}µm "
          f"slot={slot_um}µm ({slot_um/(extent/n_grid):.0f}px)")

    b = cs._build(
        extent_um=extent,
        frame_pitch_um=frame_pitch,
        n_phases=n_phases,
        carrier_period_um=24.0,
        waterline_y=cs.WATERLINE_Y,
        n_grid=n_grid,
    )
    cell_um = b["cell_um"]

    print("RAW phase mask 0 (post thickness-floor):")
    measure(b["phase_masks"][0], cell_um, "phase0")
    print("BACK water interleave (post slot-snap):")
    rep = measure(b["back_water"], cell_um, "back_water")

    # Controlled expectation: slot 15 µm, bar 45 µm; nothing 0<w<2 µm.
    print(f"\nExpected controlled minimums: slot={slot_um}µm, "
          f"bar={frame_pitch*(1-1/n_phases)}µm")
    n_sub = rep["n_sub_x"] + rep["n_sub_y"]
    print(f"total sub-floor runs (0<feat<2µm) = {n_sub}  (target 0)")
    print("PASS" if n_sub == 0 else "FAIL")


if __name__ == "__main__":
    main()
