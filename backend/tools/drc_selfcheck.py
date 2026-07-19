"""Standalone self-check for effects/drc.py. One process, bounded RAM.

Run: uv run --extra dev python tools/drc_selfcheck.py
"""
from __future__ import annotations

import time

import numpy as np

from app.patterns.effects import drc


def _fmt(d: dict) -> str:
    return (
        f"min_w={d['min_width_um']:.3f} min_gap="
        f"{d['min_gap_um'] if np.isfinite(d['min_gap_um']) else 'inf'} "
        f"n_sub={d['n_subfloor']} n={d['n_features']}"
    )


def case_rects():
    print("=== RECT CASES (floor 2.0 µm, snap_tol 0.5 µm) ===")
    rows = []

    # 1) legal 2.0 µm line — must pass untouched.
    r = np.array([[0.0, 2.0, 0.0, 10.0]])
    out = drc.drc_clean_rects(r, log=False)
    rows.append(("legal 2.0µm line", drc.drc_report(r), drc.drc_report(out),
                 out.shape[0] == 1 and abs((out[0, 1] - out[0, 0]) - 2.0) < 1e-6))

    # 2) 1.9 µm line (short by 0.1 ≤ 0.5) — must WIDEN to 2.0, keep center.
    r = np.array([[10.0, 11.9, 0.0, 10.0]])
    out = drc.drc_clean_rects(r, log=False)
    w = out[0, 1] - out[0, 0] if out.shape[0] else 0.0
    center_ok = out.shape[0] and abs(0.5 * (out[0, 0] + out[0, 1]) - 10.95) < 1e-6
    rows.append(("1.9µm line -> widen", drc.drc_report(r), drc.drc_report(out),
                 abs(w - 2.0) < 1e-6 and center_ok))

    # 3) 0.9 µm line (short by 1.1 > 0.5) — must DROP.
    r = np.array([[0.0, 0.9, 0.0, 10.0]])
    out = drc.drc_clean_rects(r, log=False)
    rows.append(("0.9µm line -> drop", drc.drc_report(r), drc.drc_report(out),
                 out.shape[0] == 0))

    # 4) 1.9 µm GAP between two colinear rects — must CLOSE (merge).
    r = np.array([[0.0, 10.0, 0.0, 5.0], [11.9, 20.0, 0.0, 5.0]])
    out = drc.drc_clean_rects(r, log=False)
    merged = out.shape[0] == 1 and abs(out[0, 0]) < 1e-6 and abs(out[0, 1] - 20.0) < 1e-6
    rows.append(("1.9µm gap -> close", drc.drc_report(r), drc.drc_report(out), merged))

    # 5) legal 2.0 µm gap — must NOT merge.
    r = np.array([[0.0, 10.0, 0.0, 5.0], [12.0, 20.0, 0.0, 5.0]])
    out = drc.drc_clean_rects(r, log=False)
    rows.append(("legal 2.0µm gap", drc.drc_report(r), drc.drc_report(out),
                 out.shape[0] == 2))

    for name, before, after, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s}  before: {_fmt(before)}"
              f"   after: {_fmt(after)}")
    return all(r[3] for r in rows)


def case_poly_sliver():
    print("=== POLY CASE: sliver wedge (min-width open) ===")
    # A rectangle 20µm long with a 0.6µm-wide spike protruding — the spike is
    # sub-floor and must be removed by the open; the body must survive.
    body = np.array([[0, 0], [20, 0], [20, 8], [0, 8], [0, 0]], dtype=float)
    spike = np.array([[9.7, 8], [10.3, 8], [10.0, 14], [9.7, 8]], dtype=float)
    before_body = drc.drc_report([body])
    before_spike = drc.drc_report([spike])
    cleaned = drc.drc_clean_polys([body, spike], log=False)
    after = drc.drc_report(cleaned)
    body_survives = len(cleaned) >= 1
    spike_gone = before_spike["min_width_um"] < 2.0  # confirmed it was sub-floor
    print(f"  body before min_w={before_body['min_width_um']:.2f}  "
          f"spike before min_w={before_spike['min_width_um']:.2f} (sub-floor={spike_gone})")
    print(f"  after clean: {len(cleaned)} ring(s), {_fmt(after)}")
    ok = body_survives and spike_gone and after["min_width_um"] >= 2.0 - 0.6
    print(f"  [{'PASS' if ok else 'FAIL'}] sliver wedge handled")
    return ok


def runtime_100k():
    print("=== RUNTIME: ~100k rects ===")
    rng = np.random.default_rng(0)
    n = 100_000
    x0 = rng.uniform(0, 30_000, n)
    # Mix of legal (2-8µm) and a slug of sub-floor (0.5-1.9µm) widths.
    w = rng.uniform(0.5, 8.0, n)
    y0 = rng.uniform(0, 30_000, n)
    h = rng.uniform(2.0, 20.0, n)
    rects = np.stack([x0, x0 + w, y0, y0 + h], axis=1)
    before = drc.drc_report(rects)
    t0 = time.perf_counter()
    out = drc.drc_clean_rects(rects, log=False)
    dt = time.perf_counter() - t0
    after = drc.drc_report(out)
    print(f"  n_in={n} -> n_out={out.shape[0]}  in {dt*1000:.0f} ms")
    print(f"  before: {_fmt(before)}")
    print(f"  after:  {_fmt(after)}")
    ok = after["min_width_um"] >= 2.0 - 1e-6 and dt < 60.0
    print(f"  [{'PASS' if ok else 'FAIL'}] 100k rects cleaned, no sub-floor, <60s")
    return ok


if __name__ == "__main__":
    results = [case_rects(), case_poly_sliver(), runtime_100k()]
    print()
    print("ALL PASS" if all(results) else "SOME FAILED")
