"""Fringe contrast surviving the 2.25 mm ply vs carrier pitch — the number that
sets the garland pitch. Same angular-spectrum model as validate_dies.py's
nearfield gate (incoherent 1 deg source, three wavelengths, eye-cell integration),
run for a ladder of carrier pitches with the leaf grating at 1.09x and 2.5 deg.

    uv run --directory backend python ../tools/dev/nearfield_ladder.py 44 55 66 77 99
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "backend"))

import validate_dies as V  # noqa: E402
from app import witness_dies as wd  # noqa: E402

z = wd.PLY_UM
n_glass = wd.GLASS_N
px = 2.0
N = 2048
angles = np.linspace(-0.5, 0.5, 11)
pitches = [float(a) for a in sys.argv[1:]] or [44.0, 55.0, 66.0, 77.0, 99.0]
print(f"ply {z:g} um, n {n_glass}, source +-0.5 deg, lambda {V.LAM_UM}")
print(f"{'carrier':>8} {'leaf':>7} {'line um':>8} {'arcmin@300':>10} {'Fresnel N':>9} {'zero-gap':>9} {'across':>7} {'survives':>8}")
for c in pitches:
    leaf = c * 1.09
    back = V._grating(N, N, px, c, 0.5, 0.0)
    front = V._grating(N, N, px, leaf, 0.5, 2.5)
    t_back, t_front = 1.0 - back, 1.0 - front
    I_geo = t_back * t_front
    I_sum = np.zeros((N, N))
    t0 = time.perf_counter()
    for lam in V.LAM_UM:
        for th in angles:
            kx = 2 * math.pi / lam * math.sin(math.radians(th))
            xs = (np.arange(N) - N / 2) * px
            U = t_back * np.exp(1j * kx * xs)[None, :]
            U = V._propagate(U, px, z, lam, n_glass)
            I_sum += np.abs(U * t_front) ** 2
    I_sum /= len(V.LAM_UM) * len(angles)
    c_geo, _ = V._fringe_contrast(I_geo, px)
    c_gap, _ = V._fringe_contrast(I_sum, px)
    fres = c ** 2 * n_glass / (4 * 0.55 * z)
    arcmin = math.degrees(math.atan(c / 300_000.0)) * 60
    print(f"{c:8.1f} {leaf:7.1f} {c/2:8.1f} {arcmin:10.2f} {fres:9.2f} {c_geo:9.3f} {c_gap:7.3f} {c_gap/max(1e-9,c_geo):8.2f}   ({time.perf_counter()-t0:.0f}s)")
