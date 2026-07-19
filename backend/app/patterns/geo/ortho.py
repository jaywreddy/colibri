"""Pure-numpy orthographic projection of lon/lat rings onto a unit sphere disk.

An orthographic projection views the globe as if from infinitely far away: the
visible hemisphere maps into the unit disk |(x, y)| <= 1, foreshortening toward
the limb exactly the way a real sphere does. Points on the far hemisphere are
culled (``cos c < 0``). A ring that straddles the limb must be *split* into the
runs that stay on the near hemisphere and *closed along the limb arc* between
them, otherwise a filled polygon would either drop the on-disk part or chord
straight across the disk.

Everything here is vectorized numpy — no shapely, no geopandas. The only heavy
lifting is Douglas–Peucker ring simplification, also pure numpy, so coarse grids
render crisp coastlines instead of noisy stair-steps.

Convention: projected ``y`` grows UP (math convention). The rasterizer flips to
image space (y down). x, y are in sphere-radius units (disk radius == 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Latitude clamp for meridian sampling so poles don't degenerate.
_POLE_CLAMP = 89.999


@dataclass(frozen=True)
class Orthographic:
    """Orthographic projection centered on (lat0, lon0), both in degrees.

    ``limb_slack`` lets points a hair behind the true limb (cos c just below 0)
    still count as visible, so rings that graze the edge close cleanly rather
    than flickering a vertex on/off. It is a cull tolerance only; coordinates are
    still clamped onto the disk by the clip step.
    """

    lat0: float = 30.0
    lon0: float = -50.0
    limb_slack: float = 0.02

    def project(self, lon_deg: np.ndarray, lat_deg: np.ndarray):
        """(lon, lat) arrays [deg] -> (x, y, cosc) arrays.

        ``x, y`` are sphere-radius units (disk radius 1, y up). ``cosc`` is the
        cosine of the angular distance from the projection center: >= 0 on the
        visible near hemisphere, < 0 behind the globe.
        """
        lon = np.radians(np.asarray(lon_deg, dtype=float) - self.lon0)
        lat = np.radians(np.asarray(lat_deg, dtype=float))
        lat0 = math.radians(self.lat0)
        clat = np.cos(lat)
        cosc = math.sin(lat0) * np.sin(lat) + math.cos(lat0) * clat * np.cos(lon)
        x = clat * np.sin(lon)
        y = math.cos(lat0) * np.sin(lat) - math.sin(lat0) * clat * np.cos(lon)
        return x, y, cosc

    def visible(self, cosc: np.ndarray) -> np.ndarray:
        return cosc >= -self.limb_slack


def project_ring(proj: Orthographic, ring_lonlat: np.ndarray):
    """Project one closed lon/lat ring and split it at the limb.

    Returns a list of pixel-space-ready **visible runs**, each an (M, 2) array of
    projected (x, y). A run that came from a straddling ring has its limb-crossing
    endpoints placed *on* the limb circle, and consecutive runs of the same ring
    are joined by an arc along the limb (added by :func:`close_on_limb` at fill
    time). Here we only produce the visible vertex runs plus, for each run, the
    limb angle at its two ends so the caller can stitch the arc.

    ring_lonlat: (N, 2) array of (lon, lat) in degrees, first != last is fine.
    """
    ring = np.asarray(ring_lonlat, dtype=float)
    if ring.shape[0] < 3:
        return []
    # Ensure closure for edge walking.
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])

    x, y, cosc = proj.project(ring[:, 0], ring[:, 1])
    vis = proj.visible(cosc)

    if vis.all():
        return [np.column_stack([x, y])]
    if not vis.any():
        return []

    # Walk edges; where visibility flips, interpolate a point onto the limb.
    runs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    n = len(ring) - 1  # last repeats first
    for i in range(n):
        a_vis, b_vis = vis[i], vis[i + 1]
        ax, ay = x[i], y[i]
        if a_vis:
            cur.append((ax, ay))
        if a_vis != b_vis:
            # Crossing: bisect the lon/lat edge for the point where cosc == 0
            # (the true limb). Keep ``vlon,vlat`` on the visible (cosc>=0) side
            # and ``ilon,ilat`` on the invisible side; converge toward cosc==0.
            if a_vis:
                vlon, vlat, ilon, ilat = ring[i][0], ring[i][1], ring[i + 1][0], ring[i + 1][1]
            else:
                vlon, vlat, ilon, ilat = ring[i + 1][0], ring[i + 1][1], ring[i][0], ring[i][1]
            m_lon = 0.5 * (vlon + ilon)
            m_lat = 0.5 * (vlat + ilat)
            for _ in range(24):
                m_lon = 0.5 * (vlon + ilon)
                m_lat = 0.5 * (vlat + ilat)
                _, _, cm = proj.project(np.array([m_lon]), np.array([m_lat]))
                if cm[0] >= 0.0:
                    vlon, vlat = m_lon, m_lat
                else:
                    ilon, ilat = m_lon, m_lat
            mx, my, _ = proj.project(np.array([m_lon]), np.array([m_lat]))
            px, py = float(mx[0]), float(my[0])
            # Snap onto the unit limb so the closing arc lands exactly on it.
            r = math.hypot(px, py) or 1.0
            px, py = px / r, py / r
            if a_vis and not b_vis:
                cur.append((px, py))
                runs.append(cur)
                cur = []
            else:  # entering visibility
                cur = [(px, py)]
    if cur:
        runs.append(cur)

    return [np.asarray(r, dtype=float) for r in runs if len(r) >= 2]


def close_on_limb(runs, arc_steps: int = 64):
    """Stitch projected visible runs of one ring into a single fillable polygon.

    Runs come back from :func:`project_ring` in ring-traversal order. Between the
    end of one run (on the limb) and the start of the next (on the limb), the true
    coastline went behind the globe, so the correct closure is the **limb arc**
    connecting the two, swept the short way. For a ring fully on the near
    hemisphere there is a single run and no arc is needed.

    Returns one (K, 2) array of projected (x, y) ready to fill, or None.
    """
    if not runs:
        return None
    if len(runs) == 1:
        return runs[0]

    # Concatenate runs, inserting a limb arc between the last point of each run
    # and the first point of the next (wrapping the final run back to the first).
    pts: list[np.ndarray] = []
    m = len(runs)
    for i in range(m):
        run = runs[i]
        pts.append(run)
        nxt = runs[(i + 1) % m]
        a = run[-1]
        b = nxt[0]
        a_ang = math.atan2(a[1], a[0])
        b_ang = math.atan2(b[1], b[0])
        d = b_ang - a_ang
        # shortest signed sweep into (-pi, pi]
        while d <= -math.pi:
            d += 2 * math.pi
        while d > math.pi:
            d -= 2 * math.pi
        steps = max(1, int(abs(d) / (2 * math.pi) * arc_steps))
        arc = []
        for s in range(1, steps):
            t = s / steps
            ang = a_ang + d * t
            arc.append((math.cos(ang), math.sin(ang)))
        if arc:
            pts.append(np.asarray(arc, dtype=float))
    return np.vstack(pts)


def clamp_to_disk(xy: np.ndarray) -> np.ndarray:
    """Clamp any points just outside the unit disk back onto it (radius 1)."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    out = r > 1.0
    if out.any():
        xy = xy.copy()
        xy[out, 0] /= r[out]
        xy[out, 1] /= r[out]
    return xy


# --------------------------------------------------------------------------
# Douglas–Peucker ring simplification (pure numpy, iterative — no recursion
# limit worries on dense rings).
# --------------------------------------------------------------------------
def simplify_ring(ring: np.ndarray, tol: float) -> np.ndarray:
    """Douglas–Peucker simplify an (N, 2) polyline/ring; keep endpoints.

    ``tol`` is the max perpendicular deviation in the ring's own units. Operates
    on lon/lat OR projected xy the same way. Returns the kept vertices in order.
    """
    pts = np.asarray(ring, dtype=float)
    n = len(pts)
    if n < 3 or tol <= 0:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a = pts[i0]
        b = pts[i1]
        seg = b - a
        seg_len2 = seg[0] * seg[0] + seg[1] * seg[1]
        rel = pts[i0 + 1:i1] - a
        if seg_len2 == 0.0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            cross = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0])
            dist = cross / math.sqrt(seg_len2)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            idx = i0 + 1 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return pts[keep]


def ring_area(ring: np.ndarray) -> float:
    """Absolute shoelace area of an (N, 2) ring (units squared)."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
