"""Compute bounds on the /sim endpoints.

The Fraunhofer path zero-pads to a square complex grid of side
max(n_angles, raster), so an unbounded n_angles is a multi-gigabyte allocation
on a 13.7 GB host. These tests pin the request-model bounds and the kernel-side
byte budget, and check that the budget refuses *before* anything is allocated.

Everything here runs on synthetic 32x32 masks — the whole file is well under a
second and never touches a real pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.api.sim import FftRequest, PropagateRequest
from app.sim import MAX_ATLAS_CELLS, MAX_FFT_CELLS, check_fft_budget
from app.sim.angular_spectrum import propagate
from app.sim.fraunhofer import fraunhofer_far_field


def _write_variant(root: Path, size: int = 32, pitch_um: float = 1.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in ("front.png", "back.png"):
        arr = (rng.random((size, size)) > 0.5).astype(np.uint8) * 255
        Image.fromarray(arr, "L").save(root / name)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "pixel_pitch_um": pitch_um,
                "substrate": {"thickness_um": 500.0, "n": 1.46},
            }
        )
    )
    return root


# --- request-model bounds -------------------------------------------------


@pytest.mark.parametrize("n_angles", [0, 8, 63, 2049, 15000])
def test_fft_rejects_out_of_range_n_angles(n_angles: int) -> None:
    with pytest.raises(ValidationError):
        FftRequest(slug="s", variant="v", n_angles=n_angles)


def test_fft_accepts_the_bound_edges() -> None:
    assert FftRequest(slug="s", variant="v", n_angles=64).n_angles == 64
    assert FftRequest(slug="s", variant="v", n_angles=2048).n_angles == 2048
    assert FftRequest(slug="s", variant="v").n_angles == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wavelengths_um": []},
        {"wavelengths_um": [0.5] * 9},
        {"wavelengths_um": [0.0, 0.55]},
        {"wavelengths_um": [50.0]},
        {"max_angle_deg": 0.0},
        {"max_angle_deg": 90.0},
        {"downsample": 0},
        {"downsample": 17},
    ],
)
def test_fft_rejects_bad_dials(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        FftRequest(slug="s", variant="v", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view_angles_deg": []},
        {"view_angles_deg": [0.0] * 17},
        {"view_angles_deg": [95.0]},
        {"wavelengths_um": [0.5] * 9},
        {"downsample": 0},
        {"observer_distance_um": 0.0},
    ],
)
def test_propagate_rejects_bad_dials(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        PropagateRequest(slug="s", variant="v", **kwargs)


def test_propagate_bounds_tile_product() -> None:
    """16 angles x 8 wavelengths is 128 four-FFT rounds — refused even though
    each list is individually in range."""
    with pytest.raises(ValidationError, match="tiles"):
        PropagateRequest(
            slug="s",
            variant="v",
            view_angles_deg=[float(i) for i in range(16)],
            wavelengths_um=[0.4 + 0.02 * i for i in range(8)],
        )
    ok = PropagateRequest(slug="s", variant="v")  # 3 x 3 default
    assert len(ok.view_angles_deg) * len(ok.wavelengths_um) == 9


# --- kernel-side byte budget ---------------------------------------------


def test_check_fft_budget_message_names_the_dials() -> None:
    check_fft_budget(MAX_FFT_CELLS, "grid", n_angles=2048)  # at the cap: fine
    with pytest.raises(ValueError, match=r"n_angles=4096") as ei:
        check_fft_budget(4096 * 4096, "grid", n_angles=4096)
    assert f"{MAX_FFT_CELLS:,}" in str(ei.value)


def test_fraunhofer_refuses_oversized_pad_before_allocating(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v")
    with pytest.raises(ValueError, match="Fraunhofer far field"):
        fraunhofer_far_field(
            root, wavelengths_um=[0.55], pixel_pitch_um=1.0, n_angles=2048, downsample=1
        )
    assert not list(root.glob("fft_atlas_*.png"))


def test_fraunhofer_refuses_oversized_atlas(tmp_path: Path) -> None:
    """Eight wavelength slabs of a Nyquist-wide 1024-pad crop overflow the
    atlas cap even though the FFT grid itself fits."""
    root = _write_variant(tmp_path / "v")
    wl = [0.4 + 0.02 * i for i in range(8)]
    with pytest.raises(ValueError, match="Fraunhofer atlas") as ei:
        fraunhofer_far_field(
            root,
            wavelengths_um=wl,
            pixel_pitch_um=1.0,
            n_angles=1024,
            max_angle_deg=89.0,
            downsample=1,
        )
    assert f"{MAX_ATLAS_CELLS:,}" in str(ei.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"downsample": 0},
        {"pixel_pitch_um": 0.0},
        {"wavelengths_um": []},
        {"wavelengths_um": [0.0]},
        {"max_angle_deg": 0.0},
    ],
)
def test_fraunhofer_validates_its_own_dials(tmp_path: Path, kwargs: dict) -> None:
    """The kernel does not trust the router: downsample=0 would make Δx = 0."""
    root = _write_variant(tmp_path / "v")
    call = {"wavelengths_um": [0.55], "pixel_pitch_um": 1.0, "n_angles": 64, **kwargs}
    with pytest.raises(ValueError):
        fraunhofer_far_field(root, **call)


def test_fraunhofer_collapsed_by_downsample(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v", size=8)
    with pytest.raises(ValueError, match="collapses"):
        fraunhofer_far_field(
            root, wavelengths_um=[0.55], pixel_pitch_um=1.0, n_angles=64, downsample=16
        )


# --- max_angle_deg is actually honored ------------------------------------


def test_fraunhofer_reports_the_angle_window_it_achieved(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v", size=32, pitch_um=1.0)
    wl = [0.65, 0.55, 0.45]
    out = fraunhofer_far_field(
        root,
        wavelengths_um=wl,
        pixel_pitch_um=1.0,
        n_angles=64,
        max_angle_deg=10.0,
        downsample=1,
    )
    pad, dx = 64, 1.0
    # sin θ per pixel is exactly λ/(pad·Δx) — the whole point of taking pitch.
    assert out["sin_theta_per_px"] == pytest.approx([lam / (pad * dx) for lam in wl])
    # The reference (longest) λ lands just inside the request, within one bin.
    assert out["half_angle_deg"][0] <= 10.0
    assert out["half_angle_deg"][0] > 10.0 - 1.0
    # A shared FFT means shorter λ cover proportionally less angle.
    assert out["half_angle_deg"] == sorted(out["half_angle_deg"], reverse=True)
    # Atlas geometry matches the reported crop.
    with Image.open(root / out["atlas_name"]) as img:
        w, h = img.size
    assert [h, w // len(wl)] == out["shape"]


def test_fraunhofer_clamps_the_window_to_nyquist(tmp_path: Path) -> None:
    """45 deg is unreachable at Δx = 1 um: |sin θ| <= λ/2Δx = 0.325 (19 deg)."""
    root = _write_variant(tmp_path / "v", pitch_um=1.0)
    out = fraunhofer_far_field(
        root,
        wavelengths_um=[0.65],
        pixel_pitch_um=1.0,
        n_angles=64,
        max_angle_deg=45.0,
        downsample=1,
    )
    assert out["requested_max_angle_deg"] == 45.0
    assert out["half_angle_deg"][0] == pytest.approx(out["nyquist_half_angle_deg"][0])
    assert out["half_angle_deg"][0] < 20.0


def test_fraunhofer_caches_the_atlas(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v")
    call = dict(
        wavelengths_um=[0.55],
        pixel_pitch_um=1.0,
        n_angles=64,
        max_angle_deg=10.0,
        downsample=1,
    )
    first = fraunhofer_far_field(root, **call)
    assert first["cached"] is False
    second = fraunhofer_far_field(root, **call)
    assert second["cached"] is True
    assert second["atlas_name"] == first["atlas_name"]
    assert second["half_angle_deg"] == first["half_angle_deg"]
    # A different window is a different cache entry.
    other = fraunhofer_far_field(root, **{**call, "max_angle_deg": 5.0})
    assert other["atlas_name"] != first["atlas_name"]


# --- propagate guards -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"downsample": 0},
        {"downsample": -1},
        {"pixel_pitch_um": 0.0},
        {"wavelengths_um": []},
        {"wavelengths_um": [0.0]},
        {"view_angles_deg": []},
    ],
)
def test_propagate_validates_its_own_dials(tmp_path: Path, kwargs: dict) -> None:
    """downsample=0 gives Δx = 0, which poisons fftfreq with infinities instead
    of raising — so it must be refused up front."""
    root = _write_variant(tmp_path / "v")
    call = {
        "pixel_pitch_um": 1.0,
        "wavelengths_um": [0.55],
        "view_angles_deg": [0.0],
        **kwargs,
    }
    with pytest.raises(ValueError):
        propagate(root, **call)
    assert not list(root.glob("asm_atlas_*.png"))


def test_propagate_collapsed_by_downsample(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v", size=8)
    with pytest.raises(ValueError, match="collapses"):
        propagate(
            root,
            pixel_pitch_um=1.0,
            wavelengths_um=[0.55],
            view_angles_deg=[0.0],
            downsample=16,
        )


def test_propagate_tiny_grid_runs_and_caches(tmp_path: Path) -> None:
    root = _write_variant(tmp_path / "v", size=32)
    call = dict(
        pixel_pitch_um=1.0,
        wavelengths_um=[0.55],
        view_angles_deg=[0.0, 10.0],
        downsample=1,
    )
    out = propagate(root, **call)
    assert out["cached"] is False
    assert out["rows"] == 2 and out["cols"] == 1
    assert propagate(root, **call)["cached"] is True
