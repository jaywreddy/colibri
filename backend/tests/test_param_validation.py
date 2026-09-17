"""ParamSpec bounds are enforced server-side.

Every case here is deliberately cheap: ``validate_params`` runs before any
geometry, so the rejections cost nothing, and the "valid" cases call the
validator directly rather than materializing a variant (generation is the
heaviest compute in the app).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.patterns.base import registry
from app.service import materialize, validate_params


MONO = "monogram-jp"  # extent_um 500..5000, overlap 0.4..0.85, *_period_um 4.15..6.02


def _specs(slug: str) -> dict:
    return {p.name: p for p in registry[slug].params}


def test_declared_bounds_are_accepted() -> None:
    for name, spec in _specs(MONO).items():
        assert spec.min is not None and spec.max is not None, name
        validate_params(MONO, {name: spec.min})
        validate_params(MONO, {name: spec.max})


def test_defaults_are_accepted() -> None:
    validate_params(MONO, registry[MONO].defaults())


def test_below_min_is_rejected() -> None:
    with pytest.raises(ValueError) as e:
        validate_params(MONO, {"extent_um": 100.0})
    assert "500" in str(e.value)  # message states the limit


def test_above_max_is_rejected() -> None:
    with pytest.raises(ValueError) as e:
        validate_params(MONO, {"extent_um": 40000.0})
    assert "5000" in str(e.value)


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError) as e:
        validate_params(MONO, {"slit_period_um": 60.0})
    assert "slit_period_um" in str(e.value)


def test_string_for_float_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_params(MONO, {"overlap": "not-a-number"})


def test_bool_for_float_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_params(MONO, {"overlap": True})


def test_nonfinite_float_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_params(MONO, {"overlap": float("inf")})


def test_fractional_int_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_params("photo-halftone", {"tone_steps": 22.5})


def test_choice_outside_choices_is_rejected() -> None:
    with pytest.raises(ValueError) as e:
        validate_params("photo-halftone", {"image": "not-a-photograph"})
    assert "beach" in str(e.value)


def test_materialize_rejects_before_generating(isolated_data_root: Path) -> None:
    """The guard fires ahead of the hash, so no variant dir is created."""
    with pytest.raises(ValueError):
        materialize(MONO, {"extent_um": 40000.0, "overlap": 0.6})
    assert not (isolated_data_root / MONO).exists()


def test_api_maps_out_of_range_to_client_error(client: TestClient) -> None:
    r = client.post(
        "/patterns/generate",
        json={"slug": MONO, "params": {"extent_um": 40000.0}},
    )
    assert r.status_code in (400, 422), r.text
    assert "5000" in r.text
