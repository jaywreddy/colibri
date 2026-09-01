"""/sim/collage — the tilt-sweep inspection sheet."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


SLUG = "monogram-carrier-reveal"   # strong x-axis reveal
FLAT = "monogram-jp"               # single layer: empty back, no effect possible
YAXIS = "bitmap-halftone"          # lines run horizontally -> effect is on y


def _metrics(client, slug, **params):
    r = client.get(f"/sim/collage/{slug}/default/metrics", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_sheet_is_a_png_carrying_its_metrics_in_a_header():
    c = TestClient(create_app())
    r = c.get(f"/sim/collage/{SLUG}/default", params={"steps": 5, "tile_px": 64})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    # The caller showing the image must not have to composite the sweep twice.
    m = json.loads(r.headers["X-Collage-Metrics"])
    assert m["effect_strength"] > 0.1


def test_metrics_endpoint_agrees_with_the_sheet(client):
    m = _metrics(client, SLUG, steps=5, span_deg=6)
    assert m["slug"] == SLUG
    assert len(m["angles_deg"]) == 5
    assert m["angles_deg"][0] == -6.0 and m["angles_deg"][-1] == 6.0
    assert 0.0 in m["angles_deg"]                    # head-on is always a column
    assert m["effect_strength"] > 0.1


def test_auto_axis_finds_an_effect_a_fixed_axis_would_miss(client):
    """A layer whose lines run horizontally is invariant under a horizontal
    shift, so an x-only sweep reads it as perfectly dead. This is not
    hypothetical — it is the difference between 0.000 and a working pattern."""
    x = _metrics(client, YAXIS, steps=7, axis="x")
    y = _metrics(client, YAXIS, steps=7, axis="y")
    auto = _metrics(client, YAXIS, steps=7, axis="auto")
    assert x["effect_strength"] == pytest.approx(0.0, abs=1e-6)
    assert y["effect_strength"] > 0.1
    assert auto["axis"] == "y"
    assert auto["effect_strength"] == pytest.approx(y["effect_strength"], rel=1e-6)


def test_auto_does_not_flip_a_pattern_that_lives_on_x(client):
    m = _metrics(client, SLUG, steps=7, axis="auto")
    assert m["axis"] == "x"


def test_a_single_layer_pattern_reports_no_effect(client):
    """An empty back layer cannot produce parallax at any angle, and the sheet
    should say so rather than inventing motion from the raster border."""
    m = _metrics(client, FLAT, steps=7, axis="auto")
    assert m["effect_strength"] == pytest.approx(0.0, abs=1e-6)
    assert m["changed_frac"] == pytest.approx(0.0, abs=1e-6)


def test_peak_pair_is_reported_and_is_a_real_pair_of_angles(client):
    m = _metrics(client, SLUG, steps=7)
    lo, hi = m["peak_pair_deg"]
    assert lo in m["angles_deg"] and hi in m["angles_deg"]


@pytest.mark.parametrize(
    "params",
    [
        {"steps": 2}, {"steps": 99}, {"span_deg": 0.0}, {"span_deg": 99},
        {"illum": "x-ray"}, {"axis": "z"},
    ],
)
def test_out_of_range_dials_are_rejected(client, params):
    r = client.get(f"/sim/collage/{SLUG}/default/metrics", params=params)
    assert r.status_code == 400, r.text


def test_tile_and_column_bounds_are_enforced(client):
    assert client.get(f"/sim/collage/{SLUG}/default", params={"tile_px": 4000}).status_code == 400
    assert client.get(f"/sim/collage/{SLUG}/default", params={"cols": 0}).status_code == 400


def test_unknown_pattern_is_404(client):
    assert client.get("/sim/collage/not-a-pattern/default/metrics").status_code == 404


def test_unknown_variant_hash_is_404_and_never_generates(client):
    """Only the literal 'default' may trigger materialization — an arbitrary
    hash must not be able to kick off pattern generation from a URL."""
    r = client.get(f"/sim/collage/{SLUG}/deadbeef99/metrics")
    assert r.status_code == 404
