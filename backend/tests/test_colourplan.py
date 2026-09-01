"""Photograph -> period field: the hue mapping, the rules, and their gates."""
from __future__ import annotations

import numpy as np
import pytest

from app.patterns.bitmap.colourplan import (
    BLUE_HUE_DEG,
    HUE_WRAP_DEG,
    PAULA_HUE,
    PAULA_PLAIN,
    PAULA_ZONES,
    PRESETS,
    ColourPlan,
    Rule,
    _hsv,
    build_period_field,
    hue_to_scale,
    plan_from_json,
    plan_to_json,
)


def _solid(rgb: tuple[float, float, float], n: int = 48) -> np.ndarray:
    return np.tile(np.asarray(rgb, np.float32), (n, n, 1))


def _hue_image(hues: list[float], n: int = 48) -> np.ndarray:
    """Vertical bands of the given hues, fully saturated and bright."""
    import colorsys

    out = np.zeros((n, n, 3), np.float32)
    w = n // len(hues)
    for i, h in enumerate(hues):
        out[:, i * w:(i + 1) * w] = colorsys.hsv_to_rgb(h / 360.0, 1.0, 1.0)
    return out


# --- the hue mapping --------------------------------------------------------


def test_red_takes_the_long_period_and_blue_the_short_one():
    """lambda = d*k, so a longer period diffracts a longer wavelength. Mapping
    red to the top of the ladder keeps the plate's colours in the photograph's
    ORDER even though the absolute hue depends on where the viewer stands."""
    red = float(hue_to_scale(np.array([0.0]))[0])
    green = float(hue_to_scale(np.array([120.0]))[0])
    blue = float(hue_to_scale(np.array([BLUE_HUE_DEG]))[0])
    assert red > green > blue


def test_magenta_folds_to_the_RED_end_not_the_blue_one():
    """The bug this constant exists for. Hue is a circle and the spectrum is a
    line; cutting at 240 sends every pink petal (hue 330-350, and the 90th
    percentile of the reference carpet is 347) to the BLUE end — the opposite
    of its colour."""
    pink = float(hue_to_scale(np.array([345.0]))[0])
    red = float(hue_to_scale(np.array([5.0]))[0])
    blue = float(hue_to_scale(np.array([BLUE_HUE_DEG]))[0])
    assert abs(pink - red) < abs(pink - blue)
    assert HUE_WRAP_DEG > BLUE_HUE_DEG


def test_hue_is_monotone_across_the_spectral_range():
    h = np.linspace(0.0, BLUE_HUE_DEG, 40)
    s = hue_to_scale(h)
    assert np.all(np.diff(s) <= 1e-6)


# --- modes ------------------------------------------------------------------


def test_plain_leaves_every_pixel_uncoloured():
    ids, per, rep = build_period_field(_hue_image([0, 60, 120, 240]), PAULA_PLAIN)
    assert ids.max() == 0
    assert rep["frac_coloured"] == 0.0


def test_hue_mode_gives_different_rungs_to_different_hues():
    plan = ColourPlan(mode="hue", coarsen_px=0, ladder_steps=12)
    ids, _, rep = build_period_field(_hue_image([0, 60, 120, 240]), plan)
    assert rep["rungs_used"] >= 3
    cols = [int(np.median(ids[:, i * 12:(i + 1) * 12])) for i in range(4)]
    assert cols == sorted(cols, reverse=True), "red should rank above blue"


def test_a_fixed_rule_pins_its_rung_regardless_of_hue():
    plan = ColourPlan(
        mode="zones", coarsen_px=0, ladder_steps=12,
        rules=(Rule(name="all", sat=(0.0, 1.0), mode="fixed", period_scale=1.0),),
    )
    ids, per, _ = build_period_field(_hue_image([0, 120, 240]), plan)
    assert len(np.unique(ids)) == 1
    assert per[int(ids[0, 0])] == pytest.approx(plan.base_period_um, rel=0.05)


def test_a_plain_rule_claims_a_region_and_leaves_it_gold():
    """The exclusion primitive: the crowd on the pavement is as saturated as the
    petals and no colour rule can tell them apart, but a bbox can."""
    plan = ColourPlan(
        mode="zones", coarsen_px=0,
        rules=(
            Rule(name="keep out", bbox=(0.0, 0.0, 0.5, 1.0), mode="plain"),
            Rule(name="rest", sat=(0.0, 1.0), mode="fixed", period_scale=1.0),
        ),
    )
    ids, _, rep = build_period_field(_hue_image([0, 120]), plan)
    assert ids[:, :20].max() == 0
    assert ids[:, 30:].min() > 0
    assert rep["rules"][0]["name"] == "keep out"


def test_first_matching_rule_wins():
    plan = ColourPlan(
        mode="zones", coarsen_px=0,
        rules=(
            Rule(name="first", sat=(0.0, 1.0), mode="fixed", period_scale=1.2),
            Rule(name="second", sat=(0.0, 1.0), mode="fixed", period_scale=0.85),
        ),
    )
    ids, per, rep = build_period_field(_hue_image([0]), plan)
    assert rep["rules"][1]["frac"] == 0.0
    assert per[int(ids[0, 0])] > plan.base_period_um


# --- the gates --------------------------------------------------------------


def test_hue_is_not_read_where_there_is_no_hue():
    """Dark hair computed to teal often enough to be picked up by a spectacle
    rule, and in hue mode it took a random rung. The fix is refusing to read
    hue below a value/saturation floor, not a better hue window."""
    dark = _solid((0.04, 0.05, 0.06))
    plan = ColourPlan(mode="hue", coarsen_px=0, hue_min_value=0.14)
    ids, _, rep = build_period_field(dark, plan)
    assert ids.max() == 0
    assert rep["frac_coloured"] == 0.0
    # and with the gate off it does take a rung
    ids2, _, _ = build_period_field(dark, ColourPlan(mode="hue", coarsen_px=0,
                                                     hue_min_value=0.0,
                                                     hue_min_sat=0.0))
    assert ids2.max() > 0


def test_an_authored_region_still_colours_its_own_shadows():
    """The value gate applies to hue-DERIVED assignment only; a sweater's dark
    folds are still the sweater."""
    plan = ColourPlan(
        mode="zones", coarsen_px=0, hue_min_value=0.5,
        rules=(Rule(name="z", sat=(0.0, 1.0), mode="fixed", period_scale=1.0),),
    )
    ids, _, _ = build_period_field(_solid((0.04, 0.05, 0.06)), plan)
    assert ids.min() > 0


def test_equalising_spreads_a_clustered_subject_across_the_ladder():
    """90% of the reference carpet's pixels fall in 33% of the ladder because
    the petals are almost all warm. Equalising is what makes neighbouring
    petals separate by several rungs instead of one."""
    warm = _hue_image([0, 8, 16, 24, 32, 40])
    base = dict(mode="hue", coarsen_px=0, ladder_steps=12)
    _, _, direct = build_period_field(warm, ColourPlan(hue_equalize=False, **base))
    _, _, eq = build_period_field(warm, ColourPlan(hue_equalize=True, **base))
    assert direct["rungs_used"] <= 3
    assert eq["rungs_used"] >= 6


def test_equalising_preserves_the_ORDER_of_the_hues():
    """It trades absolute fidelity for distinctness, but a redder band must
    still take a longer period than a yellower one or the picture's colours
    would come out shuffled."""
    warm = _hue_image([0, 12, 24, 36])
    ids, per, _ = build_period_field(
        warm, ColourPlan(mode="hue", coarsen_px=0, ladder_steps=12,
                         hue_equalize=True))
    n = ids.shape[1] // 4
    seq = [per[int(np.median(ids[:, i * n + 2:(i + 1) * n - 2]))] for i in range(4)]
    assert seq == sorted(seq, reverse=True)


# --- coarsening -------------------------------------------------------------


def test_coarsening_organises_the_field_and_never_invents_a_rung():
    """A median would: averaging rung 1 and rung 11 gives rung 6, a colour
    neither region asked for. The filter is a majority vote for that reason."""
    rng = np.random.default_rng(2)
    noisy = rng.random((60, 60, 3)).astype(np.float32)
    fine, _, r_fine = build_period_field(
        noisy, ColourPlan(mode="hue", coarsen_px=0, ladder_steps=12))
    coarse, _, r_coarse = build_period_field(
        noisy, ColourPlan(mode="hue", coarsen_px=9, ladder_steps=12))
    # fewer transitions along a row == longer runs == fewer rectangles
    assert (np.diff(coarse, axis=1) != 0).sum() < (np.diff(fine, axis=1) != 0).sum()
    assert set(np.unique(coarse)) <= set(np.unique(fine)) | {0}


# --- the litho floor --------------------------------------------------------


def test_the_report_flags_a_base_period_whose_blue_end_is_unprintable():
    """The ladder's blue end is the finest line on the plate, so it is what
    gates the whole plan — and 4.4 um, the obvious choice, does not fit."""
    plan = ColourPlan(mode="hue", coarsen_px=0, base_period_um=4.4, ladder_steps=12)
    _, _, rep = build_period_field(_hue_image([0, 120, 240]), plan)
    assert rep["min_printable_base_um"] == pytest.approx(4.816, abs=0.01)
    assert rep["base_is_printable"] is False
    assert any(not r["clears_litho_floor"] for r in rep["rungs"])

    ok = ColourPlan(mode="hue", coarsen_px=0, base_period_um=5.0, ladder_steps=12)
    _, _, rep2 = build_period_field(_hue_image([0, 120, 240]), ok)
    assert rep2["base_is_printable"] is True
    assert rep2["all_used_rungs_printable"] is True


def test_the_report_accounts_for_every_rung():
    ids, per, rep = build_period_field(_hue_image([0, 120, 240]),
                                       ColourPlan(mode="hue", coarsen_px=0))
    assert len(rep["rungs"]) == len(per)
    assert sum(r["frac"] for r in rep["rungs"]) == pytest.approx(
        rep["frac_coloured"], abs=0.02)


# --- painted masks ----------------------------------------------------------


def test_a_painted_mask_restricts_a_rule(tmp_path):
    """Colour thresholding cannot select a hairline that is not a distinct
    colour along its whole length, so an authored feature is authored."""
    from PIL import Image

    n = 48
    m = np.zeros((n, n), np.uint8)
    m[:, : n // 4] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(m, "L").save(p)
    plan = ColourPlan(
        mode="zones", coarsen_px=0,
        rules=(Rule(name="painted", sat=(0.0, 1.0), mode="fixed",
                    period_scale=1.0, mask_png=str(p)),),
    )
    ids, _, _ = build_period_field(_hue_image([0]), plan)
    assert ids[:, : n // 4].min() > 0
    assert ids[:, n // 4 + 2:].max() == 0


def test_a_missing_mask_is_an_error_not_a_silent_no_op(tmp_path):
    plan = ColourPlan(
        mode="zones",
        rules=(Rule(name="x", mode="fixed", mask_png=str(tmp_path / "nope.png")),),
    )
    with pytest.raises(FileNotFoundError):
        build_period_field(_hue_image([0]), plan)


# --- serialisation and presets ----------------------------------------------


def test_a_plan_round_trips_through_json():
    back = plan_from_json(plan_to_json(PAULA_ZONES))
    assert back.mode == PAULA_ZONES.mode
    assert back.base_period_um == PAULA_ZONES.base_period_um
    assert [r.name for r in back.rules] == [r.name for r in PAULA_ZONES.rules]
    assert back.rules[0].bbox == PAULA_ZONES.rules[0].bbox


def test_the_reference_plans_are_printable_and_distinct():
    assert PAULA_ZONES.base_period_um >= PAULA_ZONES.min_printable_base_um()
    assert PAULA_HUE.base_period_um >= PAULA_HUE.min_printable_base_um()
    assert {p.mode for p in PRESETS.values()} == {"plain", "hue", "zones"}
    names = [r.name for r in PAULA_ZONES.rules]
    assert names == ["crowd", "glasses", "sweater", "flowers"], (
        "order is load-bearing: specific rules must claim before general ones"
    )
    # the carpet is equalised, the whole-frame map is not
    assert PAULA_ZONES.hue_equalize and not PAULA_HUE.hue_equalize


# --- guards -----------------------------------------------------------------


def test_bad_plans_and_rules_are_refused():
    with pytest.raises(ValueError):
        ColourPlan(mode="rainbow")
    with pytest.raises(ValueError):
        ColourPlan(ladder_steps=0)
    with pytest.raises(ValueError):
        Rule(name="x", mode="sparkle")
    with pytest.raises(ValueError):
        Rule(name="x", period_scale=0.0)
    with pytest.raises(ValueError):
        build_period_field(np.zeros((4, 4), np.float32), PAULA_HUE)
