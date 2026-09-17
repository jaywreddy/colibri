"""``PlateSpec.kind`` — the one classification every writer dispatches on.

The five constructions used to be re-derived from the slug in nine places
(``plates`` six, ``export_fine`` two, ``witness_dies`` one) and two of them
drifted apart. These tests pin WHAT each production face is and WHAT the two
hidden two-ply exemplars are, so a slug rename or a registration change that
would silently move a face onto another writer's branch fails here first.
"""
from __future__ import annotations

import dataclasses

import pytest

from app import export_fine as EF
from app import plates as P
from app.boxes import default_box_spec
from app.plates import FaceKind, PlateSpec

# The production plan, face by face (boxes.default_box_spec).
PRODUCTION_KINDS = {
    "top": FaceKind.REGION,      # monogram-jp as a single-layer colour map
    "front": FaceKind.REGION,    # globe-atlantic, ditto
    "back": FaceKind.PHOTO,      # the Paris photograph as a line screen
    "left": FaceKind.PHOTO,      # beach
    "right": FaceKind.PHOTO,     # sunset
    "bottom": FaceKind.SOLID,    # the solid gold base plate
}


def test_production_faces_have_the_expected_kind():
    spec = default_box_spec()
    got = {fid: spec.faces[fid].kind for fid in PRODUCTION_KINDS}
    assert got == PRODUCTION_KINDS


def test_no_production_face_is_two_ply():
    """Every face of the shipping box is ONE written ply (CLAUDE.md). A
    ``TWO_PLY`` face would take the silhouette-and-carrier branch and write a
    back layer, which the witness plate refuses to dice."""
    spec = default_box_spec()
    for fid, plate in spec.faces.items():
        assert plate.single_ply, fid
        assert plate.kind is not FaceKind.TWO_PLY, fid


def test_blank_face_kind():
    assert PlateSpec(pattern_slug=P.BLANK_SLUG).kind is FaceKind.BLANK
    # single_ply does not enter the blank/solid/photo decision
    assert PlateSpec(pattern_slug=P.BLANK_SLUG, single_ply=True).kind is FaceKind.BLANK


@pytest.mark.parametrize("slug", sorted(P.SHIMMER_MOIRE_SLUGS | P.SWITCH_INTERLACE_SLUGS))
def test_exemplars_are_two_ply(slug):
    """The two hidden exemplars — the shading moiré (``monogram-jp`` on two
    plies) and the barrier switch (``globe-duo-phase``) — are the only things
    that reach ``FaceKind.TWO_PLY``, and only when they are NOT single-ply."""
    assert PlateSpec(pattern_slug=slug, single_ply=False).kind is FaceKind.TWO_PLY


def test_single_ply_monogram_is_region_not_two_ply():
    """Same slug, both sides of the decision: the production lid is a region
    map, its exemplar twin is the shading-moiré silhouette."""
    region = PlateSpec(pattern_slug="monogram-jp", single_ply=True)
    exemplar = PlateSpec(pattern_slug="monogram-jp", single_ply=False)
    assert region.kind is FaceKind.REGION
    assert exemplar.kind is FaceKind.TWO_PLY
    # ...and the writers follow it: only the exemplar has a silhouette pair.
    assert P._centerpiece_masks(region, 64) is None
    assert P._centerpiece_masks(exemplar, 64) is not None


def test_kind_is_not_part_of_the_plate_hash():
    """``kind`` is derived, so it must stay out of ``to_dict`` — otherwise
    every cached plate id would move the day the enum gained a member."""
    spec = default_box_spec().faces["top"]
    before = P.plate_hash(spec)
    assert "kind" not in spec.to_dict()
    _ = spec.kind
    assert P.plate_hash(spec) == before


def test_recipe_data_stamps_the_kind():
    spec = default_box_spec()
    for fid, expected in PRODUCTION_KINDS.items():
        rd = P._carrier_recipe_data(spec.faces[fid])
        assert rd["face_kind"] == expected.value, fid
        # the flags the renderer reads agree with the kind that set them
        assert rd["blank"] is (expected is FaceKind.BLANK)
        assert rd["solid"] is (expected is FaceKind.SOLID)
        assert rd["region_art"] is (expected is FaceKind.REGION)
        assert rd["art_solid"] is (
            expected in (FaceKind.PHOTO, FaceKind.SOLID, FaceKind.REGION))


def test_zone_masks_follow_the_kind():
    """A BLANK or SOLID face returns all-empty zones (so every ``.any()`` gate
    in ``build_plate_fine`` falls through); a REGION face carries a label map
    and no silhouette."""
    spec = default_box_spec()
    bottom = EF._build_zone_masks(spec.faces["bottom"], 60.0)   # SOLID
    assert not bottom.frame.any() and not bottom.front_art.any()
    assert bottom.art_regions is None

    top = EF._build_zone_masks(spec.faces["top"], 60.0)         # REGION
    assert top.art_regions is not None
    assert not top.front_art.any() and not top.back_art.any()


def test_side_photo_spec_keeps_the_kind():
    """``witness_dies.side_photo_spec`` swaps the photograph with
    ``dataclasses.replace``; the replacement must classify the same way (a
    cached property on the ORIGINAL must not leak into the copy)."""
    from app.witness_dies import side_photo_spec

    base = default_box_spec().faces["left"]
    assert base.kind is FaceKind.PHOTO          # populate the cache first
    other = side_photo_spec("garden", "authored", 106)
    assert other.kind is FaceKind.PHOTO
    assert other.pattern_params["image"] == "garden"

    # and a replace that changes the slug reclassifies
    solid = dataclasses.replace(base, pattern_slug=P.SOLID_SLUG)
    assert solid.kind is FaceKind.SOLID
