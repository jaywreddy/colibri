# plates.py cache-version changelog

Moved out of `backend/app/plates.py` on 2026-09-16, with the two-ply cleanup:
the module carried ~90 lines of per-version prose beside the two markers. The
markers and their CURRENT entry stay in the code, where a bump has to be made
by hand; everything older is history and lives here.

Both markers are checked on the cache-hit path, so a bump is the only thing
that invalidates a warm `backend/data/` after a code or constant change (the
cache keys hash the user spec/params only) — see CLAUDE.md, "cache versions".

## `plates.PLATE_COMPOSE_VERSION`

Bumped when the composed-plate output changes under an unchanged spec hash:
`_raster_compose_plate`, `_paste_centerpiece`, `_centerpiece_masks`, the mask
level palette (`FRAME_LEVEL` / `ART_LEVEL` / `FRAME_BUCKET*` / `RAINBOW_LEVEL`),
or `_carrier_recipe_data`. Integers; v1 and v7-v12 were never written down.

```
v2: first versioned compose — wave 1 changed compose geometry and recipe_data
    (front water band, barrier registration, litho floor) with no key to
    invalidate the caches it had already written.
v3: recipe_data publishes the solved barrier registration
    (switch_interlace_period_um / switch_barrier_phase_um) for the preview
    shader; the mask PNGs are unchanged (the barrier lives in the fab bake, see
    PLATE_SVG_VERSION v6).
v4: recipe_data publishes the EFFECTIVE capybara waterline
    (``water_waterline_y``) and the composed centerpiece mask honours the
    ``waterline`` pattern param instead of the module constant, so the shader's
    water/body split follows the mask it is drawing. Mask PNGs move only on a
    face that actually sets the param (default-param output is unchanged).
v5: recipe_data drops the dead ``water_phase_pitch_preview_um`` key (and its
    WATER_PHASE_PITCH_PREVIEW_UM constant) — no shader line ever read it and the
    uniform is gone. recipe_data is part of the cached manifest, so a warm cache
    would otherwise keep serving the key forever; a consumer added later against
    a stale manifest would find it on some faces and not others. Mask PNGs are
    unchanged by this bump.
v6: the capybara water band drops the submerged body in the PATTERN too
    (``_capybara_and_water``), so capybara-scanimation's generated masks — and
    with them the ``min_feature_um`` / ``central_extra`` (``min_back_gold_um``,
    ``min_front_gold_um``, …) this manifest copies out of ``central_cls.metadata``
    — move. The composed PNGs do NOT: ``_paste_centerpiece`` /
    ``_centerpiece_masks`` already carved the band and the plate pitch is keyed to
    the pattern's unchanged ``pixel_pitch_um``. Without the bump a warm plate slot
    keeps advertising the pre-carve measured minimums next to a re-baked SVG.
v13: FrameSpec grows ``motif_scale`` — a motif-only size dial threaded into
    the wreath grower (leaf/bloom/understory/corner-sprig sizes AND their
    station spacing along the vine; band width and vine gauge untouched). The
    default 1.0 is bit-identical to v12's geometry, but the field is part of
    the frame recipe the compose path bakes, so warm slots must re-derive
    rather than serve a manifest that predates the knob.
v14: the PRODUCTION box lands — three new compose behaviours and three new
    recipe_data keys. BLANK faces (``BLANK_SLUG``) emit nothing on either
    layer; SINGLE-PLY faces (``PlateSpec.single_ply``) move the uniform
    carrier onto the FRONT mask over the back window minus the art box and
    leave the back empty; PHOTO faces (``PHOTO_SLUG``) stamp line-screen bands
    at ART_LEVEL / RAINBOW_LEVEL instead of a silhouette. ``_carrier_recipe_data``
    grows ``blank`` / ``art_solid`` / ``single_ply`` unconditionally, so even a
    face whose PNGs are unchanged must re-derive rather than serve a manifest
    that predates the keys.
v15: RENDER IT LITERALLY. Every composed face publishes coverage rasters of
    its actual DRC-healed chrome (``literal_front``/``literal_back``, plus
    ``period_front`` where a halftone carries colour sub-gratings) and stamps
    ``recipe_data['literal']``. The manifest SHAPE changed — new ``files``
    keys and a new recipe_data key — and the payload PNGs do not exist beside
    a v14 manifest at all, so a warm slot must re-derive rather than serve a
    manifest whose renderer contract it cannot satisfy.
v16: the literal rasters become EXACT coverage. v15 filled the rings with a 2×
    supersampled PIL polygon draw, whose boundary-inclusive, phase-QUANTISED
    fill fattened every line by a whole sample: a 50%-duty carrier rastered at
    0.532 and the 5 µm colour bands at 0.71 with 46% of their texels pinned to
    255, i.e. the preview showed the colour zones as near-solid gold and the
    whole plate too heavy. ``literal_raster.layer_coverage`` now accumulates
    the analytic area instead (see that module). ``period_front`` picks each
    texel's band by overlap AREA rather than by paint order. Every
    ``literal_*``/``period_front`` PNG on disk is therefore wrong under a v15
    manifest and must be re-derived.
v17-v22: the PLATE FOR WRITING (2026-09-10) — one entry, because the six bumps
    were one design landing in stages and no cache survived any of them.
    (a) literal rasters exact, continued: the literal bake is the ACTUAL healed
    chrome for every face, so a composed face's PNGs and its ``files`` entries
    both move. (b) The lid's monogram carries NO diffraction accent — one
    shading moiré over the whole silhouette instead of two flourish-tip
    patches — so ``monogram-jp`` masks change on both layers. (c) SINGLE-PLY
    faces drop the carrier entirely (``photo.CARRIER_COV`` = 0): the front mask
    is the picture and the leaf garland on bare glass, the back stays empty,
    and the picture's edge fade now dissolves to glass rather than to a 50%
    field. (d) The leaf fill knob (``SINGLE_PLY_LEAF_FILL`` = "hue") and the
    period ladder arrive with the recipe key ``single_ply_leaf_period_um`` and
    a ``period_front`` map over the leaves. (e) The box's own dials move under
    every face: the art rim starts at the inner ply's window
    (``assembly.bonded_art_keepout_um``), the band is a fixed 2.4 mm, and the
    faces run ``carrier_scale_mode`` "fixed" at the eye-sized 65.5 µm carrier —
    all of which land in the frame geometry and in ``_carrier_recipe_data``.
v23: ``single_ply_leaf_period_um`` (recipe key AND ``period_front``) comes from
    ONE helper, ``single_ply_leaf_period_um(spec)``, and under the shipping
    "hue" fill that is the LADDER'S MEAN. v17-v22 advertised the 10 µm "lines"
    constant in the manifest while writing the 4.15-6.02 µm ladder, so the
    renderer drew a sheen at twice the period of the gold in front of it. The
    PNG masks are unchanged; recipe_data is not, and it is cached.
```

(v26 and later: see `plates.py`.)

## `plates.PLATE_SVG_VERSION`

Bumped when the fab SVG compose geometry changes (`ensure_plate_svg` /
`_bake_plate_svg`): a cached pair is only reused if it carries the current
marker. Strings, `plate-svg-vN`; v1-v3 were never written down.

```
v4: barrier-interlace front comb spans the full art box (was union-gated).
v5: coarse-bake honesty — effective periods stamped in the manifest, skipped
    diffraction-accent zones keep the surrounding carrier instead of becoming
    bare-glass holes, the capybara body carries only its 24 µm shimmer (no
    superimposed centerpiece carrier), and a degenerate water interleave keeps
    the plain carrier across the band instead of erasing it. Also in v5:
    barrier-interlace faces emit NO front diffraction accent at all (a
    silhouette-shaped front feature can never vanish under tilt), so the
    gear-quill-switch hub grating is gone from front.svg.
v6: barrier registration carried onto the plate path — the front comb and the
    back A|B lanes are cut from ONE cell-exact lattice (_barrier_plate_lattice /
    _barrier_masks) instead of two float-phased gratings that lose registration
    to sampling, so the baked comb period equals the baked interlace period and
    every open-slit centre lands on a channel boundary. Interlace face geometry
    shifts by up to a cell and the baked barrier period snaps to a multiple of
    four raster cells (recorded in svg_bake_barrier_*).
v7: the water scanimation bakes at the face's EFFECTIVE waterline
    (_water_waterline_y) instead of the module constant, so a face that sets
    the ``waterline`` param gets a band/wake matching its preview mask. Only
    such faces change; default-param geometry is byte-identical.
v8: the water band EXCLUDES the submerged capybara. The band the bake consumes
    (``_capybara_and_water``'s ``water_band``) is now ``below & ~capy`` — the
    composed preview's convention — so front.svg no longer lays slit-barrier
    bars across the animal, back.svg no longer interleaves ripple crests under
    it, and the plain back carrier is kept over the submerged body instead of
    being cleared for the band. Only the capybara face changes.
v12: production-box faces. A BLANK face bakes two empty documents; a
    SINGLE-PLY face bakes the carrier grating into front.svg (over the back
    window MINUS the art box) and an EMPTY back.svg; a PHOTO face bakes the
    line-screen bands and their colour sub-gratings as exact rectangles at the
    art box, front layer only — vector geometry that never passes through the
    coarse budget raster, so it is period-exact even here.
v13-v17: the PLATE FOR WRITING (2026-09-10), one entry for the five bumps of a
    single design. The frame band moves on EVERY face — the art rim now starts
    at the inner ply's window (``assembly.bonded_art_keepout_um``) and the band
    is a fixed 2.4 mm at motif_scale 0.68 — and the carrier bakes at the
    eye-sized 65.5 µm fixed pitch instead of the gap-scaled one. The lid bakes
    no diffraction accent (one shading moiré over the whole monogram), and a
    single-ply photo face bakes the leaf gratings of the chosen fill
    (``SINGLE_PLY_LEAF_FILL``) rather than a common-pitch louvre.
v18: a SINGLE-PLY face bakes NO carrier. v12 put a carrier grating in front.svg
    over the back window minus the art box; compose stopped painting it and
    ``export_fine.build_plate_fine`` never wrote it, so the fab SVG was the only
    path still emitting ~500 mm² of gold that the mask does not have — the
    exact drift "fab SVG = preview PNG" (CLAUDE.md) exists to catch. Only
    single-ply faces change; every other face's SVG is byte-identical.
```

(v20 and later: see `plates.py`.)
