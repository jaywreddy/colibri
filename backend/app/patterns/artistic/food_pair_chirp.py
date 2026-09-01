from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..effects.gratings import (
    beat_delta_um,
    chirped_grating,
    clip_mask,
    linear_grating_mask,
    shimmer_moire_layers,
)
from ..motifs.lab.food_pair import food_bodies_silhouette, food_steam_silhouette


def _carriers(
    period_um: float,
    steam_base_period_um: float,
    tip_period_um: float,
    extent_um: float,
):
    """``(steam_chirp, body_carrier)`` grating results for these params.

    The STEAM CHIRP is the authoritative raster grid: it carries the finest
    (sub-5 µm) features, so ``chirped_grating`` picks the pitch that samples the
    tip period at ≥4 samples/period AND fits the 400k-lattice budget
    (auto-coarsening the whole chirp together if the requested extent+period would
    overflow). The body carrier is then built at that SAME pitch so the masks
    compose 1:1 with no resampling.

    Both are cheap (≤400k-cell numpy masks) and this is the ONLY place the
    EFFECTIVE post-coarsen periods and the pitch come from, so ``generate`` and
    the metadata accessors cannot report different numbers.

    Chirp axis: angle_deg=90 → HORIZONTAL grating lines whose period varies along
    the vertical (steam-rise) axis. ``chirped_grating`` sweeps period from
    period_start at the extent's low edge to period_end at the high edge. Our grid
    is y-UP, so the high edge is the steam TIPS → the FINE period goes at the end
    (tips), COARSE at the start (cup mouth).
    """
    extent = (extent_um, extent_um)
    steam_g = chirped_grating(
        period_start_um=steam_base_period_um,  # coarse — bottom (cup mouth)
        period_end_um=tip_period_um,           # fine — top (steam tips)
        extent_um=extent,
        angle_deg=90.0,
        coarsen=True,
    )
    body_g = linear_grating_mask(
        period_um, extent, angle_deg=0.0, pitch_um=steam_g.pitch_um, coarsen=True
    )
    return steam_g, body_g


@register
class FoodPairChirp(Pattern):
    """Coffee cup + arepa with a CHIRPED-steam shimmer wave (front-only glimmer).

    The couple's shared table — Colombian coffee and an arepa. The centerpiece
    is a steaming coffee cup on a saucer beside an arepa on a plate. The FRONT
    face carries the whole scene carved into a gold stripe carrier; the BACK
    face is left plain glass, so the piece *glimmers* front-only as it tilts
    (same treatment as the J+P monogram) rather than switching to a second
    image.

    The steam is the showpiece. Where the cup/arepa BODIES take a uniform
    stripe carrier, the three rising STEAM ribbons take a **chirped** carrier
    whose period sweeps from coarse at the cup mouth to fine at the tips (built
    with ``effects.gratings.chirped_grating``). Because the local stripe period
    grows toward the base, substrate parallax walks each band of the chirp by a
    different phase fraction per degree of tilt — so a small tilt sends a
    shimmer WAVE travelling UP the steam (the fine tip bands light and dim
    first, the coarse base bands lag), reading as curling, rising steam.

    LITHO / FAB (2 µm process, 4 µm min period):
      * Body carrier and the COARSE end of the steam chirp sit at ≥ ~12 µm —
        pure shimmer, no colour.
      * The steam chirp sweeps down to a reserved finest zone at the TIPS
        (``tip_period_um`` default 4.5 µm, clamped to the 4 µm floor). Below
        ~5 µm a single grating throws a visible first-order rainbow, so this tip
        band DOUBLES AS A DESIGNED DIFFRACTION ACCENT — the steam tips flash
        iridescent at ~5–8° tilt. See ``effects.moire.diffraction_onset``.
      * All periods are held ≥ ``MIN_PERIOD_UM`` (4 µm) by the chirp generator's
        litho-floor guard; ``chirped_grating`` also auto-coarsens against the
        400k-lattice budget.

    INTEGRATOR NOTE (plates.py / shader):
      * This pattern bakes a SPATIALLY-VARYING front carrier directly into the
        front polygons (uniform on bodies, chirped on steam) — it does NOT rely
        on the shader to synthesise the carrier. When wiring the box FACE, add a
        ``'food-pair-chirp'`` branch to ``plates._centerpiece_masks`` that
        returns the pre-baked front mask (bodies∪steam carrier) as the front
        silhouette and an EMPTY back (front-only glimmer, exactly like
        ``'monogram-jp'``). If the compositor instead re-synthesises a single
        carrier over the whole silhouette, the chirp wave is lost — so prefer
        baking these front polygons through ``ensure_plate_svg`` as-is.
      * recipe_data marks the diffraction-accent zone (steam tips) via
        ``diffraction_zone`` for any per-zone shader accent the integrator adds.
    """

    slug = "food-pair-chirp"
    name = "Coffee + arepa (chirped-steam shimmer)"
    description = (
        "A steaming cup of Colombian coffee beside an arepa. The cup and arepa "
        "bodies carry a uniform gold stripe carrier; the rising steam ribbons "
        "carry a chirped carrier — coarse at the cup mouth, sweeping to a fine "
        "sub-5 µm band at the tips — so a tilt sends a shimmer wave curling up "
        "the steam and the tips flash a faint diffraction rainbow. Back face "
        "left plain glass, so the scene glimmers front-only as the piece rocks."
    )
    tags = ["chirp", "tilt-shimmer", "diffraction", "Colombia", "coffee"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Body carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("steam_base_period_um", "Steam base period", "float", 22.0, 8.0, 80.0, 0.5, "μm"),
        ParamSpec("tip_period_um", "Steam tip period", "float", 4.5, 4.0, 12.0, 0.25, "μm"),
        # Default extent kept small on purpose: the steam tips ride a sub-5 µm
        # band, and honoring that period at ≥4 samples/period within the 400k
        # lattice budget caps the extent near ~700 µm. The plate compositor
        # upscales this raster to fill the aperture, so a compact extent keeps
        # the diffraction accent REAL rather than coarsened away. Larger extents
        # remain selectable but will auto-coarsen the chirp (accent lost).
        ParamSpec("extent_um", "Extent", "float", 640.0, 400.0, 2000.0, 20.0, "μm"),
        # Shading-moiré band spacing; see monogram_jp / shimmer_moire_layers.
        ParamSpec("beat_um", "Moiré band spacing", "float", 1635.0, 200.0, 8000.0, 5.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 22.0,
        steam_base_period_um: float = 22.0,
        tip_period_um: float = 4.5,
        extent_um: float = 640.0,
        beat_um: float = 1635.0,
    ) -> float:
        steam_g, _body_g = _carriers(
            period_um, steam_base_period_um, tip_period_um, extent_um
        )
        return steam_g.pitch_um

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 22.0,
        steam_base_period_um: float = 22.0,
        tip_period_um: float = 4.5,
        extent_um: float = 640.0,
        beat_um: float = 1635.0,
    ) -> float:
        # Half the finest EFFECTIVE period anywhere in the front layer: both
        # chirp ends AND the body carrier, all at 50% duty. Omitting the body
        # let a body period finer than the steam chirp advertise a coarser
        # minimum than the geometry actually contains — and this number is
        # what GeneratedPattern checks against the 2 µm litho floor. The
        # slider floor (tip 4 µm) is exactly 2 µm line / 2 µm gap, so no
        # legal combination goes sub-floor.
        steam_g, body_g = _carriers(
            period_um, steam_base_period_um, tip_period_um, extent_um
        )
        return (
            min(
                steam_g.meta["period_end_um"],
                steam_g.meta["period_start_um"],
                body_g.period_um,
            )
            * 0.5
        )

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 22.0,
        steam_base_period_um: float = 22.0,
        tip_period_um: float = 4.5,
        extent_um: float = 640.0,
        beat_um: float = 1635.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        steam_g, body_g = _carriers(
            period_um, steam_base_period_um, tip_period_um, extent_um
        )
        # Effective (post-coarsen) steam periods for the fab record.
        steam_tip_eff = steam_g.meta["period_end_um"]
        steam_base_eff = steam_g.meta["period_start_um"]
        return (
            # Carrier period + axis are informational only, kept out of
            # recipe_data on purpose: the Pattern Lab zone UI offers tilt
            # quick-sets whenever recipe_data carries a period key, and with an
            # empty back layer there is no mask-level tilt effect to quick-set
            # to.
            {
                "body_period_um": body_g.period_um,
                "carrier_period_um": body_g.period_um,
                "switch_axis_deg": 0.0,
                "steam_base_period_um": steam_base_eff,
                "steam_tip_period_um": steam_tip_eff,
                "coarsened": bool(steam_g.coarsened or body_g.coarsened),
            },
            {
                # Steam chirp span for any per-zone shader shimmer the integrator
                # wires; the fine end is the diffraction-accent zone. (No
                # carrier period key here — see the extra note above.)
                "chirp_period_start_um": steam_base_eff,
                "chirp_period_end_um": steam_tip_eff,
                "chirp_axis_deg": 90.0,
                # Marks the reserved sub-5 µm diffraction-accent band (steam
                # tips). Integrator can drive a rainbow-flash accent here.
                "diffraction_zone": "steam_tips",
                "diffraction_period_um": steam_tip_eff,
            },
            (),
        )

    @classmethod
    def generate(
        cls,
        period_um: float = 22.0,
        steam_base_period_um: float = 22.0,
        tip_period_um: float = 4.5,
        extent_um: float = 640.0,
        beat_um: float = 1635.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        steam_g, body_g = _carriers(
            period_um, steam_base_period_um, tip_period_um, extent_um
        )
        steam_carrier = steam_g.mask
        cell_um = steam_g.pitch_um
        h_px, w_px = steam_carrier.shape
        n_grid = w_px  # square extent → square grid

        bodies = food_bodies_silhouette(extent, n_grid=n_grid)
        steam = food_steam_silhouette(extent, n_grid=n_grid)

        # SHADING MOIRE on the bodies: the back carries a full-field carrier and
        # the cup + arepa are filled parallel to it at a mismatched pitch, so
        # bands sweep through them on tilt instead of the old front-only
        # glimmer. See monogram_jp for why the beat comes from pitch rather than
        # from a crossing angle on a build with no backside alignment.
        front_bodies, back_mask = shimmer_moire_layers(
            bodies,
            back_period_um=period_um,
            delta_um=beat_delta_um(period_um, beat_um),
            cell_um=cell_um,
        )
        # The steam keeps its CHIRP (that is its whole effect, and the sub-5 µm
        # tips are the diffraction accent). It now sits over the same back
        # carrier, so the plume beats too — at a spacing that varies along its
        # length, because one of the two pitches is swept.
        front_steam = clip_mask(steam_carrier, steam)
        front_mask = front_bodies | front_steam

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            period_um=period_um,
            steam_base_period_um=steam_base_period_um,
            tip_period_um=tip_period_um,
            extent_um=extent_um,
            beat_um=beat_um,
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )
