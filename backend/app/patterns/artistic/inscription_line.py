from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import inscription
from ..effects.gratings import beat_delta_um, shimmer_moire_layers


def _resolve_grid(period_um: float, extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the carrier raster.

    ≥8 samples/period so the carrier phase offset is geometrically clean.
    Module-level because both ``generate`` and ``pixel_pitch_um`` need it and the
    plate compositor reads the pitch WITHOUT generating — one expression, so they
    cannot disagree.
    """
    n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
    return n_grid, extent_um / n_grid


@register
class InscriptionLine(Pattern):
    """Hidden cursive inscription for the box bottom — front-only shimmer.

    A single centred line of formal script (Great Vibes copperplate) carved
    into a fine gold stripe carrier on the FRONT face; the BACK face is left
    empty (plain glass — inside a box the plate compositor puts the uniform
    back carrier behind it), so the line *glimmers* as the box is tilted
    (front-only glimmer, same treatment as the J+P monogram lid) rather than
    switching to a second image. This is the private line the couple reads when
    they lift the box — engraved the way a jeweller inscribes the inside of a
    band.

    The inscription text is composed from EDITABLE parts so the date can be
    changed later without a code edit:

      * ``initials``  — the couple's initials block (choice; default "J & P")
      * ``separator`` — the glyph set between initials and year (choice)
      * ``year``      — the engagement year (int; the editable date knob)

    They assemble to e.g. ``"J & P · 2026"``. The UI (FacesPanel →
    ParameterPanel) renders int params as a slider and choice params as a
    dropdown from ``ParamSpec.choices``, so ``year`` is editable today with no
    frontend change. (See the module note on adding a true free-text ``text``
    param type end-to-end if arbitrary strings are ever needed.)

    In a box, the plate compositor dispatches the silhouette through
    ``plates._centerpiece_masks`` (front = inscription, back = empty) and draws
    the grating procedurally in the shader; this standalone ``generate`` exists
    so the pattern registers in the catalog and ``/patterns`` can preview it. It
    is a front-only stripe-carrier shimmer (moire_interactive), the same
    single-layer treatment as the monogram.
    """

    slug = "inscription-line"
    name = "Hidden inscription (J & P · 2026)"
    description = (
        "A single centred line of formal cursive script — the couple's "
        "initials and their year, set the way a jeweller inscribes the inside "
        "of a band (Great Vibes copperplate with a small heart flourish). "
        "Carved into a fine gold stripe carrier so the line shimmers with a "
        "moiré highlight as the box is tilted, the back face left empty "
        "(plain glass; inside a box the plate compositor puts the uniform "
        "back carrier behind it) so it reads as a front-only glimmer. Meant "
        "for the hidden box bottom; the year is an editable pattern parameter."
    )
    tags = ["inscription", "engagement", "cursive", "tilt-shimmer", "bottom", "hidden"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"

    # NOTE on editability: ParamSpec.type is one of float|int|bool|choice
    # (backend app/patterns/base.py) and the frontend ParameterPanel renders
    # exactly those. There is no free-text ``string`` type, so the DATE is
    # exposed as an int (``year``) — a plain slider in the UI — and the
    # surrounding text as ``choice`` dropdowns. This keeps the inscription fully
    # editable end-to-end today. To allow ARBITRARY inscription text later, add
    # a ``"text"`` case to ParamSpec + a TextRow in the frontend ParameterPanel
    # and pass it straight through to ``inscription.inscription_silhouette``'s
    # ``text`` argument — the motif already accepts a free string.
    # ``♥`` is NOT a Great Vibes glyph (it renders as a tofu box), so it is a
    # *sentinel*: compose_text turns it into a wide blank gap and the motif's
    # own drawn-heart flourish fills that gap. Every other separator is a real
    # glyph in the face. Initials are likewise real-glyph strings only.
    HEART_SEP = "♥"

    params = [
        ParamSpec(
            "initials", "Initials", "choice", "J & P",
            choices=["J & P", "J + P", "J and P", "JP"],
        ),
        ParamSpec(
            "separator", "Separator", "choice", "·",
            choices=["·", "♥", "—", "•", "of"],
        ),
        ParamSpec("year", "Year", "int", 2026, 2000, 2099, 1),
        ParamSpec("heart", "Heart flourish", "bool", True),
        ParamSpec("period_um", "Carrier period", "float", 22.0, 6.0, 80.0, 0.5, "μm"),
        # Shading-moiré band spacing; see monogram_jp / shimmer_moire_layers.
        ParamSpec("beat_um", "Moiré band spacing", "float", 1635.0, 200.0, 8000.0, 5.0, "μm"),
        ParamSpec("extent_um", "Extent", "float", 3000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def compose_text(cls, initials: str, separator: str, year: int) -> str:
        """Assemble the inscription string from its editable parts.

        Kept as a classmethod so the plate compositor (which renders the
        silhouette directly, bypassing ``generate``) can build the identical
        string from a face's ``pattern_params``. The ``♥`` separator becomes a
        wide blank gap into which the drawn-heart flourish lands.
        """
        sep = str(separator).strip()
        if sep == cls.HEART_SEP:
            # Wide gap for the drawn heart (no ♥ glyph in the face).
            joiner = "     "
        elif sep:
            # A word separator ("of") reads with normal spaces; a glyph
            # separator ("·") wants the elegant thin gap it already implies.
            joiner = f" {sep} "
        else:
            joiner = "  "
        return f"{initials}{joiner}{int(year)}"

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        initials: str = "J & P",
        separator: str = "·",
        year: int = 2026,
        heart: bool = True,
        period_um: float = 22.0,
        extent_um: float = 3000.0,
        beat_um: float = 1635.0,
    ) -> float:
        return _resolve_grid(period_um, extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        initials: str = "J & P",
        separator: str = "·",
        year: int = 2026,
        heart: bool = True,
        period_um: float = 22.0,
        extent_um: float = 3000.0,
        beat_um: float = 1635.0,
    ) -> float:
        return period_um * 0.5

    @classmethod
    def extra_metadata(
        cls,
        initials: str = "J & P",
        separator: str = "·",
        year: int = 2026,
        heart: bool = True,
        period_um: float = 22.0,
        extent_um: float = 3000.0,
        beat_um: float = 1635.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        # Informational only — kept out of recipe_data on purpose: the
        # Pattern Lab zone UI offers tilt quick-sets whenever recipe_data
        # carries a period key, and with an empty back layer there is no
        # mask-level tilt effect to quick-set to.
        return (
            {
                "carrier_period_um": period_um,
                "switch_axis_deg": 0.0,
                "text": cls.compose_text(initials, separator, year),
            },
            {},
            (),
        )

    @classmethod
    def generate(
        cls,
        initials: str = "J & P",
        separator: str = "·",
        year: int = 2026,
        heart: bool = True,
        period_um: float = 22.0,
        extent_um: float = 3000.0,
        beat_um: float = 1635.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        text = cls.compose_text(initials, separator, year)
        # The ♥ separator always wants the drawn heart in its gap.
        want_heart = bool(heart) or str(separator).strip() == cls.HEART_SEP

        n_grid, cell_um = _resolve_grid(period_um, extent_um)

        # Rect-count estimate: raster_to_polygons emits one rect per horizontal
        # run, and the carrier chops every script row into ~extent/period runs.
        # Worst case (silhouette covering the full grid) is n_grid rows ×
        # stripes-per-extent — gate before the (extent/period)-sized text
        # raster. NOTE the defaults (4000 µm / 20 µm) land at EXACTLY the 400k
        # cap, so this extent cannot be raised without also capping n_grid the
        # way monogram-carrier-reveal does (min(1200, ...)).
        n_stripes = int(math.ceil(extent_um / period_um))
        # BOTH layers are gratings now, and the back one spans the full field
        # rather than being glyph-limited — count the pair.
        check_lattice_budget(
            2 * n_grid * n_stripes,
            "inscription-line carrier pair",
            period_um=period_um,
            beat_um=beat_um,
            extent_um=extent_um,
        )

        line = inscription.inscription_silhouette(
            extent, n_grid=n_grid, text=text, heart=want_heart
        )

        # SHADING MOIRE, not a front-only glimmer: the back carries a uniform
        # carrier of its own and the script is filled parallel to it at a
        # mismatched pitch, so bands sweep along the line as the box tilts.
        # See monogram_jp for why the beat comes from pitch, not from a crossing
        # angle, on a build with no backside alignment.
        front_mask, back_mask = shimmer_moire_layers(
            line,
            back_period_um=period_um,
            delta_um=beat_delta_um(period_um, beat_um),
            cell_um=cell_um,
        )

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            initials=initials,
            separator=separator,
            year=year,
            heart=heart,
            period_um=period_um,
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
