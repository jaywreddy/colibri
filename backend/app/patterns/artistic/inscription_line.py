from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import inscription


@register
class InscriptionLine(Pattern):
    """Hidden cursive inscription for the box bottom — front-only shimmer.

    A single centred line of formal script (Great Vibes copperplate) carved
    into a fine gold stripe carrier on the FRONT face; the BACK face is left as
    plain carrier, so the line *glimmers* against the back as the box is tilted
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
    re-uses the same ``phase_shift_overlay`` carrier scheme as the monogram for
    a consistent front-only shimmer.
    """

    slug = "inscription-line"
    name = "Hidden inscription (J & P · 2026)"
    description = (
        "A single centred line of formal cursive script — the couple's "
        "initials and their year, set the way a jeweller inscribes the inside "
        "of a band (Great Vibes copperplate with a small heart flourish). "
        "Carved into a fine gold stripe carrier so the line shimmers with a "
        "moiré highlight as the box is tilted, the back face left as plain "
        "carrier so it reads as a front-only glimmer. Meant for the hidden box "
        "bottom; the year is an editable pattern parameter."
    )
    tags = ["inscription", "engagement", "cursive", "tilt-shimmer", "bottom", "hidden"]
    tier = 1
    theme = "Colombia"
    render_recipe = "phase_shift_overlay"

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
        ParamSpec("period_um", "Carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 4000.0, 500.0, 5000.0, 100.0, "μm"),
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

    @classmethod
    def generate(
        cls,
        initials: str = "J & P",
        separator: str = "·",
        year: int = 2026,
        heart: bool = True,
        period_um: float = 20.0,
        extent_um: float = 4000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        text = cls.compose_text(initials, separator, year)
        # The ♥ separator always wants the drawn heart in its gap.
        want_heart = bool(heart) or str(separator).strip() == cls.HEART_SEP

        # >=8 samples/period so the carrier phase offset is geometrically clean.
        n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
        cell_um = extent_um / n_grid

        line = inscription.inscription_silhouette(
            extent, n_grid=n_grid, text=text, heart=want_heart
        )

        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        # Vertical stripe carrier at phase 0 on the front.
        phase_front = (cols % period_pix) < (period_pix / 2.0)

        front_mask = line & phase_front[None, :]
        # Back is plain glass — the inscription glimmers front-only.
        back_mask = np.zeros_like(front_mask)

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            extra={"carrier_period_um": period_um, "text": text},
            recipe_data={
                "switch_axis_deg": 0.0,
                "carrier_period_um": period_um,
            },
        )
