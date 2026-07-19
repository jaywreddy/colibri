from __future__ import annotations

import numpy as np

from .._helpers import raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs.lab.gear_quill import gear_silhouette, quill_book_silhouette


@register
class GearQuillSwitch(Pattern):
    """Gear ↔ quill+book tilt switch — "the engineer and the historian".

    A direct sibling of :class:`ColibriGlobePhase`: two silhouettes are carved
    into a single high-frequency vertical stripe carrier with a half-period
    phase offset between the faces. The FRONT face holds the GEAR (his
    engineering) at carrier phase 0; the BACK face holds the open BOOK + QUILL
    (her history) at carrier phase π — physically shifted by Λ/2. Snell-refracted
    parallax through the 500 µm substrate walks the back layer by a fraction of
    the stripe period as the piece tilts; the sign of the walk biases which
    carrier phase the eye samples, so tilting one way cleanly reveals the gear
    and the other way reveals the quill+book, while head-on the two interlace
    into a fine line pattern with both half-visible at once.

    Wiring mirrors ``colibri-globe-phase`` exactly (recipe ``phase_shift_overlay``,
    vertical carrier → horizontal switch axis). In a box the plate compositor
    dispatches the paired silhouettes through ``plates._centerpiece_masks``;
    this standalone ``generate`` bakes them so the pattern registers in the
    catalog and the ``/patterns`` endpoint can preview it. INTEGRATOR NOTE: add a
    ``'gear-quill-switch'`` branch to ``plates._centerpiece_masks`` returning
    ``(gear_silhouette, quill_book_silhouette)`` (front, back), matching the
    ``'colibri-globe-phase'`` branch.
    """

    slug = "gear-quill-switch"
    name = "Gear ↔ quill phase-shift switch"
    description = (
        "A toothed gear (the engineer) and an open book with a quill pen (the "
        "historian), each carved into a fine gold stripe carrier at opposite "
        "carrier phases — front gear at phase 0, back quill+book at phase π. "
        "Snell parallax through the fused-silica substrate slides the back layer "
        "by a fraction of the stripe period as the piece tilts, so one motif "
        "emerges while the other recedes; head-on, the two interlace and both "
        "read at once. A tilt-reveal portrait of the couple's two crafts."
    )
    tags = ["moire", "phase-shift", "tilt-reveal", "engineer", "historian"]
    tier = 1
    theme = "Colombia"
    render_recipe = "phase_shift_overlay"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 20.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # ≥8 samples/period so the half-period phase offset between front and
        # back is geometrically clean (identical rule to ColibriGlobePhase).
        n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
        cell_um = extent_um / n_grid

        gear = gear_silhouette(extent, n_grid=n_grid)
        quill = quill_book_silhouette(extent, n_grid=n_grid)

        # BARRIER INTERLACE (Task 3). Both motifs live on the BACK layer,
        # interleaved in alternating lanes (lane pitch = half the barrier period):
        # the gear (the engineer) in the even lanes, the book+quill (the historian)
        # in the odd. The FRONT layer is a NEUTRAL slit barrier (open duty 0.5 =
        # one lane) over the union — no image, just the gate. Parallax slides the
        # barrier across the lanes on tilt, so one tilt shows ONLY the gear and the
        # other ONLY the quill (a hard swap). Barrier phase −0.25 straddles an A|B
        # lane boundary head-on for symmetric ± reveal.
        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        lane = np.floor(cols / (period_pix / 2.0)).astype(int)
        even_lane = (lane % 2 == 0)[None, :]
        union = gear | quill
        interleave_back = (gear & even_lane) | (quill & ~even_lane)
        frac = (cols / period_pix) % 1.0
        barrier_bar = ((frac >= 0.25) & (frac < 0.75))[None, :]

        front_mask = union & barrier_bar
        back_mask = interleave_back

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * 0.5,
            extra={
                "carrier_period_um": period_um,
                "phase_offset_pix": period_pix / 2.0,
            },
            recipe_data={
                # Vertical stripes → horizontal switch axis (+X), same as the
                # colibrí↔globe pair the shader already handles.
                "switch_axis_deg": 0.0,
                "carrier_period_um": period_um,
            },
        )
