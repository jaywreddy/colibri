from __future__ import annotations

from shapely import affinity
from shapely.geometry import Point
from shapely.ops import unary_union

from .._helpers import crop, linear_grating, ring_grating
from ..base import GeneratedPattern, ParamSpec, Pattern, ensure_multipolygon, register


@register
class TaironaTalbot(Pattern):
    slug = "tairona-talbot"
    name = "Tairona Talbot revival"
    description = (
        "Interior: a linear Ronchi ruling whose Talbot distance "
        "z_T = 2·Λ²·n/λ places a textbook Talbot carpet in the air past the "
        "back face. Outer annulus: concentric Tairona goldwork rings as a "
        "decorative frame — the artistic motif, kept outside the active "
        "propagation region so the (x, z) revival structure is clean."
    )
    tags = ["talbot", "self-imaging", "diffraction", "Tairona"]
    tier = 3
    theme = "Colombia"
    render_recipe = "near_field_carpet"
    params = [
        ParamSpec("period_um", "Period", "float", 20.0, 8.0, 80.0, 0.5, "μm"),
        ParamSpec("duty", "Duty cycle", "float", 0.5, 0.1, 0.9, 0.05),
        ParamSpec("wavelength_um", "Design wavelength", "float", 0.55, 0.4, 0.8, 0.005, "μm"),
        ParamSpec("back_phase_shift", "Back phase shift", "choice", "zero",
                  choices=["zero", "half"]),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 20.0,
        duty: float = 0.5,
        wavelength_um: float = 0.55,
        back_phase_shift: str = "zero",
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Interior = linear Ronchi ruling (the actual Talbot source).
        # Frame = Tairona ring grating, masked to an outer annulus so the
        #   interior reads as "pure stripes" for Talbot physics while the
        #   edge keeps the Colombia-goldwork aesthetic.
        frame_inner_r = 0.80 * extent_um / 2
        interior_half = frame_inner_r  # where linear stripes live
        interior_extent = (interior_half * 2, interior_half * 2)

        ruling = linear_grating(period_um, duty, interior_extent)
        # ring_grating(...) then intersect with (outer - inner) annulus to
        # keep only the decorative border.
        rings_full = ring_grating(period_um, duty, extent)
        outer_disk = Point(0, 0).buffer(extent_um / 2, quad_segs=128)
        inner_disk = Point(0, 0).buffer(frame_inner_r, quad_segs=128)
        frame_annulus = outer_disk.difference(inner_disk)
        rings_frame = ensure_multipolygon(rings_full.intersection(frame_annulus))

        front_geom = unary_union([ruling, rings_frame])
        front = crop(ensure_multipolygon(front_geom), extent)
        back = crop(ensure_multipolygon(unary_union([ruling, rings_frame])), extent)
        if back_phase_shift == "half":
            back = ensure_multipolygon(affinity.translate(back, xoff=period_um / 2))
            back = crop(back, extent)

        n = 1.46
        z_T_um = 2 * period_um**2 * n / wavelength_um
        plate_over_z_T = 500.0 / z_T_um
        # Carpet sweeps from the back face through 2·z_T so the revival at z=z_T
        # sits at the midpoint of the z-slider (easy target for the vision check
        # and the E2E test).
        z_min_um = 0.0
        z_max_um = 2.0 * z_T_um
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            # Finer sampling than period*duty/8 so the carpet propagator has
            # headroom when downsampled (Phase G Nyquist fix).
            pixel_pitch_um=max(0.25, period_um / 16),
            min_feature_um=period_um * duty,
            extra={
                "talbot_distance_um": z_T_um,
                "plate_over_z_T": plate_over_z_T,
            },
            recipe_data={
                "talbot_distance_um": z_T_um,
                "design_wavelength_um": wavelength_um,
                "z_min_um": z_min_um,
                "z_max_um": z_max_um,
                "n_slices": 64,
                # Stripe layout: atlas is a single 2D x-z image (one row per
                # z-slice, each row = centerline x-cut). The shader samples
                # it as a texture; the SecondaryView renders it directly as
                # the canonical Talbot carpet diagram.
                "carpet_layout": "stripe",
            },
        )
