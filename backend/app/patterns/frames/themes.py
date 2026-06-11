"""Frame themes — motif catalogue + tonal palette per theme.

Each theme is a plain dataclass. Adding a new theme = creating one config
object plus any new motif drawers it references. The colors are for the
*frontend preview* (the lithography mask is monochrome gold either way), but
they ride along in the serialized scene so the live preview renders
correctly without a separate lookup.

Motif weighting follows the spec: heavier weights = repeat the type string
multiple times in the list. The algorithm's ``rng.choice`` then naturally
biases.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameTheme:
    slug: str
    name: str
    eyebrow: str
    flowers: tuple[str, ...]
    leaves: tuple[str, ...]
    # Preview palette — used by the Canvas2D pen on the frontend, not by the
    # mask. ``gold``/``hi``/``dk`` are the 3-pass gold look; ``cream`` and
    # ``border`` are UI chrome hints.
    gold: str = "#d4a64a"
    hi: str = "#f4d97a"
    dk: str = "#7a5a1c"
    onyx_bg: tuple[str, str, str] = ("#15110a", "#0c0907", "#000000")
    ivory_bg: tuple[str, str, str] = ("#f5ecd6", "#e9ddb8", "#dccda0")


ESMERALDA = FrameTheme(
    slug="esmeralda",
    name="Esmeralda",
    eyebrow="Andean gold — Colombian flora",
    flowers=(
        # Orchids heavily weighted — they're the Colombian national flower.
        "orchid", "orchid", "orchid",
        "coffee", "coffee",
        "heliconia", "heliconia",
        "anthurium",
    ),
    leaves=(
        "wax_palm", "wax_palm",
        "plantain",
        "fern", "fern", "fern",
        "philodendron", "philodendron",
    ),
    gold="#d4a64a",
    hi="#f4d97a",
    dk="#7a5a1c",
)


THEMES: dict[str, FrameTheme] = {
    ESMERALDA.slug: ESMERALDA,
}


def get_theme(slug: str) -> FrameTheme:
    if slug not in THEMES:
        raise KeyError(f"Unknown frame theme: {slug}")
    return THEMES[slug]
