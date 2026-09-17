"""Content fingerprints for the on-disk caches.

``backend/data/`` is keyed on the USER's spec: ``variant_key`` hashes pattern
params, ``plate_hash`` hashes a ``PlateSpec``. Neither covers the code that
turns those inputs into pixels, so a warm cache would serve pre-change PNGs
forever after any edit to a generator, a constant or the manifest shape. Three
hand-bumped integers used to close that hole — and 25 compose bumps and 19 SVG
bumps later, the pattern was clear: the discipline is the failure mode. Every
one of those bumps was a human remembering; every bug it ever caught was a
human forgetting.

So the markers are COMPUTED now. Each cache names the modules whose output it
holds — its *closure* — and its marker is a digest of their source, taken once
at import.

WHAT MOVES A FINGERPRINT, AND WHAT DOES NOT

The digest is over the normalized ``ast.dump`` of each module, not its bytes:

  * comments are not in the AST at all, so a comment is free;
  * ``include_attributes=False`` drops line and column numbers, so INSERTING a
    comment or a blank line above real code is free too — the usual reason a
    byte hash of the source is useless here;
  * every bare string expression (module, class and function docstrings, and
    the attribute docstrings this codebase writes under its constants) has its
    whitespace collapsed, so reflowing a paragraph is free while REWORDING one
    is not. Prose that documents a number is part of the number's meaning, but
    where the line breaks fall is not.

Anything else — a constant, an expression, a renamed local, a new keyword
argument — changes the dump and therefore the fingerprint, whether or not it
actually changes a pixel. That asymmetry is deliberate: a spurious regenerate
costs minutes of compute, a missed one ships the wrong mask.

ASSETS count too. A plate's photographs are inputs the same way its code is, so
the prepared photo directory (the PNGs, their subject/fade mattes and their
authored colour plans) is digested by content. Re-crop a photo and the caches
that read it invalidate themselves; no marker to remember.

``CACHE_EPOCH`` is the one manual escape hatch. Bump it to invalidate
everything at once — a cache-poisoning bug, a change in something outside every
closure (Pillow's resampling, a Shapely version) — and say why.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parent

CACHE_EPOCH = 1
"""Manual invalidation of EVERY cache. History:

  * 1 (2026-09-17) — the content fingerprints replace
    ``PATTERN_GEN_VERSION`` 10, ``PLATE_COMPOSE_VERSION`` 27 and
    ``PLATE_SVG_VERSION`` "plate-svg-v20". Every warm slot regenerates once.
"""


def _normalized_dump(path: Path) -> str:
    """A module's AST, with comments gone and docstring whitespace collapsed.

    ``ast.parse`` already drops comments; ``include_attributes=False`` drops the
    positions that would otherwise make a comment's LINE significant. What is
    left is the code's structure and its literal values — which is what the
    output depends on.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        # A bare string expression is documentation, never a value anything
        # reads: module/class/function docstrings, and the attribute docstrings
        # this codebase writes under a constant. Collapse their whitespace so a
        # reflow is free; keep the words, so a rewording is not.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            node.value.value = " ".join(node.value.value.split())
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _iter_paths(entries: Sequence[str]) -> list[Path]:
    """Resolve closure entries (``app``-relative paths or globs) to files.

    A literal path that does not exist is an ERROR, not an empty contribution:
    a typo'd closure entry would quietly stop covering the module it names, and
    the whole point of the fingerprint is that nothing quietly stops covering
    anything. Globs may legitimately match nothing changed... but here every
    glob is a package directory, so an empty one is a typo too.
    """
    out: list[Path] = []
    for entry in entries:
        if any(ch in entry for ch in "*?["):
            hits = sorted(APP_ROOT.glob(entry))
            if not hits:
                raise RuntimeError(f"cache closure glob matched nothing: {entry!r}")
            out.extend(hits)
        else:
            p = APP_ROOT / entry
            if not p.is_file():
                raise RuntimeError(f"cache closure entry is not a file: {entry!r}")
            out.append(p)
    # Deterministic, de-duplicated (a module may be named by a path AND a glob).
    return sorted(set(out))


def _iter_assets(dirs: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        root = APP_ROOT / d
        if not root.is_dir():
            raise RuntimeError(f"asset directory is missing: {d!r}")
        out.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    return out


def fingerprint(modules: Sequence[str], assets: Sequence[str] = ()) -> str:
    """The marker one cache stamps into its manifests.

    ``modules`` are ``app``-relative paths or globs (the cache's closure);
    ``assets`` are ``app``-relative directories digested by raw bytes. The
    result is ``"cf<epoch>-<12 hex>"`` — short enough to read in a manifest,
    wide enough that a collision is not a thing that happens.
    """
    h = hashlib.sha256()
    h.update(f"epoch={CACHE_EPOCH}\n".encode())
    for p in _iter_paths(modules):
        h.update(f"module {p.relative_to(APP_ROOT).as_posix()}\n".encode())
        h.update(_normalized_dump(p).encode())
        h.update(b"\n")
    for p in _iter_assets(assets):
        h.update(f"asset {p.relative_to(APP_ROOT).as_posix()}\n".encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return f"cf{CACHE_EPOCH}-{h.hexdigest()[:12]}"


def explain(modules: Sequence[str], assets: Sequence[str] = ()) -> list[str]:
    """The files a closure actually covers — for tests and for the operator who
    wants to know why a cache just regenerated."""
    names: Iterable[Path] = list(_iter_paths(modules)) + list(_iter_assets(assets))
    return [p.relative_to(APP_ROOT).as_posix() for p in names]


# --- the closures -------------------------------------------------------------
# Each is the set of modules whose source decides what one cache HOLDS. They are
# written out rather than derived from the import graph on purpose: the true
# transitive closure of any of these is most of the package (compose imports
# literal imports export_fine imports every generator), which would make every
# edit anywhere invalidate everything. What is listed is what actually writes
# the bytes.

PHOTO_ASSETS = ("assets/photos",)
"""The prepared photographs, their mattes and their authored colour plans —
inputs to every cache that screens a picture."""

PATTERN_GEN_CLOSURE = (
    "patterns/*.py",
    "patterns/**/*.py",
    "rasterize.py",
    "service.py",
)
"""The pattern GENERATION path: any generator's geometry, the shared helpers
and the litho floor in ``patterns/base.py``, the rasterizer and thumbnail
composite, and the manifest shape in ``service.py`` itself."""

PLATE_COMPOSE_CLOSURE = (
    "plates/spec.py",
    "plates/recipe.py",
    "plates/photo.py",
    "plates/compose.py",
    "plates/literal.py",
    "literal_raster.py",
    "region_art.py",
    "leaf_fills.py",
    "production.py",
)
"""What composes a plate: the mask palette and the paste, ``_carrier_recipe_data``
and every shader knob it emits, the literal rasters published beside the
preview PNGs, and the production constants all of them read."""

PLATE_SVG_CLOSURE = (
    "plates/spec.py",
    "plates/recipe.py",
    "plates/photo.py",
    "plates/compose.py",
    "plates/svg.py",
    "export_svg.py",
    "region_art.py",
    "production.py",
)
"""What bakes the fab SVG. It overlaps the compose closure because the rule is
that the two must draw the SAME geometry (CLAUDE.md): ``_bake_plate_svg``
reuses ``compose``'s masks, so a change there moves both markers."""
