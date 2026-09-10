from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .export_svg import to_svg
from .patterns.base import ParamSpec, registry
from .rasterize import (
    DEFAULT_THUMBNAIL_METAL,
    THUMBNAIL_PALETTES,
    make_thumbnail,
    rasterize,
)

if TYPE_CHECKING:
    from PIL.Image import Image


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

_log = logging.getLogger("optics.service")


# --- cache slot publishing ---------------------------------------------------
# Every on-disk cache slot (pattern variant, composed plate, box) publishes the
# same way: payload files first, then the manifest, each renamed into place with
# ``os.replace`` (atomic on NTFS). So the existence of a readable manifest means
# the slot is complete, and a crash mid-generate leaves at most a stray ``*.tmp``
# instead of a truncated manifest that used to poison the slot forever (every
# hit path did a bare ``json.loads`` and re-raised JSONDecodeError until someone
# wiped backend/data by hand). Readers go through ``read_json_cache``, which
# reports corruption as a plain miss so the next request regenerates over it.


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    """Publish a JSON cache file by rename, never in place."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent))
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    """Publish a text cache file (SVG) by rename, never in place."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_png_atomic(img: "Image", path: Path) -> None:
    """Publish a PNG by rename so no reader sees a half-written raster.

    ``format`` must be explicit — the staging name ends in ``.tmp``, so PIL
    cannot infer it from the extension.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def read_json_cache(path: Path) -> dict[str, Any] | None:
    """Load a cached manifest, or None if absent, unreadable, or not an object.

    A decode error is deliberately NOT an exception: the caller treats it as a
    cache miss and regenerates over the slot.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError
        _log.warning("cache manifest unreadable (treated as miss): %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


# --- per-slot compute locks --------------------------------------------------
# FastAPI runs these sync handlers in a threadpool, so two overlapping requests
# for the SAME cache id (the debounced live-preview regen) both missed and both
# ran the full compose into the identical directory — one thread publishing a
# manifest while the other was still rewriting the PNGs it points at, and a
# concurrent double compose is exactly the "two long compute processes at once"
# the machine constraint forbids (CLAUDE.md). Serializing on a per-id lock makes
# the loser wait and then read the finished slot.
_lock_registry_guard = threading.Lock()
_cache_locks: dict[str, threading.Lock] = {}
# Locks are a few dozen bytes each and the id space is bounded in practice; the
# cap only guards a long-lived server that walks a huge number of distinct
# hashes. Pruning an *idle* lock can race a thread that fetched the object but
# has not acquired it yet, whose worst case is one duplicated compute — the
# pre-lock behaviour — so it is only worth doing far past normal usage.
_CACHE_LOCK_MAX = 4096


def cache_lock(key: str) -> threading.Lock:
    """Process-wide lock for one cache id. Namespace the key (``plate:<hash>``)."""
    with _lock_registry_guard:
        lock = _cache_locks.get(key)
        if lock is None:
            if len(_cache_locks) >= _CACHE_LOCK_MAX:
                for k, held in list(_cache_locks.items()):
                    if not held.locked():
                        del _cache_locks[k]
            lock = threading.Lock()
            _cache_locks[key] = lock
        return lock


# --- global heavy-compute gate -----------------------------------------------
# The per-slot locks above only serialize duplicate work on the SAME cache id.
# This one serializes UNRELATED heavy work process-wide, because CLAUDE.md's
# machine constraint is not per-slot: a 13.7 GB host that has kernel-bugchecked
# under concurrent compute must never run two long computes at once, even when
# they are two different faces or two different patterns. The concrete offender
# was the pattern picker's boot sweep — sixteen full default generates fired
# from the frontend on mount, straight through a box regen.
#
# BoundedSemaphore(1) rather than Lock on purpose: it is re-entrancy hostile, so
# a nested acquire deadlocks loudly instead of quietly admitting a second heavy
# compute. That invariant is what keeps the gate honest — no materialize path may
# call another materialize path while holding it (which is why the plate
# compositor reads the central pattern's METADATA rather than materializing it;
# see plates._raster_compose_plate).
heavy_compute_gate = threading.BoundedSemaphore(1)


# Bump when the pattern GENERATION path changes its output under an unchanged
# param hash: any generator's geometry (``patterns/**``), the litho floor or
# other shared helpers in ``patterns/base.py``, the rasterizer / thumbnail
# composite, or the manifest shape below (``extra`` / ``recipe_data`` keys). The
# variant hash covers user params only, so without this a warm ``backend/data``
# serves pre-change PNGs and recipe_data forever. Same contract as
# ``plates.PLATE_COMPOSE_VERSION`` and ``plates.PLATE_SVG_VERSION``; see CLAUDE.md.
# v2: first versioned generation — the litho-floor raise and the slit-lattice
#     barrier registration changed geometry with no key to invalidate it.
# v3: capybara water band carved to `below & ~capy` (the submerged body keeps
#     its carrier; ripples never print on the animal) — polygons, measured
#     min_feature_um and the min_*_gold_um extras all move.
# v7: two new registered generators — ``blank`` (bare glass, empty layers) and
#     ``photo-halftone`` (a prepared photograph as a line screen with a
#     coverage-space edge fade and an optional colour period field). New slugs
#     alone would not need a bump, but the plate compositor now reads
#     ``photo-halftone``'s metadata on the box path and the manifest shape grows
#     the ``art_solid`` recipe_data key, so a warm ``backend/data`` must
#     re-derive rather than serve variants written before either existed.
PATTERN_GEN_VERSION = 9


def _params_hash(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def variant_key(merged_params: dict[str, Any]) -> str:
    """The on-disk variant id for a FULLY MERGED param dict.

    Public because ``plates._raster_compose_plate`` has to name a variant's
    extra-layer PNG URLs without materializing the variant (see
    ``Pattern.metadata``); the URL must be the one ``_materialize_locked``
    would publish, so both sides go through this and ``extra_layer_url``.
    """
    return _params_hash(merged_params)


def extra_layer_url(slug: str, variant: str, layer_name: str) -> str:
    """Servable URL of a variant's rasterized extra layer (``<name>.png``)."""
    return f"/data/{slug}/{variant}/{layer_name}.png"


def pattern_dir(slug: str, variant: str) -> Path:
    return DATA_ROOT / slug / variant


def _range_text(spec: ParamSpec) -> str:
    lo = "-inf" if spec.min is None else f"{spec.min:g}"
    hi = "inf" if spec.max is None else f"{spec.max:g}"
    return f"[{lo}, {hi}]{' ' + spec.unit if spec.unit else ''}"


def validate_params(slug: str, params: dict[str, Any]) -> None:
    """Reject params the pattern's ParamSpec list does not allow.

    ``min``/``max`` were UI-only hints until this ran server-side: a POST could
    push extent_um/period_um orders of magnitude past the declared range and
    the generator would size a numpy lattice straight off it — most generators
    have no ``check_lattice_budget`` call, so the failure mode was a host OOM,
    not a 400. Message style follows ``check_lattice_budget`` (state the limit,
    name the dial); the /patterns and /boxes routes surface it as an HTTP 400.
    """
    specs = {p.name: p for p in registry[slug].params}
    unknown = sorted(k for k in params if k not in specs)
    if unknown:
        raise ValueError(
            f"{slug}: unknown parameter(s) {', '.join(unknown)}. "
            f"This pattern accepts {', '.join(sorted(specs)) or '(none)'}."
        )

    for name, value in params.items():
        spec = specs[name]
        # The declared default is in bounds by definition — never fight a
        # pattern whose default sits on (or outside) its own declared range.
        if value == spec.default:
            continue
        if spec.type in ("float", "int"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{slug}.{name} must be a {spec.type} (got "
                    f"{type(value).__name__} {value!r}); allowed range is "
                    f"{_range_text(spec)}."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"{slug}.{name} must be a finite {spec.type} (got {value!r}); "
                    f"allowed range is {_range_text(spec)}."
                )
            if spec.type == "int" and value != int(value):
                raise ValueError(
                    f"{slug}.{name} must be a whole number (got {value!r}); "
                    f"allowed range is {_range_text(spec)}."
                )
            if (spec.min is not None and value < spec.min) or (
                spec.max is not None and value > spec.max
            ):
                raise ValueError(
                    f"{slug}.{name} = {value:g} is outside the declared range "
                    f"{_range_text(spec)}; pick a value inside it. Extents past "
                    "the max are never needed — the plate compositor upscales "
                    "the pattern raster to fill the aperture."
                )
        elif spec.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{slug}.{name} must be true or false (got "
                    f"{type(value).__name__} {value!r})."
                )
        elif spec.type == "choice":
            if spec.choices is not None and value not in spec.choices:
                raise ValueError(
                    f"{slug}.{name} must be one of {', '.join(spec.choices)} "
                    f"(got {value!r})."
                )


def catalog() -> list[dict]:
    return [cls.descriptor() for cls in registry.values()]


def list_variants(slug: str) -> list[str]:
    root = DATA_ROOT / slug
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _variant_stale_reason(cached: dict[str, Any], cls: type) -> str | None:
    """Why a cached variant manifest may not be served, or None if it is current."""
    if cached.get("gen_version") != PATTERN_GEN_VERSION:
        return f"gen_version {cached.get('gen_version')!r} != {PATTERN_GEN_VERSION}"
    if cached.get("render_recipe") != cls.render_recipe:
        return f"render_recipe {cached.get('render_recipe')!r} != {cls.render_recipe!r}"
    return None


def materialize(
    slug: str,
    params: dict[str, Any] | None = None,
    force: bool = False,
) -> dict:
    """Generate (or load from cache) a variant of a pattern and return its manifest."""
    if slug not in registry:
        raise KeyError(f"Unknown pattern: {slug}")

    cls = registry[slug]
    merged = {**cls.defaults(), **(params or {})}
    # Before the hash, so out-of-range params can neither mint a cache dir nor
    # be served from one that predates the bounds check.
    validate_params(slug, merged)
    variant = _params_hash(merged)
    # One generate per variant, process-wide: the second caller blocks and then
    # reads the finished slot instead of racing the same compute into the same
    # directory (see cache_lock). The global heavy_compute_gate is taken further
    # in, around the generate itself, so a warm cache hit never queues behind an
    # unrelated cold compute.
    with cache_lock(f"pattern:{slug}:{variant}"):
        return _materialize_locked(cls, slug, variant, merged, force)


def _materialize_locked(
    cls: type,
    slug: str,
    variant: str,
    merged: dict[str, Any],
    force: bool,
) -> dict:
    out = pattern_dir(slug, variant)
    manifest_path = out / "manifest.json"
    if not force:
        cached = read_json_cache(manifest_path)
        if cached is None:
            if manifest_path.exists():
                _log.info(
                    "materialize regenerate slug=%s variant=%s reason=unreadable_manifest",
                    slug,
                    variant,
                )
        else:
            reason = _variant_stale_reason(cached, cls)
            if reason is None:
                _log.info("materialize cache_hit slug=%s variant=%s", slug, variant)
                return cached
            _log.info(
                "materialize regenerate slug=%s variant=%s reason=%s", slug, variant, reason
            )

    # Everything below is the heavy path, so it runs under the process-wide
    # compute gate (CLAUDE.md machine constraint; see heavy_compute_gate). Taken
    # HERE rather than around the whole function so the cache-hit return above
    # never queues behind an unrelated cold generate.
    with heavy_compute_gate:
        t0 = time.perf_counter()
        gp = cls.generate(**merged)
        out.mkdir(parents=True, exist_ok=True)

        # Rasterize both layers
        front_png = rasterize(gp.front, gp.extent_um, gp.pixel_pitch_um)
        back_png = rasterize(gp.back, gp.extent_um, gp.pixel_pitch_um)
        save_png_atomic(front_png, out / "front.png")
        save_png_atomic(back_png, out / "back.png")

        # SVG is built on demand (see ensure_pattern_svg) — eager to_svg cost
        # ~4-8 s per cold variant and nothing at runtime ever fetched the files.
        # Any pair left over in this slot was vectorized from the geometry we just
        # replaced (this branch also runs on force and on a PATTERN_GEN_VERSION
        # bump), so drop it instead of advertising stale vectors.
        for stale_svg in (out / "front.svg", out / "back.svg"):
            stale_svg.unlink(missing_ok=True)

        # Recipe-specific extra layers (e.g. view_a / view_b for stereo_lenticular).
        # Each gets rasterized and its URL stamped into recipe_data[<name>_png] so
        # the frontend can load them without the generator doing path bookkeeping.
        extra_layer_urls: dict[str, str] = {}
        for layer_name, layer_poly in gp.extra_layers.items():
            layer_png = rasterize(layer_poly, gp.extent_um, gp.pixel_pitch_um)
            save_png_atomic(layer_png, out / f"{layer_name}.png")
            extra_layer_urls[f"{layer_name}_png"] = extra_layer_url(slug, variant, layer_name)

        # Thumbnail — one chip per litho metal, so the catalog tiles can match
        # the box's selected metal (the masks are metal-agnostic graylevel
        # codes; the palette is the chip's only colour choice). The GOLD output
        # is byte-identical to what this always wrote, so warm caches stay
        # valid and PATTERN_GEN_VERSION does not move; the extra chips are
        # additive, and the read-only thumbnail route composes them lazily for
        # variants cached before this existed.
        thumb = make_thumbnail(front_png, back_png, size=256)
        save_png_atomic(thumb, out / "thumbnail.png")
        for _metal in THUMBNAIL_PALETTES:
            if _metal == DEFAULT_THUMBNAIL_METAL:
                continue
            save_png_atomic(
                make_thumbnail(front_png, back_png, size=256, metal=_metal),
                out / f"thumbnail_{_metal}.png",
            )

        # Manifest — published last and by rename, so a readable manifest implies the
        # PNGs above are complete.
        manifest = {
            "slug": slug,
            "variant": variant,
            # Generation-code marker; a mismatch on the hit path is a miss (the
            # variant hash covers user params only). See PATTERN_GEN_VERSION.
            "gen_version": PATTERN_GEN_VERSION,
            "name": cls.name,
            "description": cls.description,
            "tags": cls.tags,
            "params": merged,
            "substrate": {
                "thickness_um": gp.substrate.thickness_um,
                "material": gp.substrate.material,
                "n": gp.substrate.n,
            },
            "extent_um": list(gp.extent_um),
            "pixel_pitch_um": gp.pixel_pitch_um,
            "min_feature_um": gp.min_feature_um,
            "extra": gp.extra,
            # Render-recipe hints for the frontend shader. The class-level default
            # is "stylized_amplitude" (flat-mask composite); patterns that want a
            # physics-correct recipe override Pattern.render_recipe on the class
            # and can stash per-variant file refs in GeneratedPattern.recipe_data.
            "render_recipe": cls.render_recipe,
            "recipe_data": {**gp.recipe_data, **extra_layer_urls},
            "files": {
                "front_png": f"/data/{slug}/{variant}/front.png",
                "back_png": f"/data/{slug}/{variant}/back.png",
                # Lazy (mirrors the plate manifest contract): empty until
                # ensure_pattern_svg builds the files on demand — which is always,
                # at this point, since the regenerate above cleared any stale pair.
                "front_svg": "",
                "back_svg": "",
                "thumbnail": f"/data/{slug}/{variant}/thumbnail.png",
            },
        }
        write_json_atomic(manifest_path, manifest)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        pixels = front_png.size[0] * front_png.size[1]
        _log.info(
            "materialize done slug=%s variant=%s pixels=%d pitch_um=%.3f %dms",
            slug,
            variant,
            pixels,
            gp.pixel_pitch_um,
            dt_ms,
        )
        return manifest


def ensure_pattern_svg(slug: str, variant: str) -> tuple[Path, Path] | None:
    """Lazily build the SVG pair for an existing pattern variant.

    Mirrors ``plates.ensure_plate_svg``: the materialize path skips the
    eager ``to_svg`` (it cost seconds per cold variant with zero runtime
    consumers); callers that genuinely want the authoritative vector form
    (fab/export tooling) hit this instead. Regenerates the polygons from the
    manifest's params (deterministic), writes front.svg/back.svg, stamps the
    URLs back into the manifest.

    Returns (front, back) paths or None if the variant is unknown (or its
    manifest is unreadable — the params to regenerate from live in it).

    Takes the variant's cache lock: the ``to_svg`` pass is seconds of compute and
    it rewrites the shared manifest, so it must not run beside a materialize of
    the same slot.
    """
    out = pattern_dir(slug, variant)
    manifest_path = out / "manifest.json"
    if slug not in registry or not manifest_path.exists():
        return None
    front_svg = out / "front.svg"
    back_svg = out / "back.svg"
    if front_svg.exists() and back_svg.exists():
        return front_svg, back_svg

    with cache_lock(f"pattern:{slug}:{variant}"):
        if front_svg.exists() and back_svg.exists():
            return front_svg, back_svg
        manifest = read_json_cache(manifest_path)
        if manifest is None:
            return None
        cls = registry[slug]
        gp = cls.generate(**{**cls.defaults(), **manifest.get("params", {})})
        write_text_atomic(front_svg, to_svg(gp.front, gp.extent_um))
        write_text_atomic(back_svg, to_svg(gp.back, gp.extent_um))

        files = manifest.setdefault("files", {})
        files["front_svg"] = f"/data/{slug}/{variant}/front.svg"
        files["back_svg"] = f"/data/{slug}/{variant}/back.svg"
        write_json_atomic(manifest_path, manifest)
    return front_svg, back_svg


def seed_defaults() -> list[dict]:
    """Materialize every registered pattern's default variant.

    NOT called at startup anymore (it cost minutes of cold wall clock before
    the first request could be answered) — the box/plate path materializes
    central patterns lazily with a disk cache. Kept as the explicit warm
    command (``just seed``).
    """
    return [materialize(slug) for slug in registry]
