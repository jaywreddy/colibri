"""What moves a cache fingerprint, and what must not.

The three hand-bumped markers (``PATTERN_GEN_VERSION`` 10,
``PLATE_COMPOSE_VERSION`` 27, ``PLATE_SVG_VERSION`` "plate-svg-v20") are
computed now — see ``app/cache_fingerprint.py``. The contract has two halves
and both are load-bearing:

  * a COMMENT or a reflowed docstring must be free, or nobody will document
    anything without first weighing a full cache rebuild against it;
  * a CONSTANT or an expression must not be free, because that is the entire
    reason the markers exist.

The tests below edit copies in ``tmp_path`` rather than the real modules — the
fingerprint reads files off disk, so editing a real one would be a very
effective way to invalidate the developer's own warm cache.

``tests/test_cache_integrity.py`` owns the other half: what a STALE marker does
to a warm slot.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import cache_fingerprint as cf

SOURCE = '''"""A module docstring.

With a second paragraph that wraps
across two lines.
"""
# a leading comment
FLOOR_UM = 2.0
"""The attribute docstring this codebase writes under a constant."""


def area(w: float, h: float) -> float:
    """Multiply."""
    # an inline comment
    return w * h
'''


def _dump(tmp_path: Path, text: str, name: str = "m.py") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return cf._normalized_dump(p)


def test_comments_do_not_change_the_dump(tmp_path: Path) -> None:
    base = _dump(tmp_path, SOURCE)
    edited = SOURCE.replace("# a leading comment", "# a MUCH longer comment\n# on two lines")
    edited = edited.replace("# an inline comment", "")
    assert _dump(tmp_path, edited, "n.py") == base


def test_blank_lines_and_position_do_not_change_the_dump(tmp_path: Path) -> None:
    """``include_attributes=False``: line numbers are not part of the dump, so
    inserting anything above real code is free."""
    base = _dump(tmp_path, SOURCE)
    edited = "# a new banner\n\n\n" + SOURCE
    assert _dump(tmp_path, edited, "n.py") == base


def test_reflowed_docstrings_do_not_change_the_dump(tmp_path: Path) -> None:
    base = _dump(tmp_path, SOURCE)
    edited = SOURCE.replace(
        "With a second paragraph that wraps\nacross two lines.",
        "With a second paragraph that wraps across two lines.",
    )
    assert edited != SOURCE
    assert _dump(tmp_path, edited, "n.py") == base


def test_reworded_docstrings_DO_change_the_dump(tmp_path: Path) -> None:
    """Whitespace is free; words are not. Prose that documents a number is part
    of what the number means."""
    base = _dump(tmp_path, SOURCE)
    edited = SOURCE.replace("""The attribute docstring""", """The REWRITTEN docstring""")
    assert _dump(tmp_path, edited, "n.py") != base


@pytest.mark.parametrize(
    "old,new",
    [
        ("FLOOR_UM = 2.0", "FLOOR_UM = 2.5"),          # a constant
        ("return w * h", "return w * h * 2.0"),        # an expression
        ("def area(w: float", "def area(width: float"),  # a signature
    ],
)
def test_code_edits_DO_change_the_dump(tmp_path: Path, old: str, new: str) -> None:
    base = _dump(tmp_path, SOURCE)
    edited = SOURCE.replace(old, new)
    assert edited != SOURCE, f"{old!r} did not appear in the fixture"
    assert _dump(tmp_path, edited, "n.py") != base


def test_fingerprints_are_stable_and_distinct() -> None:
    """Recomputing must give the same answer (nothing in the digest depends on
    iteration order or a timestamp), and the three caches must not collide."""
    gen = cf.fingerprint(cf.PATTERN_GEN_CLOSURE, cf.PHOTO_ASSETS)
    compose = cf.fingerprint(cf.PLATE_COMPOSE_CLOSURE, cf.PHOTO_ASSETS)
    svg = cf.fingerprint(cf.PLATE_SVG_CLOSURE, cf.PHOTO_ASSETS)
    assert gen == cf.fingerprint(cf.PATTERN_GEN_CLOSURE, cf.PHOTO_ASSETS)
    assert len({gen, compose, svg}) == 3
    for fp in (gen, compose, svg):
        assert fp.startswith(f"cf{cf.CACHE_EPOCH}-")
        assert len(fp) == len(f"cf{cf.CACHE_EPOCH}-") + 12


def test_the_markers_the_caches_stamp_are_the_fingerprints() -> None:
    from app.plates.compose import PLATE_COMPOSE_FINGERPRINT
    from app.plates.svg import PLATE_SVG_FINGERPRINT
    from app.service import PATTERN_GEN_FINGERPRINT

    assert PATTERN_GEN_FINGERPRINT == cf.fingerprint(cf.PATTERN_GEN_CLOSURE, cf.PHOTO_ASSETS)
    assert PLATE_COMPOSE_FINGERPRINT == cf.fingerprint(cf.PLATE_COMPOSE_CLOSURE, cf.PHOTO_ASSETS)
    # The SVG marker is written into the file, so it carries a readable prefix.
    assert PLATE_SVG_FINGERPRINT == f"plate-svg-{cf.fingerprint(cf.PLATE_SVG_CLOSURE, cf.PHOTO_ASSETS)}"


def test_the_closures_cover_what_they_claim() -> None:
    """A closure entry that stops resolving is the failure mode the whole
    mechanism exists to prevent, so the modules that actually write the bytes
    are named here too."""
    gen = cf.explain(cf.PATTERN_GEN_CLOSURE, cf.PHOTO_ASSETS)
    assert "patterns/base.py" in gen
    assert "rasterize.py" in gen
    assert "service.py" in gen
    assert any(p.startswith("patterns/") and p.endswith(".py") for p in gen)
    assert any(p.startswith("assets/photos/") for p in gen)

    compose = cf.explain(cf.PLATE_COMPOSE_CLOSURE, cf.PHOTO_ASSETS)
    for name in ("plates/compose.py", "plates/recipe.py", "plates/photo.py",
                 "plates/spec.py", "plates/literal.py", "production.py"):
        assert name in compose, name

    svg = cf.explain(cf.PLATE_SVG_CLOSURE, cf.PHOTO_ASSETS)
    assert "plates/svg.py" in svg
    # The fab SVG must compose what the preview PNG does (CLAUDE.md), so the
    # shared geometry is inside BOTH closures.
    assert {"plates/compose.py", "plates/recipe.py"} <= set(svg) & set(compose)


def test_a_photo_recrop_moves_the_plate_fingerprints(tmp_path: Path, monkeypatch) -> None:
    """The assets are inputs like the code is. Faked by pointing APP_ROOT at a
    copy, because re-cropping a real photograph to prove it would be rude."""
    app_root = tmp_path / "app"
    (app_root / "assets" / "photos").mkdir(parents=True)
    (app_root / "m.py").write_text(SOURCE, encoding="utf-8")
    photo = app_root / "assets" / "photos" / "beach.png"
    photo.write_bytes(b"\x89PNG original")
    monkeypatch.setattr(cf, "APP_ROOT", app_root)

    before = cf.fingerprint(("m.py",), ("assets/photos",))
    photo.write_bytes(b"\x89PNG recropped")
    after = cf.fingerprint(("m.py",), ("assets/photos",))
    assert before != after


def test_a_typo_in_a_closure_is_loud(tmp_path: Path, monkeypatch) -> None:
    """A missing entry must raise, not contribute nothing: silently covering
    one module less is exactly the drift the fingerprints replace."""
    monkeypatch.setattr(cf, "APP_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="not a file"):
        cf.fingerprint(("plates/compse.py",))
    with pytest.raises(RuntimeError, match="matched nothing"):
        cf.fingerprint(("pattrens/**/*.py",))
