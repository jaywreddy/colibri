"""Vision-based verifier for the optics visual-signature harness (Layer 3).

Reads the PNGs + meta.json sidecars produced by `just test-visual`
(Playwright spec `frontend/tests/e2e/visualSignatures.spec.ts`) and
grades each scene with a Claude vision call. For each scene the model
is prompted with:

    - the physics claim
    - what a correct rendering should show (plate + optional side panel)
    - known failure modes
    - the captured image(s)

It responds with a JSON verdict. We aggregate the verdicts into a
human-scannable `visualReport.md` at the run root and write a
per-scene `<scene>.verdict.json` next to each PNG.

Scenes whose NATIVE checks already failed are skipped: no point
spending tokens on a confirmed regression. We still include them
in the report with the native failure as the reason.

Usage:
    uv run --directory backend --extra dev python ../tools/visual_verifier.py \\
        --run-dir frontend/test-results/visual/latest

Env vars:
    ANTHROPIC_API_KEY — required.
    OPTICS_VERIFY_MODEL — override the model (default: claude-sonnet-4-5).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL = os.environ.get(
    "OPTICS_VERIFY_MODEL", "claude-sonnet-4-5-20250929"
)
MAX_IMAGE_EDGE = 512  # downscale before sending — keeps token cost bounded
MAX_OUTPUT_TOKENS = 400


# --------------------------------------------------------------------------
# Image prep
# --------------------------------------------------------------------------

def _load_and_shrink(path: Path, max_edge: int = MAX_IMAGE_EDGE) -> tuple[bytes, str]:
    """Read an image, downscale if larger than max_edge, return PNG bytes + media type."""
    from PIL import Image

    img = Image.open(path)
    img.load()
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    # Always re-encode as PNG — consistent and lossless at these sizes.
    buf = io.BytesIO()
    # Strip alpha if present; vision models handle RGB better.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class SceneMeta:
    path: Path
    slug: str
    name: str
    required: bool
    claim: str
    plate_signature: str
    secondary_signature: str | None
    fail_modes: list[str]
    native_passed: bool
    native_failures: list[str]
    plate_image: Path
    secondary_image: Path | None
    uniforms: dict[str, Any]
    variant: str | None


@dataclass
class SceneVerdict:
    passed: bool
    confidence: float
    reasoning: str
    concerns: list[str]
    skipped_reason: str | None = None
    raw_response: str | None = None


@dataclass
class RunSummary:
    total: int = 0
    native_failed: int = 0
    verified_pass: int = 0
    verified_fail: int = 0
    required_failures: list[tuple[str, str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Scene discovery
# --------------------------------------------------------------------------

def discover_scenes(run_dir: Path) -> list[SceneMeta]:
    scenes: list[SceneMeta] = []
    for meta_path in sorted(run_dir.rglob("*.meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] {meta_path}: could not parse ({e})", file=sys.stderr)
            continue

        scene = data.get("scene") or {}
        nc = data.get("native_checks") or {}
        # captureScene (frontend/tests/e2e/helpers.ts) writes the canvas PNG
        # under "image"; older sidecars used "plateImage". Accept both.
        plate_rel = data.get("plateImage") or data.get("image")
        sec_rel = data.get("secondaryImage")
        if not plate_rel:
            continue
        plate_path = meta_path.parent / plate_rel
        sec_path: Path | None = (
            meta_path.parent / sec_rel if isinstance(sec_rel, str) else None
        )
        if sec_path and not sec_path.exists():
            sec_path = None
        scenes.append(
            SceneMeta(
                path=meta_path,
                slug=str(scene.get("slug") or data.get("slug") or meta_path.parent.name),
                name=str(scene.get("name") or meta_path.stem.replace(".meta", "")),
                required=bool(scene.get("required", False)),
                claim=str(scene.get("claim") or ""),
                # writeEnrichedMeta (visualSignatures.spec.ts) emits
                # scene.signature; older sidecars used scene.plateSignature.
                plate_signature=str(
                    scene.get("plateSignature") or scene.get("signature") or ""
                ),
                secondary_signature=(
                    str(scene.get("secondarySignature"))
                    if scene.get("secondarySignature")
                    else None
                ),
                fail_modes=list(scene.get("failModes") or []),
                native_passed=bool(nc.get("passed", True)),
                native_failures=list(nc.get("failures") or []),
                plate_image=plate_path,
                secondary_image=sec_path,
                uniforms=dict(data.get("uniforms") or {}),
                variant=data.get("variant"),
            )
        )
    return scenes


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a physics-aware reviewer grading renderings of optical "
    "diffraction patterns. You will be shown one or two captured images "
    "and a rubric describing what a correct rendering should look like "
    "and what known regressions look like. Grade strictly against the "
    "rubric; do not speculate about aesthetics. Respond with JSON only — "
    'no prose, no code fences. Schema: {"pass": <bool>, "confidence": '
    '<0..1>, "reasoning": "<1-2 sentences>", "concerns": [<strings>]}.'
)


def build_user_prompt(s: SceneMeta, has_secondary: bool) -> str:
    lines = [
        f"Pattern: {s.slug}. Scene: {s.name}.",
        f"Physics claim: {s.claim}",
        f"A correct rendering shows (plate canvas): {s.plate_signature}",
    ]
    if has_secondary and s.secondary_signature:
        lines.append(
            f"A correct rendering shows (side panel image): {s.secondary_signature}"
        )
    if s.fail_modes:
        lines.append("Known regressions look like: " + "; ".join(s.fail_modes))
    if has_secondary:
        lines.append(
            "The first image is the plate (main 3D canvas); "
            "the second is the side panel (SecondaryView <img>)."
        )
    else:
        lines.append("The image is the plate (main 3D canvas).")
    lines.append('Respond with JSON only: {"pass": bool, "confidence": 0..1, "reasoning": "<1-2 sentences>", "concerns": [<strings>]}.')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# API call
# --------------------------------------------------------------------------

def grade_scene(client: Any, s: SceneMeta, model: str) -> SceneVerdict:
    if not s.native_passed:
        return SceneVerdict(
            passed=False,
            confidence=1.0,
            reasoning="Skipped LLM grading: native pre-checks already failed.",
            concerns=list(s.native_failures),
            skipped_reason="native_checks_failed",
        )

    images: list[tuple[bytes, str]] = [_load_and_shrink(s.plate_image)]
    has_secondary = False
    if s.secondary_image and s.secondary_image.exists():
        images.append(_load_and_shrink(s.secondary_image))
        has_secondary = True

    content: list[dict[str, Any]] = []
    for data, media_type in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": build_user_prompt(s, has_secondary)})

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        return SceneVerdict(
            passed=False,
            confidence=0.0,
            reasoning=f"API call failed: {e}",
            concerns=["api_error"],
            skipped_reason="api_error",
        )

    text = "".join(
        b.text for b in getattr(resp, "content", []) if getattr(b, "type", None) == "text"
    )
    verdict = _parse_verdict_json(text)
    verdict.raw_response = text
    return verdict


def _parse_verdict_json(text: str) -> SceneVerdict:
    """Extract the JSON object from a model response, forgiving a markdown fence."""
    stripped = text.strip()
    # Tolerate a fenced code block even though the prompt forbids it.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if m:
        stripped = m.group(1)
    else:
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            stripped = m.group(0)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return SceneVerdict(
            passed=False,
            confidence=0.0,
            reasoning=f"Could not parse JSON from response: {text[:200]}",
            concerns=["parse_error"],
        )
    return SceneVerdict(
        passed=bool(data.get("pass", False)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        reasoning=str(data.get("reasoning", ""))[:800],
        concerns=[str(c) for c in (data.get("concerns") or [])][:10],
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_verdict_sidecar(s: SceneMeta, v: SceneVerdict) -> Path:
    out = s.path.with_suffix("")  # strip .json
    out = out.with_name(out.name.replace(".meta", "") + ".verdict.json")
    out.write_text(
        json.dumps(
            {
                "slug": s.slug,
                "scene": s.name,
                "required": s.required,
                "passed": v.passed,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
                "concerns": v.concerns,
                "skipped_reason": v.skipped_reason,
                "native_passed": s.native_passed,
                "native_failures": s.native_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def write_report(
    run_dir: Path, scenes: list[SceneMeta], verdicts: list[SceneVerdict]
) -> Path:
    lines: list[str] = []
    lines.append("# Visual verification report")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Scenes examined: {len(scenes)}")
    summary = RunSummary(total=len(scenes))
    for s, v in zip(scenes, verdicts):
        if not s.native_passed:
            summary.native_failed += 1
        elif v.passed:
            summary.verified_pass += 1
        else:
            summary.verified_fail += 1
        if s.required and not v.passed:
            summary.required_failures.append(
                (s.slug, s.name, v.reasoning or "no reasoning")
            )
    lines.append(f"- Native pre-check failures: {summary.native_failed}")
    lines.append(f"- LLM verdict PASS: {summary.verified_pass}")
    lines.append(f"- LLM verdict FAIL: {summary.verified_fail}")
    lines.append(f"- Required-scene failures: {len(summary.required_failures)}")
    lines.append("")
    lines.append("| req | status | slug | scene | reasoning |")
    lines.append("|---|---|---|---|---|")
    for s, v in zip(scenes, verdicts):
        if not s.native_passed:
            status = "NATIVE-FAIL"
            reason = "; ".join(s.native_failures)[:160]
        elif v.skipped_reason:
            status = f"SKIP ({v.skipped_reason})"
            reason = v.reasoning[:160]
        else:
            status = "PASS" if v.passed else "FAIL"
            reason = v.reasoning[:160]
        req = "*" if s.required else ""
        safe_reason = reason.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {req} | {status} | {s.slug} | {s.name} | {safe_reason} |")
    report_path = run_dir / "visualReport.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Vision verifier for optics visual harness")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Output directory from `just test-visual` (contains per-slug subdirs with *.meta.json)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the API call; only validate native checks and produce a report",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    scenes = discover_scenes(run_dir)
    if not scenes:
        print(f"No *.meta.json files under {run_dir}", file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "ANTHROPIC_API_KEY not set. Re-run with --dry-run to skip LLM calls.",
                file=sys.stderr,
            )
            return 2
        try:
            import anthropic
        except ImportError:
            print(
                "The `anthropic` package is not installed. Run: `uv sync --extra dev` "
                "inside backend/ first.",
                file=sys.stderr,
            )
            return 2
        client = anthropic.Anthropic()

    verdicts: list[SceneVerdict] = []
    for s in scenes:
        print(
            f"[grade] {s.slug}/{s.name} (required={s.required}, native_pass={s.native_passed})",
            file=sys.stderr,
        )
        if args.dry_run or client is None:
            if s.native_passed:
                verdicts.append(
                    SceneVerdict(
                        passed=True,
                        confidence=0.0,
                        reasoning="Dry-run: native checks passed; LLM grading skipped.",
                        concerns=[],
                        skipped_reason="dry_run",
                    )
                )
            else:
                verdicts.append(
                    SceneVerdict(
                        passed=False,
                        confidence=1.0,
                        reasoning="Dry-run: native checks failed.",
                        concerns=list(s.native_failures),
                        skipped_reason="native_checks_failed",
                    )
                )
        else:
            verdicts.append(grade_scene(client, s, args.model))
        write_verdict_sidecar(s, verdicts[-1])

    report = write_report(run_dir, scenes, verdicts)
    print(f"Wrote report: {report}", file=sys.stderr)

    # Non-zero exit on any required-scene failure.
    required_failed = sum(
        1 for s, v in zip(scenes, verdicts) if s.required and not v.passed
    )
    if required_failed > 0:
        print(
            f"{required_failed} required scene(s) failed visual verification.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
