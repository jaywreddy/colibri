"""One-shot: write verdict.json + visualReport.md from an in-session Claude
vision grading pass. Run after `just test-visual` when an API key isn't
available for the standard `tools/visual_verifier.py` flow.

Usage:
    RUN_DIR=<abspath> uv run --directory backend --extra dev \
        python ../tools/dev/write_verdicts.py
"""
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])

# (slug, scene, required, passed, confidence, reasoning, concerns)
VERDICTS = [
    (
        "cafetero-iridescence", "ambient-front", True, True, 0.90,
        "Strong polychromatic rainbow fan visible across the gold terraces with red, orange, yellow, green, cyan, and blue streaks distributed horizontally. Clear grating dispersion, not flat gold.",
        [],
    ),
    (
        "cafetero-iridescence", "ambient-tilted", False, True, 0.85,
        "Rainbow bands clearly offset vs ambient-front: colors concentrated on the left edge while the right side fades to gold or dark. View-angle dispersion is working.",
        [],
    ),
    (
        "cafetero-iridescence", "laser-green", False, True, 0.75,
        "Under green laser the plate is predominantly gold terraces with localized green patches (not a continuous rainbow), consistent with discrete diffraction orders under monochromatic illumination.",
        ["Green patches are diffuse rather than sharply discrete orders; shader may be mixing broadband bleed, worth checking the laser-color uniform path."],
    ),
    (
        "caravel-latent", "tilt-neutral", True, True, 0.60,
        "Thin vertical slit grid visible; no caravel ship is showing, which matches the hidden-at-neutral-tilt claim.",
        ["Chevron / ocean-wave carrier is not clearly visible between the slits; the texture behind the slit barrier reads as near-uniform rather than a distinct wave pattern."],
    ),
    (
        "caravel-latent", "tilt-right", True, False, 0.75,
        "No caravel ship silhouette is discernible at this tilt. Plate shows only the slit pattern with perspective distortion, not the latent hull shape the scene promises.",
        ["Matches the 'no ship appears at any tilt' failure mode; latent image may be too weak for this viewing angle or the slit-period / parallax mapping is not revealing the encoded scene."],
    ),
    (
        "colibri-hologram", "default", True, True, 0.75,
        "Plate shows a binary black/gold CGH speckle mask with no recognizable imagery; secondary reconstruction contains faint hummingbird silhouettes offset from center, consistent with an off-axis 4-cell carrier (primary + twin replicas).",
        ["Multiple dim replicas are visible rather than a single dominant silhouette; worth verifying twin-image suppression and whether higher-order replicas bleed through the carrier cell."],
    ),
    (
        "compass-rose-spiral", "ambient-front", True, False, 0.80,
        "Plate shows only concentric rings at the periphery and a large dark central bullseye; no pinwheel or radial-spiral moire beat is visible anywhere. Matches the 'static concentric rings, no spiral pattern' failure mode.",
        ["Either the two ring gratings have identical pitch (no beat generated) or the outer binary mask is suppressing the moire region; the claim requires a pinwheel moire and none is present."],
    ),
    (
        "compass-rose-spiral", "ambient-tilted", False, False, 0.75,
        "Same concentric-ring content as ambient-front, just perspective-rotated. There is no pinwheel pattern to rotate with parallax because none exists head-on.",
        ["Downstream consequence of the ambient-front regression; fixing the base pattern should restore this scene."],
    ),
    (
        "emerald-facet-moire", "ambient-front", True, True, 0.70,
        "Fine gold diamond/hex lattice visible with clearly coarser-scale dark fringes running vertically; a moire beat is present in the overlay of the two lattices.",
        ["Beat pattern reads as vertical striping rather than the promised magnified-hex 'ghost gem'; rotational or pitch mismatch between the two lattices may need tuning to produce hex-shaped beats."],
    ),
    (
        "meridian-speckle", "default", True, True, 0.65,
        "Plate is the meridian+parallel grid over binary speckle as expected. Secondary shows a centered flat-top speckled distribution (not dominated by DC), consistent with a diffuser-style Fraunhofer transform of the aperture.",
        ["Faint cross / grid ghost visible in the secondary on top of the speckle; artifact of the regular meridian structure leaking through the otherwise random aperture. Not incorrect, but noted."],
    ),
    (
        "muzo-emerald-zone", "z-near-focal", True, True, 0.90,
        "Plate shows a hex-clipped Fresnel zone plate with a clearly concentrated bright focal region in the lower-central area; secondary tile stack shows the focal tile visibly brighter and sharper than neighbors. Focal behavior is nominal.",
        [],
    ),
    (
        "muzo-emerald-zone", "z-pre-focal", False, True, 0.85,
        "At pre-focal z the plate shows diffuse ring-diffraction with no sharp central peak and is visibly dimmer than z-near-focal, consistent with 'before focus' expectation.",
        [],
    ),
    (
        "sombrero-vueltiao-parallax", "tilt-left", True, True, 0.80,
        "Concentric hat-brim oval bands are clearly visible through the vertical slits at this tilt; scene A content is showing.",
        [],
    ),
    (
        "sombrero-vueltiao-parallax", "tilt-right", True, False, 0.70,
        "At the opposite tilt the plate shows the same concentric oval brim content as tilt-left (just perspective-rotated); the expected scene-B (pinta triangles / woven motif) does not emerge.",
        ["Matches the 'same content as tilt-left' failure mode; the two scenes are not interlacing correctly through the slit barrier, or the tilt angle is not large enough to trigger the parallax swap."],
    ),
    (
        "tairona-talbot", "z-zero", True, True, 0.80,
        "Plate shows a dense vertical stripe grating at its native period, matching the Ronchi baseline. Secondary Talbot-carpet strip shows the (x,z) revival pattern clearly.",
        ["The decorative 'Tairona ring frame' mentioned in the signature is absent; plate is pure stripes. Artistic framing is not wired in but the physics is correct."],
    ),
    (
        "tairona-talbot", "z-half", True, False, 0.70,
        "Plate at zSlice=0.5 looks essentially identical to zSlice=0; no visible half-period shift, inversion, or period doubling as the Talbot physics requires. The z-slider appears to have no visible effect on the plate rendering.",
        ["Possible regression: the shader may read the carpet atlas at zSlice but the z=0 and z=0.5 rows happen to look similar, or the fractional-Talbot revival encoding is off. Re-run with a scene at z~0.25 to confirm."],
    ),
    (
        "wayuu-kanasu-moire", "ambient-front", True, True, 0.85,
        "Dense diamond weave visible with coarse large-scale moire fringe bands across the plate; the pitch-mismatch beat is clearly operating.",
        [],
    ),
    (
        "wayuu-kanasu-moire", "ambient-tilted", False, True, 0.80,
        "Fringe pattern is in a visibly different spatial arrangement from head-on, with beat bands rearranged by the perspective change; parallax response is working.",
        [],
    ),
]


def main() -> None:
    meta_by_key = {}
    for meta_path in sorted(root.glob("*/*.meta.json")):
        m = json.loads(meta_path.read_text())
        s = m.get("scene", {})
        key = (s.get("slug"), s.get("name"))
        meta_by_key[key] = (meta_path, m)

    for slug, name, required, passed, confidence, reasoning, concerns in VERDICTS:
        key = (slug, name)
        assert key in meta_by_key, f"missing meta for {key}"
        meta_path, m = meta_by_key[key]
        native = m.get("native_checks", {}) or {}
        verdict_path = meta_path.with_name(
            meta_path.name.replace(".meta.json", "") + ".verdict.json"
        )
        payload = {
            "slug": slug,
            "scene": name,
            "required": required,
            "passed": passed,
            "confidence": confidence,
            "reasoning": reasoning,
            "concerns": concerns,
            "skipped_reason": None,
            "native_passed": native.get("passed", True),
            "native_failures": native.get("failures", []),
            "grader": "claude-session-inline-vision",
        }
        verdict_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {verdict_path.relative_to(root)}")

    total = len(VERDICTS)
    native_fail = 0
    verified_pass = sum(1 for v in VERDICTS if v[3])
    verified_fail = sum(1 for v in VERDICTS if not v[3])
    req_fails = [
        (s, n, r)
        for (s, n, req, p, c, r, k) in VERDICTS
        if req and not p
    ]

    lines = []
    lines.append("# Visual verification report")
    lines.append("")
    lines.append(f"- Run directory: `{root}`")
    lines.append(f"- Scenes examined: {total}")
    lines.append(f"- Native pre-check failures: {native_fail}")
    lines.append(f"- LLM verdict PASS: {verified_pass}")
    lines.append(f"- LLM verdict FAIL: {verified_fail}")
    lines.append(f"- Required-scene failures: {len(req_fails)}")
    lines.append(
        "- Grader: Claude vision (in-session; substitutes for the API-driven verifier)"
    )
    lines.append("")
    lines.append("| req | status | slug | scene | reasoning |")
    lines.append("|---|---|---|---|---|")
    for slug, name, required, passed, confidence, reasoning, concerns in VERDICTS:
        req = "*" if required else ""
        status = "PASS" if passed else "FAIL"
        safe = reasoning.replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {req} | {status} | {slug} | {name} | {safe} |")

    lines.append("")
    if req_fails:
        lines.append("## Required-scene failures")
        lines.append("")
        for slug, name, reason in req_fails:
            lines.append(f"- **{slug} / {name}** - {reason}")

    report = root / "visualReport.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote report: {report}")
    print(
        f"summary: {verified_pass} PASS, {verified_fail} FAIL, {len(req_fails)} required-scene fail"
    )


if __name__ == "__main__":
    main()
