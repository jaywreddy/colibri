/**
 * Effect-scenario catalog — the physical-honesty contract for every
 * view-dependent effect the 3D renderer claims to produce.
 *
 * Each scenario states WHY the effect is physically honest and what a
 * cheat would look like. The Playwright driver (effectsPhysical.spec.ts)
 * enforces the native pixel metrics; the claims below also ride along in
 * each scenario's meta.json so the vision verifier (tools/visual_verifier.py)
 * can grade the captured frame sequences semantically.
 *
 * The four honesty axioms, verified across scenarios:
 *   1. View-dependent  — pixels change when the camera direction changes.
 *   2. Time-invariant  — a static camera yields a static frame. No uTime.
 *   3. Texture-driven  — imagery comes from the backend litho rasters
 *                        (the fabricated-chrome PNGs and the period map),
 *                        bound per-face from that face's own manifest.
 *   4. Geometric       — the back gold layer sits at the paraxial T/n air gap
 *                        below the front one; every cross-layer illusion
 *                        emerges from perspective across that gap. The scene
 *                        holds the two layers as real planes at that
 *                        separation, and a literal face — which composites both
 *                        in one pass on the outer plane — reads the gap back off
 *                        that live mesh separation (BoxScene's uInnerGapUm
 *                        sync), so moving the plane still collapses the effect.
 *                        Collapsing the gap registers the layers (parallax
 *                        vanishes); a partial collapse moves the fringes
 *                        proportionally less. A screen-space or scrolling fake
 *                        cannot satisfy this.
 */

/**
 * WHAT RUNS ON WHAT.
 *
 * Every production wall is SINGLE-PLY and LITERAL: one written ply over a bare
 * inner ply, published as a raster of the fabricated chrome that the shader
 * simply samples (plate.frag::runLiteralLayer). Such a face has no second layer
 * to beat against, so its only view-dependent optics beyond plain metal shading
 * is the DIFFRACTION SHEEN the period map carries — the sub-5 µm colour
 * gratings and the 6 µm garland leaf gratings, which are orders of magnitude
 * below what a 2048 px raster can hold and so arrive as a per-pixel pitch map
 * instead. The literal-sheen / time / illumination / lid / turntable scenarios
 * therefore run against the DEFAULT box exactly as it ships (front =
 * globe-atlantic, see backend/app/boxes.py) — nothing is injected.
 *
 * The other two branches of plate.frag are for constructions that NEED a second
 * written ply, so each has a hidden EXEMPLAR this suite assigns to the front
 * face and nothing else in the product reaches:
 *
 *   interlace — BOTH images interlaced in the BACK layer under a neutral slit
 *               barrier in FRONT (CLAUDE.md's image-switch rule). The
 *               globe-duo-phase pattern stays registered for exactly this.
 *   moire     — the shading moiré of a two-ply garland: the same production
 *               monogram slug, composed with `single_ply: false` so the plate
 *               gets a back carrier to beat against. Assigning the ply is the
 *               whole exemplar; there is no separate pattern.
 *
 * Two entries retired with the patterns they previewed: one drove the deleted
 * single-plane stereo_lenticular recipe, the other the banned
 * phase_shift_overlay (a front-layer image that vanishes under tilt, which
 * CLAUDE.md names as an anti-pattern). The physics both pinned — a cross-layer
 * effect that tracks the REAL plane gap — is what the two-ply moiré parallax
 * scenario measures.
 */
export const TEST_PATTERNS = {
  interlace: 'globe-duo-phase',
  moire: 'monogram-jp',
} as const;

export type EffectScenario = {
  name: string;
  /** Physics claim, phrased for the vision grader. */
  claim: string;
  /** What a correct frame sequence looks like. */
  signature: string;
  /** Known cheats/regressions this scenario is designed to catch. */
  failModes: string[];
};

export const EFFECT_SCENARIOS: Record<string, EffectScenario> = {
  'time-invariance': {
    name: 'time-invariance',
    claim:
      'With a static camera and closed lid, the rendering is completely still: every optical effect is view-driven, none is animated by a time uniform.',
    signature: 'Consecutive frames captured 400 ms apart are pixel-identical.',
    failModes: [
      'Fringes drift or shimmer with the camera parked (time-animated shader cheat)',
      'Pattern texture scrolls on its own',
    ],
  },
  'literal-sheen-flow': {
    name: 'literal-sheen-flow',
    claim:
      'Every wall of the shipping box draws the raster of its OWN fabricated chrome, bound from its own manifest, and its diffraction sheen comes from the per-pixel period map of the gratings actually written there — so orbiting the camera makes the gold shift and flash progressively, with no procedural lattice anywhere in the shader.',
    signature:
      'A sweep of camera azimuths shows the same gold plate — same art, same outline, same foil frame — with its sheen and highlights at progressively different strengths across the design.',
    failModes: [
      'Plate frozen while the camera moves (shading not view-dependent)',
      'Appearance jumps discontinuously between angles',
      'Plate shows a synthesized stripe pattern unrelated to the mask imagery',
      'A wall is still on the 1x1 placeholder, or bound to another face’s raster',
      'A wall that declares a period map renders without it (flat gold, no sheen)',
    ],
  },
  'two-ply-moire-parallax': {
    name: 'two-ply-moire-parallax',
    claim:
      'On the two-ply exemplar the moire fringes are caused by the substrate GEOMETRY: the back gold layer sits a paraxial T/n air gap below the front layer, so at a FIXED oblique view the gap sets the fringe positions. Collapsing the gap toward registration moves the fringes; a partial collapse moves them proportionally less; recapturing at the same gap is pixel-identical.',
    signature:
      'Frames at the same oblique camera position at the design gap / near-registration / 95% gap: the gold plate imagery is the same design but the beat fringes sit at clearly different positions, with the partial collapse moving them less than the full one.',
    failModes: [
      'Fringes unchanged when the inner layer is registered to the outer (gap collapse ignored — parallax faked in screen space)',
      'Response does not scale with the gap (partial collapse moves fringes as much as full)',
      'Same-gap recapture differs (nondeterministic rendering)',
      'The gap is not the manifest’s own T/n (every crossing angle is wrong by that ratio)',
    ],
  },
  'barrier-interlace-swap': {
    name: 'barrier-interlace-swap',
    claim:
      'The two-ply barrier interlace is a real parallax barrier: BOTH images live in the BACK layer as alternating lanes and the FRONT layer is a neutral slit comb over the whole art box. Tilting across the barrier axis walks the back lanes under the fixed comb, so one tilt shows A and the other shows B - a hard swap that emerges from perspective across the paraxial T/n gap, with no view-sign term anywhere.',
    signature:
      'Two captures tilted +14deg and -14deg across the barrier axis show clearly different imagery inside the centerpiece; head-on the plate reads as a spatial mix, because a close perspective camera splits it into left/right viewing zones (real barrier behaviour) rather than blending.',
    failModes: [
      'Both tilts show the same image (lanes not moving under the comb - parallax not applied)',
      'The swap follows the sign of the view vector rather than the geometry (the banned phase-overlay cheat)',
      'The front layer carries one of the two images (a front-layer image cannot vanish under tilt)',
    ],
  },
  'lid-transition': {
    name: 'lid-transition',
    claim:
      'The lid opens by rigid rotation about the brass hinge axis on the back top edge, animated by damped pursuit of the target angle; closing returns the scene to its exact initial state.',
    signature:
      'Frames during opening show the lid at monotonically increasing angles, front edge swinging up and back; the final re-closed frame matches the initial closed frame.',
    failModes: [
      'Lid translating or scaling instead of rotating',
      'Rotation overshooting or reversing mid-animation',
      'Re-closed frame differs from the initial frame (state leak)',
    ],
  },
  'illumination-modes': {
    name: 'illumination-modes',
    claim:
      'Ambient, laser, and backlight are distinct physical lighting models evaluated against the same masks: ambient shades the gold, laser lights only the transmission (two-aperture) term in the laser color, backlight lights transmission in white.',
    signature:
      'Three captures of the same view that differ strongly; the laser capture is dominated by the laser color in the pattern area.',
    failModes: [
      'Modes render identically (uniform not wired)',
      'Laser color not reflected in the transmitted light',
    ],
  },
  'turntable-flow': {
    name: 'turntable-flow',
    claim:
      'Auto-rotate is a real camera orbit: engaging it makes the whole rendering evolve smoothly, and disengaging it freezes the frame again.',
    signature: 'Frames differ while the turntable is on and are identical once it is off.',
    failModes: ['Canvas static with auto-rotate enabled', 'Motion continues after disabling'],
  },
};
