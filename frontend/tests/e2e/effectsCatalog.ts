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
 *   3. Texture-driven  — imagery comes from the backend litho masks
 *                        (front/back PNGs), bound per-face from the manifest.
 *   4. Geometric       — the back gold layer renders on a REAL inner plane at
 *                        the paraxial T/n air gap below the outer plane; every
 *                        cross-layer illusion emerges from perspective across
 *                        that gap. Collapsing the gap to zero registers the
 *                        layers (parallax vanishes); a partial collapse moves
 *                        the fringes proportionally less. A screen-space or
 *                        scrolling fake cannot satisfy this.
 */

/**
 * The two-ply EXEMPLARS this suite assigns to the front face, and why they
 * exist at all.
 *
 * Every production wall is SINGLE-PLY and LITERAL: the backend publishes a
 * raster of the fabricated chrome and the shader samples it, so the moiré and
 * the barrier switch emerge from perspective across the real T/n plane gap with
 * no analytic grating anywhere. The moiré / parallax / time / illumination /
 * lid / turntable scenarios below therefore run against the DEFAULT box exactly
 * as it ships (front = globe-atlantic, see backend/app/boxes.py) — nothing is
 * injected.
 *
 * `interlace` is the one construction a literal single-ply face cannot show,
 * because it needs a second written ply: BOTH images interlaced in the BACK
 * layer under a neutral slit barrier in FRONT (CLAUDE.md's image-switch rule).
 * globe-duo-phase stays registered as a hidden dev exemplar so that branch of
 * plate.frag keeps a subject; it is not a face of the box.
 *
 * The retired entries: `stereo` (globe-rotation-stereo) previewed the deleted
 * single-plane stereo_lenticular recipe, and `reveal` (monogram-carrier-reveal)
 * stood in for the banned phase_shift_overlay. Both patterns went with the
 * catalogue; the physics each was pinning is covered by the moiré-parallax
 * scenario, which scales the real plane gap.
 */
export const TEST_PATTERNS = {
  interlace: 'globe-duo-phase',
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
  'moire-fringe-flow': {
    name: 'moire-fringe-flow',
    claim:
      'The gold moire is produced by sampling the real fabricated-chrome rasters of the two layers across the paraxial T/n plane gap, so orbiting the camera makes the beat fringes flow continuously across the plate.',
    signature:
      'A sweep of camera azimuths shows the same gold plate with fringe bands at progressively shifted positions; the plate outline and foil frame stay put.',
    failModes: [
      'Fringes frozen while the camera moves (parallax not applied)',
      'Pattern jumps discontinuously between angles',
      'Plate shows a procedural stripe pattern unrelated to the mask imagery',
    ],
  },
  'moire-parallax-physics': {
    name: 'moire-parallax-physics',
    claim:
      'The fringes are caused by the substrate GEOMETRY: the back gold layer renders on a real inner plane at the paraxial T/n air gap below the outer plane, so at a FIXED oblique view the gap sets the fringe positions. Collapsing the gap to zero registers the layers and moves the fringes; a partial (60%) collapse moves them proportionally less; recapturing at the same gap is pixel-identical.',
    signature:
      'Frames at the same oblique camera position at the design gap / gap 0 / 60% gap: the gold plate imagery is the same design but the beat fringes sit at clearly different positions, with the partial collapse moving them less than the full one.',
    failModes: [
      'Fringes unchanged when the inner plane is registered to the outer (gap collapse ignored — parallax faked in screen space)',
      'Response does not scale with the gap (partial collapse moves fringes as much as full)',
      'Same-gap recapture differs (nondeterministic rendering)',
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
