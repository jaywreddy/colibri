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
 * Which catalog pattern each recipe's effect tests exercise — re-curated
 * 2026-07-19 after the parallax-honesty audit (app/sim2d.py + the taxonomy
 * investigation):
 *
 * - moire: the moire tests run against the default box front face — whatever
 *   default_box_spec ships there (currently globe-duo-phase, see
 *   backend/app/boxes.py) — rendered through the two-plane foliage_moire
 *   recipe like every composed plate.
 * - stereo: stays on globe-rotation-stereo (parallax barrier: slit front,
 *   both interlaced scenes in BACK — measured 0.895/0.000 channel separation
 *   at ±p/4 shift). jp-monogram-phase is ALSO a stereo_lenticular barrier
 *   after its rebuild and may be the stronger showpiece, but we do not swap
 *   the test pattern without a fresh composite to judge it by.
 * - reveal: the phase_shift_overlay recipe (2) is RETIRED with zero catalog
 *   users and deleted from the shader/API — its two-image front/back phase
 *   split could never switch under honest parallax (the front layer does not
 *   move; the 3D "flip" was an explicit view-sign bias cheat). Its former
 *   users were rebuilt as stereo_lenticular barriers or retagged to
 *   moire_interactive. Its slot in the suite is the honest T5 carrier reveal
 *   (single image halftoned onto a stripe carrier in FRONT, uniform
 *   image-free carrier in BACK).
 */
export const TEST_PATTERNS = {
  stereo: 'globe-rotation-stereo',
  reveal: 'monogram-carrier-reveal',
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
      'The dual-layer gold moire is produced by sampling the real front/back lithography masks through a Snell-refracted parallax shift, so orbiting the camera makes the beat fringes flow continuously across the plate.',
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
  'stereo-lenticular-flip': {
    name: 'stereo-lenticular-flip',
    claim:
      'The stereo lenticular plate is a parallax-barrier: slits over two interlaced scene masks. Tilting the view across the slit axis flips which baked scene (view A vs view B) is visible; head-on, a close camera splits the plate into left/right viewing zones showing each scene.',
    signature:
      'Two captures tilted +14deg and -14deg across the slit axis show clearly different imagery on the plate; the head-on capture shows a spatial mix (both scenes present in different zones across the plate). (In the box preview the outer plane is a neutral slit barrier over BOTH silhouettes interlaced on the inner plane at the paraxial T/n gap; the flip EMERGES from perspective across that gap — no view-sign mix — and re-flips periodically beyond the first zone.)',
    failModes: [
      'Both tilts show the same image (view textures not bound or mix not view-driven)',
      'Head-on capture identical to one extreme (hard switch, no blend zone)',
    ],
  },
  'carrier-reveal-tilt': {
    name: 'carrier-reveal-tilt',
    claim:
      'The carrier reveal is the honest replacement for the retired phase_shift_overlay recipe: the figure is halftoned onto a stripe carrier in the FRONT mask on the outer plane, and a uniform image-free anti-phase carrier at the same period sits on the inner plane at the paraxial T/n gap (the box preview binds foliage_moire, two real planes — no view-sign bias). Tilting to the half-period Snell shift theta(p/2) = asin(n*sin(atan(p/(2t)))) de-registers the carriers by perspective across the gap and the figure contrast appears; zeroing the gap re-registers them and the reveal collapses. The effect depends only on the tilt MAGNITUDE, so opposite tilts must match — the physical signature that separates it from the old recipe-2 cheat, whose flip came from an explicit view-sign term.',
    signature:
      'Head-on the plate reads as a near-uniform fine carrier; at the manifest-derived tilt +/-theta(p/2) the figure stands out clearly, and zeroing the two-plane gap collapses it back to registration (the reveal is parallax-driven, not view-sign biased).',
    failModes: [
      'Tilting to theta(p/2) produces no contrast change (carriers not de-registering — parallax not applied)',
      'Opposite tilts differ strongly (view-sign bias — the retired phase_shift_overlay cheat)',
      'Tilt hardcoded at +/-14 deg instead of derived from the carrier period (zone aliasing: 84 um of shift is ~1-4 periods, landing arbitrarily mid-zone)',
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
