precision highp float;

#include './lib/parallax.glsl'

varying vec2 vUv;
varying vec3 vViewDirTangent;
varying vec3 vLightDirTangent;
varying vec3 vNormalWorld;

uniform sampler2D uFront;
uniform sampler2D uBack;

uniform vec2 uExtentUm;        // physical (width, height) of the plate surface (μm)
uniform float uThicknessUm;    // substrate thickness
uniform float uN;              // refractive index of substrate

uniform int uIllumination;     // 0=ambient, 1=laser, 2=backlight
uniform vec3 uLaserColor;
uniform vec3 uBacklightColor;
uniform vec3 uAmbientColor;

// Which render recipe to run. Numeric IDs match RECIPE_IDS in frontend/src/api.ts:
//   0 stereo_lenticular, 1 moire_interactive, 3 foliage_moire (every composed
//   box plate). Id 2 (phase_shift_overlay) is RETIRED — the hole is
//   intentional so 0/1/3 never renumber.
uniform int uRecipe;

// --- stereo_lenticular uniforms (only read when uRecipe == 0) ---------------
uniform sampler2D uViewA;          // tilt-positive scene
uniform sampler2D uViewB;          // tilt-negative scene
uniform float uSlitOrientation;    // radians; slit-normal direction (0 = +X)
uniform float uSlitPeriodUm;       // slit period Λ_slit (μm)

// --- shared foliage/centerpiece uniforms (read when uRecipe == 3) -----------
// Historically declared for the retired phase_shift_overlay recipe, but they
// are LIVE on the foliage_moire path: uCarrierPeriodUm drives the frame back
// carrier and uSwitchAxis the grating axis of the capybara body shimmer + the
// legacy 2-phase centerpiece fallback. Do NOT delete with recipe 2.
uniform float uSwitchAxis;         // radians; axis we project view onto
uniform float uCarrierPeriodUm;    // carrier stripe period (μm)

// --- foliage_moire uniforms (only read when uRecipe == 3) -------------------
// Two fine gratings drawn ANALYTICALLY (no baked raster → no aliasing rings):
//   back  grating: uCarrierPeriodUm @ uCarrierAngle, fills the back window
//                  mask (uBack) which spans the whole exposed face.
//   front grating: uSlitPeriodUm    @ uSlitAngle,    fills the foliage
//                  silhouette mask (uFront).
// A small angle offset + period ratio between them produces the moiré beat;
// the substrate parallax slides the back grating so fringes travel with tilt.
uniform float uCarrierAngle;       // radians; back grating orientation
uniform float uSlitAngle;          // radians; front grating orientation (bucket center)
uniform float uGratingDuty;        // gold-line fraction of a period (0..1)
// Per-motif frame-band angle encoding. The FRONT frame mask carries a small
// palette of graylevels (see plates.py FRAME_BUCKET0 / FRAME_BUCKET_STEP): a
// motif painted at bucket b sits at level (bucket0 + b·step)/255. We recover b
// and rotate that motif's louvre by (b - (count-1)/2)·span, so each species /
// leaf shimmers in its own fringe direction and beats differently against the
// uniform back carrier. All bucket levels stay inside (FRAME_MIN, ART_MIN) so
// the frame/art classification is unaffected.
uniform float uFrameBucket0;       // L of bucket 0, normalized 0..1
uniform float uFrameBucketStep;    // L step per bucket, normalized 0..1
uniform float uFrameBucketCount;   // number of buckets
uniform float uFrameAngleSpan;     // radians between adjacent buckets
// Centerpiece tilt-switch carrier (foliage_moire, centerpiece region). The
// colibrí (front) and globe (back) silhouettes are filled with this vertical
// stripe carrier; the back rides half a period out of phase and the parallax
// shift biases which phase the eye samples -> one tilt reveals the bird, the
// other the globe. uSwitchAxis (declared above) is the projection axis.
uniform float uCenterPeriodUm;     // centerpiece stripe period (um)

// --- TWO-PLANE geometric renderer (foliage_moire) ---------------------------
// The box preview draws the gold-on-quartz pattern as TWO real PlaneGeometry
// surfaces separated by the physical slab (outer front at +T/2, inner back at
// -T/2). Each plane runs THIS shader with uLayer selecting which single layer it
// draws. There is NO cross-layer texture sampling and NO in-shader parallax for
// the gratings: every cross-layer illusion (leaf moire, colibri<->globe switch,
// gear/quill switch) EMERGES from the perspective projection of the two real
// surfaces, as validated in the two-plane rig. Each plane binds its own mask to
// uFront; output alpha = the layer's gold coverage so the glass gaps are
// transparent and the inner plane shows through the outer one.
uniform float uLayer;              // 0 = outer/front plane, 1 = inner/back plane

// --- Pattern Scale (Task 1b) ------------------------------------------------
// Multiplier applied to the MAGNIFIED-PREVIEW period family on BOTH planes: the
// frame back carrier, the frame louvre, and the capybara body shimmer (the
// backend already ships those three pre-magnified by PREVIEW_PITCH_MAGNIFY).
// Larger scales (2/4/8×) blow that fine structure up so it resolves on screen
// without physically zooming the camera. Frontend only — no baked geometry
// changes, so the fab masks are untouched.
//
// The CENTERPIECE family (barrier/switch comb, interlace lanes, water comb) is
// deliberately EXCLUDED: the geometric parallax gap is T/n regardless of scale,
// so scaling the barrier pitch p scales the switch tilt angle atan(p/4 / (T/n))
// with it — a 4× scale reports a 9.9° swap for a part that swaps at 2.5°. The
// preview must answer "does a comfortable hand tilt perform the swap?" with the
// FAB number, so the centerpiece always draws at exact fab pitch and legibility
// there is a camera-zoom / pixel-ratio question, never a pitch question.
uniform float uPatternScale;

// --- Barrier-interlace tilt switch (Task 3) ---------------------------------
// > 0.5 on the hard-swap faces (globe-duo California↔Colombia, gear↔quill,
// colibrí wing flap). The centerpiece is re-architected from the phase-offset
// construction to a TRUE lenticular/Poemotion barrier interlace: BOTH images live
// on the BACK layer, interleaved in alternating lanes (A in even lanes, B in odd,
// lane pitch = half the barrier pitch uCenterPeriodUm), and the OUTER plane is a
// NEUTRAL slit barrier (open duty 0.5 → one lane wide) spanning the FULL
// centerpiece ART BOX — never the A∪B silhouette union. The swap EMERGES from the
// two real planes' parallax: tilt one way and the slot sits over the A-lanes (only
// A shows), tilt the other and it sits over the B-lanes (only B shows) — a hard
// swap, not a phase redistribution. A and B are the front/back silhouette PNGs;
// each plane reads BOTH (they are the two halves of the ONE back-layer interleaved
// image + its barrier), which is the sanctioned construction for this recipe (not
// the emergent-moiré cross-sampling the two-plane rig forbids).
uniform float uSwitchInterlace;
// Solved barrier REGISTRATION phase (µm), from recipe_data
// ``switch_barrier_phase_um`` — the ONE convention all three consumers share
// (fab SVG bake, this shader, the standalone generators): x measured from the
// FACE CENTRE, open-slit centres at k·p + phase, back channel A starting on that
// same boundary's +x side. Read it; never assume it. The lattice used to be
// anchored on the face EDGE here, which drifts from the fab lattice by
// (extent/2) mod p — half a period on the default 50 mm face, i.e. the preview
// showed B where the part shows A and the swap ran backwards.
uniform float uSwitchBarrierPhaseUm;

// --- water scanimation (foliage_moire, capybara back face) ------------------
// When uWaterScanN > 0 the centerpiece BACK-art region (uBack ART level) is the
// WATER BAND, and the FRONT-art region (uFront ART level) is the still capybara.
// Instead of the 2-phase colibrí/globe switch, the water region renders N-phase
// travelling wavelets whose crest field slides sideways by ONE phase step per
// parallax increment — the box rock walks the ripple through N phases so the
// water appears to flow around the animal (the same barrier-grid scanimation
// the standalone pattern + fab SVG bake, rendered analytically here). N == 0
// (every non-capybara face) disables this branch entirely.
//
// FLOW RATE — there is no preview knob for it, and there was never a live one (a
// uWaterPhasePitchPreviewUm uniform was declared, bound and documented as the
// divisor that walks the phase, but no line of GLSL ever read it; the two-plane
// rewrite had already made the phase emerge geometrically). The rate is set by the
// geometry: the outer comb reveals the next 1/N lane after uCenterPeriodUm/N µm of
// substrate parallax — 15 µm at the fab 60 µm pitch and N=4, i.e. ~2.5° of tilt per
// ripple phase through the T/n gap, which is also the fab timing
// (WATER_SCAN_FAB_PITCH_UM = 60 µm). Retune it by changing the barrier pitch or N,
// never by a preview-only constant.
uniform float uWaterScanN;               // ripple phase count (0 = disabled)
uniform float uWaterRippleWavelengthUm;  // crest spacing along the flow axis (μm)
// Body shimmer period (μm) over the DRY capybara: recipe_data
// ``water_body_carrier_preview_um`` — the 24 µm period the fab path actually bakes
// there (WATER_SCAN_FAB_CARRIER_UM), pre-magnified for preview like the frame pair
// and therefore scaled by uPatternScale. It is NOT the frame carrier this branch
// used to reuse (uCarrierPeriodUm, 22 µm design): the two disagreed by the 22-vs-24
// gap on the one region the eye lands on. <= 0 → fall back to uCarrierPeriodUm.
uniform float uWaterBodyPeriodUm;
// Effective waterline in ART-BOX v (0 = box top, 1 = box bottom), i.e. where the
// dry body ends and the water band begins. Drives the body/water split AND the
// flow wake's depth shear + calm patch, so it must match the mask the plate baked
// (capybara_scanimation.WATERLINE_Y, exposed there as a 0.4-0.85 param). Bound
// from recipe_data; there is no in-shader constant to drift from any more.
uniform float uWaterWaterlineY;
// Centerpiece ART-BOX uv-rect. Two consumers: the capybara flow WAKE geometry
// (body center, waterline, calm patch — all authored in the 0..1 art box) and the
// barrier-interlace comb, which spans the WHOLE box (see uSwitchInterlace). The
// centerpiece is a SQUARE of side (CENTERPIECE_FILL·aperture) centered on the
// plate, so on a non-square face it maps to different uv half-extents per axis.
// uArtBoxHalfUv = (halfWidthUv, halfHeightUv); uArtBoxCenterUv = its uv center
// (normally (0.5,0.5)). See plates.py recipe_data water_art_half_uv /
// water_art_center_uv and the BoxScene uniform binding. Fallback (0,0) → treat the
// whole face as the art box (approx; only correct on a square face fully filled by
// the centerpiece) — BoxScene logs that degradation rather than taking it quietly.
uniform vec2 uArtBoxHalfUv;              // (halfW, halfH) of the art box in uv
uniform vec2 uArtBoxCenterUv;            // uv center of the art box

// --- diffraction rainbow accent (foliage_moire) -----------------------------
// A reserved graylevel (RAINBOW_LEVEL = 200/255 ≈ 0.784, see plates.py) marks
// "diffraction accent zone": in fab these pixels get a 4.4 µm (sub-5 µm) 45°
// grating that fans white light into a first-order rainbow on tilt. In PREVIEW
// we can't render true diffraction, so we fake the hologram-foil look: an
// angle-dependent spectral sheen whose hue sweeps with the view vector's
// projection, gated to a narrow travelling highlight band. uRainbowLevel < 0
// disables the whole feature so pre-accent manifests render identically.
uniform float uRainbowLevel;       // L of the accent level, normalized 0..1 (<0 = off)

const vec3 GOLD = vec3(0.902, 0.737, 0.314);
const vec3 GOLD_BACK = vec3(0.4, 0.32, 0.12);

vec3 ambientLit(vec3 base) {
  // Simple Lambert + subtle specular on gold
  float ndl = max(0.0, vLightDirTangent.z);
  float ndv = max(0.0, vViewDirTangent.z);
  vec3 h = normalize(vLightDirTangent + vViewDirTangent);
  float ndh = max(0.0, h.z);
  float spec = pow(ndh, 80.0) * 0.35;
  return base * (0.2 + 0.9 * ndl) + vec3(1.0, 0.85, 0.6) * spec * (0.3 + ndv);
}

// ----------------------------------------------------------------------------
// Recipe 1: moire_interactive — sample front & back with physical parallax.
// The Snell-refracted shift means rotating/orbiting the camera actually
// produces moving moiré fringes.
// ----------------------------------------------------------------------------
vec3 runMoireInteractive(vec3 viewTangent) {
  vec2 shift = parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);
  float frontGold = texture2D(uFront, vUv).r;
  float backGold  = texture2D(uBack,  vUv - shift).r;

  // Transmission through *both* apertures — this is where moiré fringes
  // show up as bright/dark beats.
  float transmission = (1.0 - frontGold) * (1.0 - backGold);
  float reflected    = max(frontGold, backGold * 0.55);

  vec3 color;
  if (uIllumination == 0) {
    vec3 goldShade = GOLD * reflected * (0.3 + 0.7 * max(0.0, vLightDirTangent.z));
    color = goldShade + vec3(0.04) * transmission;
    float overlap = frontGold * backGold;
    color *= (1.0 - 0.35 * overlap);
  } else if (uIllumination == 1) {
    color = uLaserColor * transmission * (0.45 + 0.55 * max(0.0, vLightDirTangent.z));
    color += GOLD * 0.12 * reflected;
  } else {
    color = uBacklightColor * transmission;
    color += GOLD_BACK * reflected * 0.25;
  }
  return color;
}

// ----------------------------------------------------------------------------
// Recipe 0: stereo_lenticular — parallax-barrier slit grating + two interlaced
// scenes baked into view_a / view_b textures. The sign of the projected view
// vector on the slit-normal axis picks which scene is visible through the
// slits. Switch half-angle ≈ arctan(p/2·t).
// ----------------------------------------------------------------------------
vec3 runStereoLenticular(vec3 viewTangent) {
  vec2 shift = parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);

  // EMERGENT parallax barrier. A front comb (opaque bars, slit open half of each
  // period) sits over a BACK layer carrying the two scenes interlaced column by
  // column at a half-period offset (view A in one half of each period, view B in
  // the other). The substrate parallax slides the interlace under the fixed
  // front comb, so the slit uncovers view-A columns at one tilt and view-B
  // columns at the other — the stereo switch FALLS OUT of which column the slit
  // reveals, not a view-projection blend. Supersampled across the pixel footprint
  // so the ~Λ_slit comb + interlace average honestly at any zoom.
  vec2 slitNormal = vec2(cos(uSlitOrientation), sin(uSlitOrientation));
  vec2 pF = vUv * uExtentUm;
  vec2 pB = (vUv - shift) * uExtentUm;
  vec2 duvx = dFdx(vUv); vec2 duvy = dFdy(vUv);
  vec2 dFx = dFdx(pF);   vec2 dFy = dFdy(pF);
  vec2 dBx = dFdx(pB);   vec2 dBy = dFdy(pB);
  float period = max(1.0, uSlitPeriodUm);

  float sceneAcc = 0.0;   // scene light through the slits
  float barrierAcc = 0.0; // opaque-bar coverage (visible gold barrier)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      float ox = (float(i) - 1.0) * 0.3333333;
      float oy = (float(j) - 1.0) * 0.3333333;
      vec2 sUv = vUv + duvx * ox + duvy * oy;
      vec2 sUvB = sUv - shift;
      float uFrontC = dot(pF + dFx * ox + dFy * oy, slitNormal) / period;
      float uBackC  = dot(pB + dBx * ox + dBy * oy, slitNormal) / period;
      float slitOpen = step(fract(uFrontC), 0.5);             // front comb slit
      float pickB    = step(0.5, fract(uBackC));              // interlace column
      float scene = mix(texture2D(uViewA, sUvB).r,
                        texture2D(uViewB, sUvB).r, pickB);
      sceneAcc   += scene * slitOpen;
      barrierAcc += (1.0 - slitOpen);
    }
  }
  float sceneVis  = sceneAcc / 9.0;   // interlaced scene seen through the slits
  float frontGold = barrierAcc / 9.0; // barrier bars (gold)
  float transmission = sceneVis;

  vec3 color;
  if (uIllumination == 0) {
    vec3 sceneGold = GOLD * transmission * (0.35 + 0.7 * max(0.0, vLightDirTangent.z));
    vec3 barrier = GOLD * frontGold * (0.2 + 0.5 * max(0.0, vLightDirTangent.z)) * 0.6;
    color = sceneGold + barrier;
  } else if (uIllumination == 1) {
    color = uLaserColor * transmission * (0.5 + 0.5 * max(0.0, vLightDirTangent.z));
    color += GOLD * 0.08 * frontGold;
  } else {
    color = uBacklightColor * transmission * 0.9;
    color += GOLD_BACK * frontGold * 0.18;
  }
  return color;
}

// (Recipe 2, phase_shift_overlay, was RETIRED and its runPhaseShiftOverlay
// function deleted: a front-layer image can never vanish under parallax
// because the front mask does not move with tilt. Its former users are now
// stereo_lenticular barriers or moire_interactive single-layer art.)

// ----------------------------------------------------------------------------
// Recipe 3: foliage_moire — box-first moiré carrier.
//   uFront = foliage silhouette mask (whole face, art shape).
//   uBack  = uniform carrier-window mask (whole exposed face rectangle).
// Both gratings are generated procedurally here so nothing is baked into the
// PNGs (no raster aliasing). The gold-line coverage of each layer is a fine
// grating; the transmission through both beats into travelling moiré fringes.
// ----------------------------------------------------------------------------

// Analytic 1-D grating coverage at physical point p (μm), grating rotated by
// `angle`, period `periodUm`, duty `duty`. Antialiased via fwidth so it never
// shimmers into aliasing regardless of zoom — the coverage smoothly averages
// to `duty` when a pixel spans many lines.
float gratingCoveragePhase(vec2 pUm, float angle, float periodUm, float duty, float phase) {
  float c = cos(angle);
  float s = sin(angle);
  float coord = (pUm.x * c + pUm.y * s) / max(1.0, periodUm) + phase; // in periods
  float f = fract(coord);
  // Distance-to-edge antialiasing: width of one pixel in period units.
  float w = fwidth(coord);
  // Two smoothstep edges make a band [0, duty] = gold. Clamp AA width so we
  // gracefully fade to the mean coverage (duty) when lines subpixel-collapse.
  float aa = clamp(w, 0.0004, 0.5);
  float line = smoothstep(0.0, aa, f) - smoothstep(duty, duty + aa, f);
  // When a pixel spans >~1 period, fade to the average duty (prevents moiré
  // aliasing against the pixel grid — the real fringes come from layer beats).
  float collapse = smoothstep(0.35, 0.9, w);
  return mix(line, duty, collapse);
}

float gratingCoverage(vec2 pUm, float angle, float periodUm, float duty) {
  return gratingCoveragePhase(pUm, angle, periodUm, duty, 0.0);
}

// NOTE: the old single-plane path evaluated BOTH layers in one fragment and
// supersampled their product (gratingHard + overlapSupersampled) to synthesise
// the moiré beat analytically. That whole dual-layer-in-one-shader machinery is
// GONE: the beat now emerges from the perspective projection of the two REAL
// planes (see runFoliageMoireLayer), so each fragment only ever draws its own
// single layer. The helpers were removed with it.

// Region thresholds on the graylevel masks (see plates.py). The frame band
// carries angle-bucket levels up to 166/255 ≈ 0.651; the reserved diffraction
// accent level is 200/255 ≈ 0.784; ART_LEVEL = 255/255 = 1.0. We carve three
// windows so the accent level classifies as neither frame nor art:
//   FRAME   : (FRAME_MIN, RAINBOW_MIN)   frame band / carrier window
//   RAINBOW : [RAINBOW_MIN, ART_MIN)     diffraction accent zone
//   ART     : [ART_MIN, 1.0]             centerpiece silhouette
// Keep these in sync with plates.py FRAME_BUCKET*/RAINBOW_LEVEL/ART_LEVEL.
const float FRAME_MIN = 0.2;    // r above this = frame/window (band tops at ~0.651)
const float RAINBOW_MIN = 0.72; // r in [this, ART_MIN) = diffraction accent (0.784)
const float ART_MIN = 0.86;     // r above this = centerpiece art silhouette (1.0)

// Angle-dependent spectral sheen for a diffraction-accent pixel — the PREVIEW
// stand-in for a real sub-5 µm rainbow fan. The hue sweeps with the view
// vector's tangent-space projection (so it travels as the piece tilts, like a
// hologram-foil sticker); a narrow travelling band keeps it a tasteful moving
// highlight rather than a flat rainbow wash. Returns an ADDITIVE colour.
vec3 diffractionSheen(vec3 viewTangent, vec2 pUm, float ndl) {
  // Drive the spectrum off the view vector's in-plane projection along the
  // fixed 45° accent grating normal (matches the fab grating orientation).
  vec2 gnorm = vec2(0.70710678, 0.70710678); // cos/sin 45°
  float proj = dot(viewTangent.xy, gnorm);
  // A little spatial term so the fan is not perfectly uniform across the zone
  // (real first-order angle varies with position on a curved read-out).
  float spatial = (pUm.x * gnorm.x + pUm.y * gnorm.y) * 0.0006;
  float hue = fract(proj * 1.6 + spatial + 0.5);        // 0..1 rainbow ramp
  // Narrow highlight band that travels with tilt: brightest where the first
  // order would flash, fading fast to either side. fwidth AA on the band edge.
  float band = 0.5 + 0.5 * cos((proj * 3.14159265) * 2.0);
  band = pow(clamp(band, 0.0, 1.0), 6.0);               // narrow the peak
  // HSV(hue,1,1) → RGB, saturated spectral colour.
  vec3 rgb = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  float lit = 0.35 + 0.65 * ndl;
  return rgb * band * lit;
}

// --- FLOWING-CURRENT water ripple (capybara back face) ----------------------
// A field of long, undulating STREAMLINES (ridges along the flow/x axis) whose
// transverse undulation travels downstream by one wavelength across the N
// animation frames. ``phaseStep`` (0..N) is walked by the substrate parallax,
// so a hand rock sweeps the crests laterally in ONE consistent direction — the
// water reads as a continuous DIRECTIONAL current, not stepping bands. A wake
// opens around the half-submerged body: streamlines PART in y around it, a calm
// elliptical patch sits under the belly, and the undulation is shoved downstream
// behind it (a trailing tongue / V-wake). This is the analytic twin of the
// Python builder's _flow_streamline_field / _flow_amplitude (capybara_
// scanimation.py) — the two MUST stay in lock-step (same constants below).
//
// Works in NORMALIZED art-box coords (uvN in 0..1, y-DOWN) for the wake geometry
// (body center, waterline) and converts the ripple wavelength from μm to
// normalized via the face extent, so the crest spacing and wake match the baked
// fab geometry regardless of plate size. `bodyCx/Cy` mirror the Python constants;
// the waterline is the uWaterWaterlineY UNIFORM, because Python threads it through
// as a parameter — a const here silently desynchronizes the preview from the mask.
const float FLOW_DIR       = 1.0;   // +1: crests advance toward +x as phase grows
const float FLOW_BODY_CX   = 0.46;  // body-center x (normalized, matches motif)
const float FLOW_BODY_CY   = 0.60;
const float FLOW_BAND_FRAC = 0.55;  // streamline spacing = wavelength * this
const float FLOW_A1        = 0.42;  // primary transverse undulation amplitude
const float FLOW_A2        = 0.14;  // second-harmonic amplitude
const float FLOW_SHEAR     = 0.35;  // deeper streamlines lag → raked current
const float FLOW_WAKE_INFL = 0.30;  // gaussian radius of body influence
const float FLOW_WAKE_PART = 0.22;  // how far streamlines part (y) around body
const float FLOW_WAKE_SHOVE= 0.22;  // downstream shove (trailing tongue)

// tanh is not a built-in in GLSL ES 1.00 (this ShaderMaterial compiles at that
// version — no glslVersion:GLSL3), so provide it explicitly. Clamp the argument
// to avoid exp() overflow at large |x| (saturates to ±1 anyway).
float tanhApprox(float x) {
  float e = exp(2.0 * clamp(x, -10.0, 10.0));
  return (e - 1.0) / (e + 1.0);
}

// Scalar field whose near-integer iso-lines are the flowing streamlines. wLenN
// is the ripple wavelength in NORMALIZED units. Mirrors _flow_streamline_field.
float flowStreamlineField(vec2 uvN, float wLenN, float nPhases, float phaseStep) {
  float depth = clamp((uvN.y - uWaterWaterlineY) / max(1e-6, (1.0 - uWaterWaterlineY)), 0.0, 1.0);
  float travel = wLenN * FLOW_DIR * (phaseStep / max(1.0, nPhases));
  float shear = 6.2831853 * FLOW_SHEAR * depth;

  float dx = uvN.x - FLOW_BODY_CX;
  float dy = uvN.y - FLOW_BODY_CY;
  float r = sqrt(dx * dx + dy * dy) + 1e-3;
  float infl = exp(-((r / FLOW_WAKE_INFL) * (r / FLOW_WAKE_INFL)));
  float yPart = FLOW_WAKE_PART * tanhApprox(dy / 0.10) * infl;
  float downstream = 0.5 + 0.5 * tanhApprox(dx * FLOW_DIR / 0.10);
  float centerline = exp(-((dy / 0.18) * (dy / 0.18)));
  float xShove = FLOW_WAKE_SHOVE * downstream * centerline * infl * FLOW_DIR;

  float phaseX = 6.2831853 * (uvN.x - travel - xShove) / max(1e-4, wLenN) + shear;
  float undul = (FLOW_A1 * sin(phaseX) + FLOW_A2 * sin(2.0 * phaseX + 0.6)) * (1.0 - 0.3 * depth);

  float bandGap = wLenN * FLOW_BAND_FRAC;
  return (uvN.y + yPart) / max(1e-4, bandGap) - undul;
}

// Crest strength 0..1: strong open water, calm elliptical patch under the belly,
// gently fading with depth. Mirrors _flow_amplitude.
float flowAmplitude(vec2 uvN) {
  float depth = clamp((uvN.y - uWaterWaterlineY) / max(1e-6, (1.0 - uWaterWaterlineY)), 0.0, 1.0);
  float amp = 1.0 - 0.35 * depth;
  float bx = FLOW_BODY_CX;
  float by = FLOW_BODY_CY + 0.16;
  float ex = (uvN.x - bx - 0.05 * FLOW_DIR) / 0.17;
  float ey = (uvN.y - by) / 0.11;
  float rb = sqrt(ex * ex + ey * ey);
  float calm = clamp((rb - 0.55) / 0.9, 0.0, 1.0);
  return clamp(amp * calm, 0.0, 1.0);
}

// Gold coverage of the streamline crests for one animation frame. uvN normalized
// (0..1, y-down), wavelengthUm μm crest spacing, extentUm the face size (μm).
float waterRippleCoverage(vec2 uvN, float wavelengthUm, float extentUm, float nPhases, float phaseStep) {
  // Convert crest spacing μm → normalized so the wake geometry (in normalized
  // coords) and the crest density stay consistent at any plate size.
  float wLenN = clamp(wavelengthUm / max(1.0, extentUm), 0.04, 0.4);
  float s = flowStreamlineField(uvN, wLenN, nPhases, phaseStep);
  float amp = flowAmplitude(uvN);
  float f = s - floor(s + 0.5);            // signed distance to nearest streamline
  float d = abs(f);
  // Preview crest half-width. Wider than the fab crest (which is floored to the
  // 2 µm litho minimum in the backend builder, not here) so the flowing ridges
  // read clearly through the 25%-open slit barrier at a resolvable zoom/scale.
  float crestHalf = 0.30 * amp;            // crest half-width (calm → vanishes)
  float aa = fwidth(s) + 1e-3;
  float crest = 1.0 - smoothstep(crestHalf, crestHalf + aa, d);
  return clamp(crest * step(0.05, amp), 0.0, 1.0);
}

// Map a face uv into the centerpiece ART-BOX (0..1, y-DOWN, matching the Python
// flow field). The art box is a (CENTERPIECE_FILL·aperture)-side square centered
// on the plate; uArtBoxHalfUv/CenterUv carry its uv half-extents + center. On a
// non-square face the box maps to different uv half-extents per axis. Fallback
// (0,0) → treat the whole face as the art box. Coordinates outside 0..1 mean the
// fragment is outside the box — see artBoxInside.
vec2 artBoxUV(vec2 uv) {
  vec2 artScale  = (uArtBoxHalfUv.x > 0.0) ? uArtBoxHalfUv : vec2(0.5);
  vec2 artCenter = (uArtBoxHalfUv.x > 0.0) ? uArtBoxCenterUv : vec2(0.5);
  vec2 uvArt = (uv - artCenter) / (2.0 * artScale) + vec2(0.5);
  uvArt.y = 1.0 - uvArt.y;   // vUv y-UP; flow field authored y-DOWN
  return uvArt;
}

// 1 inside the centerpiece art-box square, 0 outside. The barrier-interlace comb
// is gated on THIS, not on the A∪B silhouette union: a union-clipped comb is
// itself a static front image (its envelope is the union, and the front mask does
// not move with tilt, so no tilt angle can gate it out — the banned construction),
// and the comb must physically cover every column a back lane can slide under
// within the first zone. Both fab writers span the box for exactly this reason
// (plates.py `art_box & barrier_comb`, export_fine's barrier bars).
float artBoxInside(vec2 uvArt) {
  vec2 lo = step(vec2(0.0), uvArt);
  vec2 hi = step(uvArt, vec2(1.0));
  return lo.x * lo.y * hi.x * hi.y;
}

// x (µm) for the barrier lattice: measured from the FACE CENTRE and shifted by the
// solved registration phase, so open-slit centres land on k·uCenterPeriodUm +
// uSwitchBarrierPhaseUm and channel A starts on that boundary's +x side — the one
// convention plates.py::_barrier_masks, export_fine and the generators all use.
// (The frame gratings stay in raw face-uv µm; only the barrier is registered.)
vec2 barrierPUm(vec2 pUm) {
  return vec2(pUm.x - 0.5 * uExtentUm.x - uSwitchBarrierPhaseUm, pUm.y);
}

// Slit-comb / slit-barrier gold coverage along the x axis at physical point
// pUm (µm). One period `pitchUm`; the OPEN slot occupies the first `openFrac` of
// each period (gold BAR fills the remainder). Antialiased via fwidth; when a
// pixel spans more than ~a period it fades to the mean bar coverage (1-openFrac)
// so the fine comb reads as flat gold at the default (sub-pixel) zoom instead of
// aliasing — the barrier structure only resolves once the CAMERA (zoom / render
// pixel ratio) makes a period span several pixels. Pattern Scale deliberately does
// not touch this pitch (see uPatternScale), because scaling it would scale the
// switch tilt angle. This is the outer-plane half of both the water scanimation
// (openFrac = 1/N) and the barrier-interlace switch (openFrac = 0.5).
float slitBarCoverage(vec2 pUm, float pitchUm, float openFrac, float phase) {
  float coord = pUm.x / max(1.0, pitchUm) + phase;
  float f = fract(coord);
  float w = fwidth(coord);
  float aa = clamp(w, 0.0004, 0.5);
  // Gold where f >= openFrac (the closed bar). Two smoothstep edges keep the
  // slot open [0, openFrac) and the bar solid [openFrac, 1).
  float bar = smoothstep(openFrac, openFrac + aa, f);
  float collapse = smoothstep(0.35, 0.9, w);
  return mix(bar, 1.0 - openFrac, collapse);
}

// Single real surface (outer front OR inner back, per uLayer). Draws ONLY this
// layer's gold coverage from its own mask (bound to uFront) — no cross-layer
// sampling, no grating parallax. The cross-layer illusions emerge from the
// perspective projection of the two physical planes in the scene. Returns
// straight colour + alpha (= gold coverage), so gaps are transparent and the
// inner plane shows through the outer one.
vec4 runFoliageMoireLayer(vec3 viewTangent) {
  float mR = texture2D(uFront, vUv).r;   // THIS layer's own mask
  float oR = texture2D(uBack, vUv).r;    // the OTHER layer's mask (the switch's
                                         // second silhouette — see uSwitchInterlace)
  vec2 pUm = vUv * uExtentUm;
  float duty = clamp(uGratingDuty, 0.05, 0.95);
  bool isBack = uLayer > 0.5;

  // Pattern Scale: applied to the magnified-preview family only (frame carrier,
  // frame louvre, capybara body shimmer), on BOTH planes so their moiré geometry
  // stays self-consistent. The centerpiece pitch is EXCLUDED — see uPatternScale:
  // the T/n gap does not scale, so scaling p would scale the switch tilt angle and
  // the preview would answer the "does it swap at a hand tilt?" question wrong by
  // exactly the scale factor.
  float scale = max(uPatternScale, 0.01);
  float carrierP = uCarrierPeriodUm * scale;   // frame back carrier
  float slitP    = uSlitPeriodUm    * scale;   // frame front louvre
  float centerP  = uCenterPeriodUm;            // switch / water-comb / barrier pitch (exact fab)
  // Capybara body shimmer: its own fab period, magnified like the frame pair.
  float bodyP = (uWaterBodyPeriodUm > 0.0) ? (uWaterBodyPeriodUm * scale) : carrierP;

  // Shared region windows (see FRAME_MIN / RAINBOW_MIN / ART_MIN).
  float rainbowOn = step(0.0, uRainbowLevel);
  float rainbowHere = rainbowOn * step(RAINBOW_MIN, mR) * (1.0 - step(ART_MIN, mR));
  float band = step(FRAME_MIN, mR) * (1.0 - step(RAINBOW_MIN, mR)); // frame / carrier win
  band = max(band, rainbowHere);
  float art = step(ART_MIN, mR);                                    // centerpiece silhouette
  float artOther = step(ART_MIN, oR);                               // the switch's other image

  bool isWater = uWaterScanN > 0.5;
  bool isInterlace = uSwitchInterlace > 0.5;

  float cov = 0.0;
  // Shading buckets that decouple DISPLAY brightness from occlusion (alpha).
  // The water slit-barrier keeps a high alpha so it still occludes the inner
  // ripple lanes (the scanimation mechanism), but is DISPLAYED as a dim recessed
  // barrier so the bright water behind reads as the subject. The inner water
  // crests are DISPLAYED extra-bright (the flowing water) even though the inner
  // plane is otherwise the dim recessed layer.
  float dimCov = 0.0;   // rendered as a dim/recessed barrier
  float hotCov = 0.0;   // rendered extra-bright (flowing water)
  if (!isBack) {
    // OUTER plane: per-motif foliage louvre in the frame band …
    float bucket = 0.0;
    if (uFrameBucketStep > 0.0) {
      bucket = clamp(floor((mR - uFrameBucket0) / uFrameBucketStep + 0.5),
                     0.0, max(0.0, uFrameBucketCount - 1.0));
    }
    float frameAngle = uSlitAngle
      + (bucket - 0.5 * (uFrameBucketCount - 1.0)) * uFrameAngleSpan;
    float louvre = gratingCoverage(pUm, frameAngle, slitP, duty);
    cov = band * louvre;

    // … plus the centerpiece front-layer geometry.
    if (isWater) {
      // Capybara: the dry body (above the waterline) shimmers with a fine
      // carrier; the water band (below) carries the SLIT BARRIER comb (60 µm
      // pitch, 15 µm open slot = 1/N, 45 µm bar). The animated flow FALLS OUT of
      // this comb occluding the inner ripple lanes as the box tilts.
      vec2 uvArt = artBoxUV(vUv);
      float body  = art * step(uvArt.y, uWaterWaterlineY);
      float water = art * step(uWaterWaterlineY, uvArt.y);
      float shimmer = gratingCoverage(pUm, uSwitchAxis, bodyP, duty);
      // Water comb: the fab bake anchors this lattice on the water band, not on
      // the face centre, and publishes no phase for it — so it stays anchored with
      // its own inner ripple lanes below (both on raw face-uv µm), which is what
      // the scanimation mechanism needs. Absolute phase only picks which ripple
      // frame shows head-on.
      float comb = slitBarCoverage(pUm, centerP, 1.0 / uWaterScanN, 0.0);
      float combCov = water * comb;
      cov += body * shimmer + combCov;
      dimCov += combCov;   // barrier occludes (alpha) but displays recessed
    } else if (isInterlace) {
      // Barrier-interlace switch: neutral slit barrier (open duty 0.5) over the
      // FULL centerpiece art box (artBoxInside — NEVER the A∪B union). Exactly one
      // lane class shows per slot. The barrier stays opaque (occludes) but displays
      // recessed so the revealed image (A or B, on the inner plane) is what the eye
      // reads on tilt.
      float comb = slitBarCoverage(barrierPUm(pUm), centerP, 0.5, 0.25);
      float combCov = artBoxInside(artBoxUV(vUv)) * comb;
      cov += combCov;
      dimCov += combCov;
    } else {
      // Legacy 2-phase switch: front silhouette filled with the switch carrier.
      float front = gratingCoveragePhase(pUm, uSwitchAxis, centerP, duty, 0.0);
      cov += art * front;
    }
  } else {
    // INNER plane: uniform back carrier fills the window …
    float carrier = gratingCoverage(pUm, uCarrierAngle, carrierP, duty);
    cov = band * carrier;

    // … plus the centerpiece back-layer geometry.
    if (isWater) {
      // N interleaved ripple frames: slot k (a 1/N-wide lane of every comb
      // period) carries ripple phase k. The travelling current EMERGES when the
      // outer comb reveals successive slots with tilt.
      vec2 uvArt = artBoxUV(vUv);
      float artWidthUm = uExtentUm.x * (2.0 * ((uArtBoxHalfUv.x > 0.0) ? uArtBoxHalfUv.x : 0.5));
      float coord = pUm.x / max(1.0, centerP);
      float slotf = floor(fract(coord) * uWaterScanN);      // which phase this lane holds
      float laneCrest = waterRippleCoverage(uvArt, uWaterRippleWavelengthUm, artWidthUm,
                                            uWaterScanN, slotf);
      float meanCrest = waterRippleCoverage(uvArt, uWaterRippleWavelengthUm, artWidthUm,
                                            uWaterScanN, 0.0);
      float collapse = smoothstep(0.35, 0.9, fwidth(coord));
      float crestCov = art * mix(laneCrest, meanCrest, collapse);
      cov += crestCov;
      hotCov += crestCov;   // flowing water — display bright even though inner
    } else if (isInterlace) {
      // A in even lanes, B in odd (lane pitch = half the barrier pitch), on the
      // SAME registered lattice as the outer comb (barrierPUm) — that shared
      // lattice IS the registration: lane 0 starts at an open-slit centre, so head-on
      // every slit straddles an A|B boundary and ±tilt reveals one class cleanly.
      // Inner uFront = B (this layer's mask), uBack = A (the front silhouette). At the
      // default sub-pixel zoom the lanes collapse to both images half-shown (the
      // head-on interlace); zoom in and the discrete A|B lanes resolve.
      float aVal = artOther;            // A (front silhouette, via uBack)
      float bVal = art;                 // B (this back silhouette, via uFront)
      float lanePitch = 0.5 * centerP;
      float laneCoord = barrierPUm(pUm).x / max(1.0, lanePitch);
      float isOdd = mod(floor(laneCoord), 2.0);   // 0 = even → A, 1 = odd → B
      float laneCov = mix(aVal, bVal, isOdd);
      float meanCov = 0.5 * (aVal + bVal);
      float collapse = smoothstep(0.35, 0.9, fwidth(laneCoord));
      float ic = mix(laneCov, meanCov, collapse);
      cov += ic;
      hotCov += ic;   // the revealed image reads bright even on the inner plane
    } else {
      // Legacy 2-phase switch: back silhouette at half-period phase.
      float back = gratingCoveragePhase(pUm, uSwitchAxis, centerP, duty, 0.5);
      cov += art * back;
    }
  }

  float ndl = max(0.0, vLightDirTangent.z);
  vec3 color;
  if (uIllumination == 1) {
    // Laser transmission: bright where the layer is OPEN (gaps) — but the alpha
    // masks the gold, so tint the gold coverage instead for a consistent look.
    color = uLaserColor * cov * (0.5 + 0.5 * ndl);
  } else if (uIllumination == 2) {
    color = GOLD_BACK * cov * 0.4;
  } else {
    // Ambient. Outer plane = bright first-surface gold; inner plane is deeper /
    // dimmer so the bare carrier recedes behind the foliage. The dim/hot buckets
    // (water barrier / flowing water) override that default so the scanimation
    // reads: the barrier recedes and the water behind it glows.
    float normalCov = clamp(cov - dimCov - hotCov, 0.0, 1.0);
    vec3 tint = isBack ? mix(GOLD_BACK, GOLD, 0.55) : GOLD;
    float lift = isBack ? (0.22 + 0.5 * ndl) : (0.30 + 0.72 * ndl);
    color = tint * normalCov * lift;
    color += GOLD * 0.3 * normalCov * (0.3 + 0.7 * ndl) * (isBack ? 0.4 : 1.0); // glint
    // Recessed barrier: dim, so the flowing water behind it dominates the read.
    color += mix(GOLD_BACK, GOLD, 0.15) * dimCov * (0.14 + 0.22 * ndl);
    // Flowing water: bright first-surface gold regardless of plane.
    color += GOLD * hotCov * (0.45 + 0.7 * ndl);
    // Diffraction accent: OUTER plane only, labelled angle-hue sheen.
    if (!isBack && rainbowHere > 0.0) {
      color += diffractionSheen(viewTangent, pUm, ndl) * rainbowHere;
    }
  }
  return vec4(color, clamp(cov, 0.0, 1.0));
}

void main() {
  vec3 viewTangent = normalize(vViewDirTangent);

  // Recipe 3 (foliage_moire = every box face) is the TWO-PLANE geometric path:
  // it returns its own alpha so the two real surfaces composite in the scene.
  if (uRecipe == 3) {
    gl_FragColor = runFoliageMoireLayer(viewTangent);
    return;
  }

  // Legacy single-plane recipes (standalone-pattern previews) stay opaque.
  // (uRecipe == 2 no longer exists — phase_shift_overlay is retired.)
  vec3 color;
  if (uRecipe == 0) {
    color = runStereoLenticular(viewTangent);
  } else {
    // Default + uRecipe == 1: moire_interactive.
    color = runMoireInteractive(viewTangent);
  }

  gl_FragColor = vec4(color, 1.0);
}
