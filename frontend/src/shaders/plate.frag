precision highp float;

#include './lib/parallax.glsl'

// ITEM 3 — world-space surface basis from plate.vert; the view and light directions
// are built PER FRAGMENT in main() and passed down as explicit parameters. The old
// vViewDirTangent / vLightDirTangent varyings are gone: they interpolated normalized
// directions across a 4-vertex quad spanning a whole 50 mm face (see plate.vert).
varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vNormalWorld;
varying vec3 vTangentWorld;
varying vec3 vBitangentWorld;

// Key-light direction in world space. Treated as a DIRECTIONAL light — see main().
uniform vec3 uLightWorld;

uniform sampler2D uFront;
uniform sampler2D uBack;

uniform vec2 uExtentUm;        // physical (width, height) of the plate surface (μm)
uniform float uThicknessUm;    // substrate thickness
uniform float uN;              // refractive index of substrate

uniform int uIllumination;     // 0=ambient, 1=laser, 2=backlight
uniform vec3 uLaserColor;
uniform vec3 uBacklightColor;
uniform vec3 uAmbientColor;    // ambient illuminant tint (linear; white today)

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

// --- litho metal (renderer-audit item 5) -------------------------------------
// The fabricated masks are metal-agnostic; the CONDUCTOR RESPONSE is not. These
// carry the per-box metal choice (spec.metal: gold | chrome | chrome-ar) into
// the foliage_moire shading: diffuse-lobe albedo, its dim second-surface
// sibling, and the Fresnel F0 that drives both the angular desaturation gain
// and the specular lobe's spectral character. Defaults are EXACTLY the legacy
// GOLD constants (bound at material creation), so a gold box renders
// bit-identically to the pre-uniform build and the @effects gates are
// untouched. Legacy single-plane recipes (0/1) keep the GOLD constants — they
// preview standalone patterns, not the box's fab metal.
uniform vec3 uMetalAlbedo;         // linear diffuse-lobe albedo (GOLD default)
uniform vec3 uMetalAlbedoBack;     // dim second-surface sibling (GOLD_BACK default)
uniform vec3 uMetalF0;             // normal-incidence Fresnel (GOLD_F0 default)

// Per-metal ENERGY SPLIT + grazing behaviour. Colour alone does not make a
// conductor read as itself: what separates mask chrome from gold is how the
// reflected energy divides between the broad body lobe and the tight mirror
// lobe, and how the reflectance climbs toward grazing.
//
//   uMetalBody    weight of the broad (rough-scatter) lobe. A near-mirror film
//                 puts LESS energy here — the missing energy shows up in the
//                 specular lobe instead. Grey body + weak highlight is exactly
//                 what makes a metal read as PAINT.
//   uMetalSpec    weight of the half-vector lobe.
//   uMetalGloss   its exponent (surface smoothness proxy).
//   uMetalGrazing reflectance the film approaches at 90° incidence. 1.0 for a
//                 BARE conductor (gold, chrome — Fresnel really does go to
//                 unity). An AR-coated mask chrome is NOT bare: its low
//                 reflectance is a thin-film interference + absorption stack
//                 that degrades toward grazing but never reaches unity, so it
//                 gets a real ceiling. Without one, the normalized gain F/F0
//                 blows a 0.06-F0 AR film up by up to 17x and it renders
//                 IDENTICALLY to bright chrome — the bug this fixes.
//   uMetalSheen   diffraction-accent efficiency, scaled by the film's own
//                 reflectance (a rainbow blazing off near-black AR is wrong).
//
// Defaults are gold's numbers EXACTLY (1.0 / 0.5 / 80.0 / 1.0 / 1.0), so the
// gold path is arithmetically unchanged and the @effects gates are untouched.
uniform float uMetalBody;
uniform float uMetalSpec;
uniform float uMetalGloss;
uniform float uMetalGrazing;
uniform float uMetalSheen;

// ENVIRONMENT REFLECTION weight (uMetalEnv) + the surround it reflects.
//
// Until now this shader lit the litho metal with a SINGLE directional light and
// nothing else. Gold survives that because its identity lives in a warm broad
// body lobe — but a smooth mirror film has almost no body, and a tight
// half-vector lobe fires only in a narrow band, so chrome rendered as flat grey
// paint no matter how its colour was set. What actually makes chrome read as
// chrome is that it REFLECTS THE ROOM.
//
// uMetalEnv is the mirror-ness of the film — the same parameter family as
// body/spec/gloss: energy this deposit returns as an environment reflection
// rather than broad scatter. It is 0 for gold, whose look is already calibrated
// around the body lobe (and whose pixels the @effects gates pin), and high for
// the smooth chrome mask film. uSky/uGroundColor mirror the scene's
// HemisphereLight, so the plate metal and the foil/solder/hinge read as lit by
// one room instead of two.
//
// Honest by the renderer contract: view-dependent (it is the reflection
// vector), time-invariant (no time term), litho-mask-driven (gated by
// normalCov, so it exists only where metal exists) and geometric.
uniform float uMetalEnv;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;

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

// --- physically baked diffraction (see backend app/diffraction.py) ----------
// The accent's colour is no longer invented here. The backend integrates the
// grating equation, the square-wave order series and the CIE colour matching
// functions into a 1-D table indexed by the OPTICAL PATH TERM
//
//     u = period * ( dot(V, g) + dot(L, g) )        [um]
//
// where g is the in-plane grating vector. Order m lands in the eye at
// lambda = u/m, so u carries every geometric dependency and ONE table serves
// any pitch. uRainbowPeriodUm / uRainbowAngleRad are the REAL fabricated
// grating's parameters, published in recipe_data straight from the constants
// the mask is baked with — change the fab grating and this preview changes
// with it, which the old hand-tuned hue ramp could not do.
uniform sampler2D uDiffLut;
uniform float uDiffUMax;           // u (um) at the last table entry
uniform float uDiffReady;          // 1 once the table has been uploaded
uniform float uRainbowPeriodUm;    // fabricated accent-grating period
uniform float uRainbowAngleRad;    // fabricated accent-grating orientation
uniform float uRainbowZeroOrder;   // eta_0 = duty^2: share left in specular
// Accent INTERLEAVE (see backend gratings.band_select / export_fine). The accent
// zone carries the diffraction grating and a moire louvre in alternating
// sub-acuity bands, both written at the accent's 45 deg axis so they share one
// lattice. Drawn here from the SAME published numbers the mask is baked with,
// so the preview stops showing a blend the part does not have.
uniform float uAccentBandPitchUm;  // interleave band pitch
uniform float uAccentMoirePeriodUm;// the moire band's louvre period

// ITEM 2b — these are LINEAR-LIGHT reflectances. They used to be sRGB display
// codes multiplied by lighting terms and written straight to the framebuffer, i.e.
// the whole plate was lit in GAMMA space: a `lift` of 0.30 on an sRGB code is only
// 0.30^2.4 ~= 0.065 of full linear radiance — ~4.6x darker than intended, which is
// why the terminator was harsh and why the rig needed a 0.30 ambient floor and an
// AmbientLight(0.5) to compensate. main() now runs <tonemapping_fragment> +
// <colorspace_fragment>, so this file is linear throughout and the plates share ONE
// transfer function with the metalwork for the first time.
//
// The values are the exact sRGB->linear images of the previous constants, so the old
// LOOK is the porting baseline; the lighting coefficients below were then re-derived
// to hold displayed mid-tone brightness (see runFoliageMoireLayer's ambient branch).
// Moving GOLD to measured Au F0 is a separate art call — item 4 does that for the
// SPECULAR lobe only, where F0 is the physically meaningful quantity.
const vec3 GOLD = vec3(0.791, 0.503, 0.080);       // was sRGB (0.902, 0.737, 0.314)
const vec3 GOLD_BACK = vec3(0.133, 0.084, 0.013);  // was sRGB (0.4,   0.32,  0.12)

// ITEM 4 — normal-incidence Fresnel reflectance (F0) of evaporated gold, linear.
// This is a MEASURED physical constant, not an art value: it is what sets both the
// colour of the specular lobe and the rate at which the metal desaturates toward
// white as the view goes grazing, which is the actual visual signature of gold.
// Kept separate from GOLD (the diffuse-lobe albedo, which is the ported old look) so
// the head-on appearance the item-2 retune calibrated is not disturbed.
const vec3 GOLD_F0 = vec3(1.000, 0.766, 0.336);

// (The `ambientLit` helper that used to sit here was DEAD — nothing ever called it —
// and it read the two varyings item 3 removed. It did contain the only correct
// half-vector specular in the file, which is now implemented for real, inline, in
// runFoliageMoireLayer's ambient branch. Deliberately deleted rather than fixed up:
// a second, divergent copy of the lighting model is exactly how the branch drifted
// into having no view-dependent response at all.)

// ----------------------------------------------------------------------------
// Recipe 1: moire_interactive — sample front & back with physical parallax.
// The Snell-refracted shift means rotating/orbiting the camera actually
// produces moving moiré fringes.
// ----------------------------------------------------------------------------
vec3 runMoireInteractive(vec3 viewTangent, vec3 lightTangent) {
  vec2 shift = parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);
  float frontGold = texture2D(uFront, vUv).r;
  float backGold  = texture2D(uBack,  vUv - shift).r;

  // Transmission through *both* apertures — this is where moiré fringes
  // show up as bright/dark beats.
  float transmission = (1.0 - frontGold) * (1.0 - backGold);
  float reflected    = max(frontGold, backGold * 0.55);

  vec3 color;
  if (uIllumination == 0) {
    vec3 goldShade = GOLD * reflected * (0.3 + 0.7 * max(0.0, lightTangent.z));
    color = goldShade + vec3(0.04) * transmission;
    float overlap = frontGold * backGold;
    color *= (1.0 - 0.35 * overlap);
  } else if (uIllumination == 1) {
    color = uLaserColor * transmission * (0.45 + 0.55 * max(0.0, lightTangent.z));
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
vec3 runStereoLenticular(vec3 viewTangent, vec3 lightTangent) {
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
    vec3 sceneGold = GOLD * transmission * (0.35 + 0.7 * max(0.0, lightTangent.z));
    vec3 barrier = GOLD * frontGold * (0.2 + 0.5 * max(0.0, lightTangent.z)) * 0.6;
    color = sceneGold + barrier;
  } else if (uIllumination == 1) {
    color = uLaserColor * transmission * (0.5 + 0.5 * max(0.0, lightTangent.z));
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
// ITEM 6 — EXACT box prefilter for a duty-cycle pulse train.
//
// Both analytic gratings used to prefilter with a pair of smoothsteps plus an ad-hoc
// `collapse = smoothstep(0.35, 0.9, w)` fade to the mean. That fade starts destroying
// real structure at w ~= 0.35 periods/pixel — well BELOW the w = 0.5 Nyquist limit —
// so fringe contrast was being thrown away in a band where it is still legitimately
// representable, and was hard-flattened above w = 0.9 where a true box filter still
// carries a decaying |sin(pi*w*duty)|/(pi*w) ripple.
//
// `pulseIntegral` is the antiderivative of the unit-period pulse (gold where
// fract(x) is in [0, duty)); differencing it across the pixel footprint is the exact
// area average. Verified: G(0) = 0, G(duty) = duty, G(1) = duty, and the average
// equals `duty` EXACTLY at w = 1 (and at every integer w), decaying to it in between —
// so collapse-to-the-mean is now exactly monotone in w rather than a tuned guess.
float pulseIntegral(float x, float duty) {
  return floor(x) * duty + min(fract(x), duty);
}

// Box average of the pulse train over a window of width `w` centred on `coord`.
//
// PRECISION: `coord` reaches a few thousand periods on a 50 mm face at a 20 um pitch,
// and differencing two pulseIntegral values of that magnitude would catastrophically
// cancel for a small w (float32 has ~1e-4 resolution at 1250, which is the same order
// as the window itself). So reduce to one period FIRST — the box average is periodic
// in `coord` with period 1 — and only then integrate over the span. All magnitudes
// then scale with `w`, not with `coord`.
float boxPulse(float coord, float duty, float w) {
  w = max(w, 1e-4);
  float lo = fract(coord - 0.5 * w);          // window start, reduced into [0, 1)
  // pulseIntegral(lo) collapses to min(lo, duty) because floor(lo) == 0.
  return clamp((pulseIntegral(lo + w, duty) - min(lo, duty)) / w, 0.0, 1.0);
}

float gratingCoveragePhase(vec2 pUm, float angle, float periodUm, float duty, float phase) {
  float c = cos(angle);
  float s = sin(angle);
  float coord = (pUm.x * c + pUm.y * s) / max(1.0, periodUm) + phase; // in periods
  // fwidth(coord) is exactly the right support: the projected extent of the pixel
  // parallelogram onto the grating axis is |dFdx| + |dFdy|. No AA floor is needed —
  // fwidth already IS one pixel, so the ramp is one pixel wide by construction, which
  // is the minimal correct antialiasing.
  return boxPulse(coord, duty, fwidth(coord));
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
vec3 diffractionSheen(vec3 viewTangent, vec3 lightTangent, vec2 pUm, float ndl) {
  // Grating vector g in the surface tangent frame, from the FABRICATED angle.
  vec2 g = vec2(cos(uRainbowAngleRad), sin(uRainbowAngleRad));

  // Optical path term. viewTangent/lightTangent are unit directions in the
  // tangent frame, so their .xy ARE the sines of the angles from the normal;
  // projecting on g gives exactly the grating-equation terms. Because the
  // camera is perspective, viewTangent varies per fragment, so u varies across
  // the zone — that is what makes the spectrum SWEEP across the accent as the
  // piece moves, rather than flashing it as one flat colour.
  float u = uRainbowPeriodUm * (dot(viewTangent.xy, g) + dot(lightTangent.xy, g));

  // |u|: order m and -m are mirror images about the specular direction (u = 0).
  float idx = clamp(abs(u) / max(uDiffUMax, 1e-3), 0.0, 1.0);
  vec3 spectral = texture2D(uDiffLut, vec2(idx, 0.5)).rgb;

  // Lambertian-ish incidence falloff, matching the rest of the ambient branch.
  return spectral * (0.35 + 0.65 * ndl) * uDiffReady;
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
  // ITEM 6 — exact box average (see boxPulse). boxPulse returns the OPEN-slot
  // coverage, i.e. the pulse that is gold-free over [0, openFrac); this function
  // returns the gold BAR, so it is the complement. Collapses to the mean bar coverage
  // (1 - openFrac) exactly at w = 1 rather than being faded there by hand.
  return 1.0 - boxPulse(coord, openFrac, fwidth(coord));
}

// Single real surface (outer front OR inner back, per uLayer). Draws ONLY this
// layer's gold coverage from its own mask (bound to uFront) — no cross-layer
// sampling, no grating parallax. The cross-layer illusions emerge from the
// perspective projection of the two physical planes in the scene. Returns
// straight colour + alpha (= gold coverage), so gaps are transparent and the
// inner plane shows through the outer one.
vec4 runFoliageMoireLayer(vec3 viewTangent, vec3 lightTangent, float envUp) {
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
  // Fraction of this fragment sitting in the accent's DIFFRACTION band; only
  // that share returns a spectrum (the other band is a plain moire louvre).
  float accentDiffFrac = 0.0;
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
    // The accent is NOT frame band: it carries its own interleaved pair at the
    // accent axis. Draw the frame louvre only where the frame actually is.
    float frameOnly = band * (1.0 - rainbowHere);
    cov = frameOnly * louvre;
    if (rainbowHere > 0.0) {
      // Which interleave band is this fragment in? Bands are perpendicular to
      // the accent axis, so the selector is the same projection the gratings use.
      vec2 ag = vec2(cos(uRainbowAngleRad), sin(uRainbowAngleRad));
      float bandCoord = dot(pUm, ag) / max(1.0, uAccentBandPitchUm);
      float inDiff = boxPulse(bandCoord, 0.5, fwidth(bandCoord));
      float diffCov = gratingCoverage(pUm, uRainbowAngleRad, uRainbowPeriodUm, duty);
      float moireCov = gratingCoverage(pUm, uRainbowAngleRad, uAccentMoirePeriodUm, duty);
      cov += rainbowHere * mix(moireCov, diffCov, inDiff);
      // Only the diffraction band diffracts; the sheen is weighted by how much
      // of this fragment's footprint that band actually occupies.
      accentDiffFrac = rainbowHere * inDiff;
    }

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
      // BAND-LIMITED slot mix (renderer-audit item 1, sibling of the interlace
      // lane fix above). The old path hard-picked ONE phase slot with floor()
      // and faded to the phase-0 pattern via the same retired smoothstep window
      // — under-filtered lattice beating into rings, and the w→∞ limit was the
      // WRONG pattern (phase 0, not the phase average). Slot k occupies
      // fract(coord) ∈ [k/N, (k+1)/N), so its pixel-footprint occupancy is
      // boxPulse(coord - k/N, 1/N, w); the occupancies partition the footprint
      // (they sum to 1), so weighting each phase's crest field by its occupancy
      // is the exact box-filtered selector. Fixed 4-iteration loop with a step()
      // gate (N is 1..4; GLSL ES 1.00 wants constant bounds).
      float wSlot = fwidth(coord);
      float nPh = max(1.0, uWaterScanN);
      float laneCrest = 0.0;
      for (int k = 0; k < 4; k++) {
        float fk = float(k);
        float valid = step(fk + 0.5, nPh);
        float occ = valid * boxPulse(coord - fk / nPh, 1.0 / nPh, wSlot);
        laneCrest += occ * waterRippleCoverage(uvArt, uWaterRippleWavelengthUm, artWidthUm,
                                               uWaterScanN, fk);
      }
      float crestCov = art * laneCrest;
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
      //
      // BAND-LIMITED lane parity (renderer-audit item 1). The old selector was a
      // hard floor(laneCoord) mod 2 with an ad-hoc smoothstep(0.35, 0.9, fwidth)
      // fade to the mean — the same tuned-guess construction ITEM 6 already
      // retired for the gratings, and the last unfiltered lattice in the file.
      // Under-filtered, it beat against the screen pixel grid into wood-grain
      // interference rings across the whole centerpiece at mid zoom (screen-space
      // aliasing masquerading as physical moiré). The odd-lane parity is a duty-0.5
      // pulse train with period TWO lanes, so the exact pixel-footprint average is
      // the same boxPulse machinery: fract((laneCoord - 1) / 2) in [0, 0.5) ⇔ the
      // lane is odd. Hard lane pick as w → 0, EXACTLY the 50/50 mean at every
      // integer window — monotone in w, no tuned window, no ring band.
      float aVal = artOther;            // A (front silhouette, via uBack)
      float bVal = art;                 // B (this back silhouette, via uFront)
      float lanePitch = 0.5 * centerP;
      float laneCoord = barrierPUm(pUm).x / max(1.0, lanePitch);
      float oddFrac = boxPulse(0.5 * (laneCoord - 1.0), 0.5, 0.5 * fwidth(laneCoord));
      float ic = mix(aVal, bVal, oddFrac);
      cov += ic;
      hotCov += ic;   // the revealed image reads bright even on the inner plane
    } else {
      // Legacy 2-phase switch: back silhouette at half-period phase.
      float back = gratingCoveragePhase(pUm, uSwitchAxis, centerP, duty, 0.5);
      cov += art * back;
    }
  }

  float ndl = max(0.0, lightTangent.z);
  vec3 color;
  if (uIllumination == 1) {
    // Laser transmission: bright where the layer is OPEN (gaps) — but the alpha
    // masks the gold, so tint the gold coverage instead for a consistent look.
    color = uLaserColor * cov * (0.5 + 0.5 * ndl);
  } else if (uIllumination == 2) {
    color = uMetalAlbedoBack * cov * 0.4;
  } else {
    // Ambient. Both planes are the SAME gold under the same lighting; the inner one
    // is darker only by its physical two-interface transmission (item 5's T2), which
    // is ~0.93 head-on and falls toward 0.42 at 80° — so the recession is now a real
    // view-dependent second-surface cue rather than a fixed dimming factor. The
    // dim/hot buckets (water barrier / flowing water) still override the default so
    // the scanimation reads: the barrier recedes and the water behind it glows.
    //
    // NOTE for the visual re-check: retiring the old ~40% arbitrary back-plane
    // dimming raises the inner plane's baseline substantially. That is the physically
    // correct budget, but the dimCov/hotCov bucket balance was tuned against the old
    // constants, so the water-barrier vs. flowing-water read is the one thing here
    // that wants eyes on it (see the commit message).
    // ITEM 2d — coefficients RE-DERIVED for linear-light shading. Every constant
    // here was originally hand-tuned against display-space output; with GOLD now
    // linear and <tonemapping_fragment>/<colorspace_fragment> in main(), the same
    // numbers would read wrong at both ends (dark side washed out, because the old
    // 0.30 floor was compensating for gamma-space lighting; highlights rolled off,
    // because they used to hard-clip). The retune target is the PREVIOUS BUILD'S
    // DISPLAYED MID-TONE BRIGHTNESS, so measured @effects deltas stay in the same
    // band: at ndl = 0.5, full coverage, the outer plane lands within ~0.5% of the
    // old displayed channel mean, the inner plane within ~1%, and the dim/hot buckets
    // within ~1%. The dark end comes back ~5% dim and the peak ~13% dim — the peak is
    // NeutralToneMapping's shoulder replacing a hard clip, which is the entire point:
    // the clipped peak used to collapse R and G to near-equal and read yellow-white
    // instead of gold.
    //
    // ITEM 4 — real conductor response. This branch previously NEVER read the view
    // direction (except inside diffractionSheen): its whole shading was
    // tint * normalCov * lift plus a second term LABELLED "glint" that used ndl, not
    // a half-vector, so it was not specular at all. Net effect: the gold litho layer
    // had zero view-dependent material response and no specular lobe — flat diffuse
    // paint. Real evaporated gold on quartz is a near-mirror conductor whose
    // reflectance rises toward unity and desaturates toward white at grazing
    // incidence, and that angular behaviour IS what makes gold read as gold.
    //
    // This is MATERIAL RESPONSE, not a synthesised optical effect: view-dependent,
    // time-invariant (no time term anywhere), litho-mask-driven (every term is gated
    // by normalCov, so it appears only where gold actually exists) and geometric.
    // Categorically different from the self-admitted preview stand-in in
    // diffractionSheen.
    // clamp (not max): fp drift can push a normalized z fractionally above 1.0,
    // and GLSL pow() is undefined for a negative base — NaN on some drivers.
    float ndv = clamp(viewTangent.z, 0.0, 1.0);
    float schlick = pow(1.0 - ndv, 5.0);
    // Metal-aware conductor response (item 5 of the renderer audit): the
    // uniforms default to the GOLD constants, so gold is bit-identical.
    // Grazing CEILING (uMetalGrazing): 1.0 reproduces the bare-conductor
    // Schlick term exactly (gold, chrome); an AR stack tops out far lower.
    vec3 F = uMetalF0 + (vec3(uMetalGrazing) - uMetalF0) * schlick;
    // NORMALIZED Fresnel for the diffuse-ish lobe: exactly 1.0 at normal incidence,
    // rising toward 1/F0 = (1.0, 1.31, 2.98) at grazing. Folding it in this way adds
    // gold's correct angular desaturation WITHOUT shifting the head-on brightness the
    // item-2 retune just calibrated. Over the ±14° cone the suite samples, schlick
    // runs 0 -> 0.031, so this gain runs 1.0 -> ~(1.00, 1.01, 1.06): a physically
    // correct term that is near-constant everywhere the tests look while varying
    // strongly at the 60-80° views a user actually orbits to.
    vec3 fresnelGain = F / max(uMetalF0, vec3(1e-3));
    // A REAL half-vector specular lobe, carrying gold's own spectral character (F)
    // rather than the white highlight a naive rig would give — a white highlight on
    // gold is precisely what made the metal read as chrome.
    //
    // Guard the normalize: when the light is exactly opposite the view direction the
    // sum is the zero vector and normalize() returns NaN, which would propagate
    // straight into gl_FragColor as a hard artifact (and poison the pixel metrics).
    // There is no specular lobe in that configuration anyway.
    vec3 hSum = lightTangent + viewTangent;
    float hLen = length(hSum);
    float spec = (hLen > 1e-4) ? pow(max(0.0, hSum.z / hLen), uMetalGloss) : 0.0;

    // ITEM 5 — the inner plane's dimming is now REAL SECOND-SURFACE PHYSICS instead
    // of hand-picked numbers. It used to be tint = mix(GOLD_BACK, GOLD, 0.55) and a
    // separate, lower `lift` — roughly 40% arbitrary dimming with no derivation. The
    // correct budget for a layer on the far surface is TWO air/quartz transmissions:
    // light enters the front face (1 - Rq), reflects off the back gold, and exits the
    // front face (1 - Rq again, by reciprocity) — so (1 - Rq)^2.
    //
    // R0 is DERIVED from uN (bound from the manifest), not hardcoded, so a different
    // substrate index is honored automatically instead of silently keeping fused
    // silica's numbers.
    float r0 = (uN - 1.0) / (uN + 1.0);
    r0 = r0 * r0;                                   // 0.035 at n = 1.46
    float Rq = r0 + (1.0 - r0) * schlick;           // Schlick, shares item 4's term
    float T2 = (1.0 - Rq) * (1.0 - Rq);
    // Outer gold is deposited on the AIR-side face, so nothing attenuates it.
    float layerT = isBack ? T2 : 1.0;
    //
    // DELIBERATE DEVIATION from the plan, which also asked for `Rq` as a veiling-glare
    // term on the outer plane: there is no physical source for it here. The outer gold
    // sits on the air-exposed surface, so there is no air/quartz interface ABOVE it to
    // reflect a veil, and where the gold does NOT cover, the quartz slab mesh
    // (MeshPhysicalMaterial, its own ior/Fresnel) already renders that first-surface
    // flare itself. Adding it in this shader would be a fabricated effect and/or a
    // double count, which the honesty contract forbids. The view-dependent depth cue
    // survives intact through T2 alone.

    float normalCov = clamp(cov - dimCov - hotCov, 0.0, 1.0);
    // The old "glint" term's BROAD (ndl-driven) energy is folded into `lift` here,
    // because the specular replacing it is a tight pow(.,80) lobe that carries almost
    // none of it: 0.155 + 1.00*ndl plus the glint's 0.045 + 0.30*ndl = the
    // 0.20 + 1.30*ndl item 2 solved for. There is no longer an isBack variant — both
    // planes are the same gold under the same lighting, and the ONLY thing that makes
    // the inner one darker is now T2 below.
    float lift = 0.20 + 1.30 * ndl;
    // ENERGY SPLIT in the accent zone. A lamellar grating leaves only
    // eta_0 = duty^2 of its return in the zeroth (specular) order; the rest is
    // redistributed into the diffracted orders — which is precisely the
    // spectral term added below. So inside the accent the ordinary metal
    // response is scaled down by eta_0 and the rainbow takes over, rather than
    // being tinted on top of full-strength metal (which washed it out).
    float zeroOrder = mix(1.0, uRainbowZeroOrder, accentDiffFrac);
    color = uMetalAlbedo * normalCov * lift * uMetalBody * fresnelGain * zeroOrder;
    color += F * spec * normalCov * uMetalSpec * ndl * zeroOrder;
    // Environment reflection: a sky/ground hemisphere sampled along the mirror
    // direction and tinted by the film's own Fresnel F, so it carries the
    // metal's spectral character AND brightens toward grazing — the two cues
    // that read as "polished conductor" rather than "grey paint". Exactly zero
    // on gold (uMetalEnv = 0), so that path is untouched.
    if (uMetalEnv > 0.0) {
      vec3 envCol = mix(uGroundColor, uSkyColor, smoothstep(-0.55, 0.75, envUp));
      color += F * envCol * uMetalEnv * normalCov * zeroOrder;
    }
    // Recessed barrier: dim, so the flowing water behind it dominates the read.
    color += mix(uMetalAlbedoBack, uMetalAlbedo, 0.15) * dimCov * (0.05 + 0.08 * ndl);
    // Flowing water: bright first-surface gold regardless of plane.
    color += uMetalAlbedo * hotCov * (0.43 + 0.67 * ndl);
    // Diffraction accent: OUTER plane only, labelled angle-hue sheen.
    if (!isBack && rainbowHere > 0.0) {
      color += diffractionSheen(viewTangent, lightTangent, pUm, ndl) * accentDiffFrac * uMetalSheen;
    }
    // Ambient illuminant tint. uAmbientColor was bound by BoxScene but read by no
    // branch in any recipe — inert plumbing. Consume it here (rather than delete the
    // uniform, which is part of the material's bound surface) so the ambient
    // illuminant can actually be coloured. It is white today, and Color.setHex
    // already converts sRGB -> linear working space, so this is a no-op at the
    // current binding and correct if that binding ever changes. Applied after the
    // sheen on purpose: the illuminant spectrum modulates the diffracted spectrum too.
    //
    // `layerT` (item 5) rides along here so it attenuates the WHOLE second-surface
    // radiance, not just the main lobe — the hot bucket (inner water crests, inner
    // interlace reveal) travels the same two-interface path. It is exactly 1.0 on the
    // outer plane, so the dim bucket and the diffraction sheen (both outer-only) are
    // untouched by construction.
    color *= uAmbientColor * layerT;
  }
  // HASHED alpha-to-coverage (renderer-audit item 1, second half). The lattice
  // functions above are exactly box-filtered, but a FRACTIONAL coverage still
  // has to leave this shader as alpha, and alphaToCoverage quantizes it into
  // the GPU's ORDERED per-pixel MSAA sample patterns — smooth coverage ramps
  // then band into wood-grain interference rings across the sub-resolved combs
  // (screen-space dither moiré masquerading as physical moiré). Standard fix:
  // decorrelate with a static screen-space hash so the quantization error is
  // fine uniform grain instead of structured rings. The hash has NO time term
  // and no view term — bit-identical frame to frame, so time-invariance and
  // the idle-frame contract hold. The 4·a·(1−a) gate is zero at a = 0 and
  // a = 1: solid gold, bare glass, and the hard NEAREST-sampled mask edges are
  // untouched; only interior fractional-coverage regions (exactly where the
  // rings lived) are dithered.
  float aCov = clamp(cov, 0.0, 1.0);
  float hashA = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  aCov = clamp(aCov + (hashA - 0.5) * (4.0 * aCov * (1.0 - aCov)) * 0.5, 0.0, 1.0);
  return vec4(color, aCov);
}

void main() {
  // ITEM 3 — build the view and light directions PER FRAGMENT, then project them onto
  // the interpolated world basis. `cameraPosition` is declared by three's own fragment
  // prefix, so this needs no new uniform.
  vec3 tangentWorld   = normalize(vTangentWorld);
  vec3 bitangentWorld = normalize(vBitangentWorld);
  vec3 normalWorld    = normalize(vNormalWorld);

  vec3 viewDirWorld = normalize(cameraPosition - vWorldPos);
  // DIRECTIONAL, not positional. plate.vert used to treat uLightWorld as a POINT
  // light (normalize(uLightWorld - worldPos)) while the scene's keyLight is a
  // THREE.DirectionalLight fed that very same vector — the plate and the PBR rig
  // disagreed about what the uniform means. Numerically this is near-neutral today
  // (the root scale of ~0.028 makes plate world positions tiny against the 3-unit
  // light distance), which is exactly why it is safe to unify now, before the
  // divergence can grow into something that has to be untangled under a regression.
  vec3 lightDirWorld = normalize(uLightWorld);

  vec3 viewTangent = vec3(
    dot(viewDirWorld, tangentWorld),
    dot(viewDirWorld, bitangentWorld),
    dot(viewDirWorld, normalWorld)
  );
  vec3 lightTangent = vec3(
    dot(lightDirWorld, tangentWorld),
    dot(lightDirWorld, bitangentWorld),
    dot(lightDirWorld, normalWorld)
  );

  // Mirror direction for the environment term (uMetalEnv): where a specular ray
  // leaving this fragment toward the eye came FROM. Only its world-up component
  // is needed — the surround is a sky/ground hemisphere gradient.
  float envUp = reflect(-viewDirWorld, normalWorld).y;

  // ITEM 2c — SINGLE exit point so every recipe runs the colour pipeline. The three
  // chunks below operate on `gl_FragColor` BY NAME, so the old recipe-3 early
  // `return` would have skipped them; hence the if/else-if/else shape rather than an
  // early-out. Recipe 3 (foliage_moire = every box face) is the TWO-PLANE geometric
  // path and returns its own alpha so the two real surfaces composite in the scene;
  // the legacy single-plane recipes (standalone-pattern previews) stay opaque.
  // (uRecipe == 2 no longer exists — phase_shift_overlay is retired.)
  if (uRecipe == 3) {
    gl_FragColor = runFoliageMoireLayer(viewTangent, lightTangent, envUp);
  } else if (uRecipe == 0) {
    gl_FragColor = vec4(runStereoLenticular(viewTangent, lightTangent), 1.0);
  } else {
    // Default + uRecipe == 1: moire_interactive.
    gl_FragColor = vec4(runMoireInteractive(viewTangent, lightTangent), 1.0);
  }

  // makePlateShader builds a ShaderMaterial (NOT RawShaderMaterial), so three's full
  // fragment prefix — toneMapping(), toneMappingExposure, linearToOutputTexel() — is
  // already prepended and resolveIncludes() runs on this file. The angle-bracket form
  // survives vite-plugin-glsl untouched: its include regex character class explicitly
  // excludes '<' and '>', so only the './lib/parallax.glsl' form above is its
  // business. No build-tooling change is needed.
  //
  // Per-pass correctness (this is what makes the two-plane renderer safe): three
  // disables tone mapping and forces LinearSRGBColorSpace whenever it is rendering
  // into a render target. These two chunks therefore write LINEAR, un-tone-mapped
  // values into the glass slab's transmission backdrop RT — where the inner plane
  // lives — and tone-mapped, sRGB-encoded values into the default framebuffer. The
  // inner plane gets tone-mapped exactly once, by the glass fragment that composites
  // it; the outer plane exactly once, directly. No double application.
  //
  // sRGBTransferOETF passes .a through untouched, so alphaToCoverage (which reads
  // the gold-coverage alpha) is unaffected.
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
