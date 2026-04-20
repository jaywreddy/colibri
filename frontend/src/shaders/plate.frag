precision highp float;

#include './lib/parallax.glsl'
#include './lib/wavelength_rgb.glsl'

varying vec2 vUv;
varying vec3 vViewDirTangent;
varying vec3 vLightDirTangent;
varying vec3 vNormalWorld;

uniform sampler2D uFront;
uniform sampler2D uBack;
uniform sampler2D uFftAtlas;   // optional — Tier 2 diffraction halo atlas
uniform float uUseFft;         // 0.0 = no FFT, 1.0 = composite FFT halo
uniform int uWavelengthSlot;   // which slab of atlas (0=R,1=G,2=B) to sample

uniform float uExtentUm;       // physical width of the plate surface (μm)
uniform float uThicknessUm;    // substrate thickness
uniform float uN;              // refractive index of substrate

uniform int uIllumination;     // 0=ambient, 1=laser, 2=backlight
uniform vec3 uLaserColor;
uniform vec3 uBacklightColor;
uniform vec3 uAmbientColor;

// Which render recipe to run. Numeric IDs match RECIPE_IDS in frontend/src/api.ts:
//   0 iridescent_grating, 1 stereo_lenticular, 2 moire_interactive,
//   3 near_field_carpet, 4 far_field_hologram, 5 stylized_amplitude.
uniform int uRecipe;

// --- iridescent_grating uniforms (only read when uRecipe == 0) --------------
uniform float uGratingPeriodUm;    // Λ (μm)
uniform float uGratingOrientation; // radians; period vector direction (0 = +X)
uniform float uLaserWavelengthUm;  // for laser-mode grating spot placement

// --- stereo_lenticular uniforms (only read when uRecipe == 1) ---------------
uniform sampler2D uViewA;          // left-eye / tilt-negative scene
uniform sampler2D uViewB;          // right-eye / tilt-positive scene
uniform float uSlitOrientation;    // radians; slit-normal direction (0 = +X)
uniform float uSlitPeriodUm;       // slit period Λ_slit (μm), for crossfade width

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

vec3 fftHaloColor(vec3 viewTangent) {
  float ax = clamp(viewTangent.x, -0.7, 0.7);  // sin of tilt-x
  float ay = clamp(viewTangent.y, -0.7, 0.7);
  vec2 local = vec2((ax + 0.7) / 1.4, (ay + 0.7) / 1.4);
  float slabW = 1.0 / 3.0;
  float slab = float(uWavelengthSlot);
  vec2 atlasUv = vec2(slab * slabW + local.x * slabW, local.y);
  return texture2D(uFftAtlas, atlasUv).rgb;
}

// ----------------------------------------------------------------------------
// Recipe 5 / legacy: flat-mask composite with Snell parallax. Using
// parallax_offset from lib/parallax.glsl fixes the historical
// compute-and-discard bug.
// ----------------------------------------------------------------------------
vec3 runStylized(vec3 viewTangent) {
  vec2 backUv = vUv - parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);
  vec4 f = texture2D(uFront, vUv);
  vec4 b = texture2D(uBack, backUv);
  float frontGold = f.r;
  float backGold = b.r;

  vec3 backlight = (uIllumination == 2) ? uBacklightColor : vec3(0.0);
  vec3 laserTint = (uIllumination == 1) ? uLaserColor : vec3(1.0);

  vec3 color = mix(
    mix(backlight, GOLD_BACK * laserTint, backGold),
    GOLD * laserTint,
    frontGold
  );

  if (uIllumination == 0) {
    color = ambientLit(color);
    color += (1.0 - viewTangent.z) * vec3(0.08, 0.07, 0.04) * frontGold;
  } else if (uIllumination == 1) {
    color *= 0.4 + 0.6 * max(0.0, vLightDirTangent.z);
    if (uUseFft > 0.5) {
      float aperture = (1.0 - frontGold) * (1.0 - backGold);
      color += fftHaloColor(viewTangent) * aperture * 0.9;
    }
  } else {
    float t = (1.0 - frontGold) * (1.0 - backGold);
    color = backlight * t + GOLD_BACK * 0.15 * backGold + GOLD * 0.25 * frontGold;
  }
  return color;
}

// ----------------------------------------------------------------------------
// Recipe 0: iridescent_grating — analytical diffraction-grating dispersion.
//
// For a reflective grating of period Λ, the m-th order satisfies
//   sin(θ_r) − sin(θ_i) = m λ / Λ
// where θ_i is the incident angle and θ_r the reflected/observed angle, both
// measured from the plate normal and projected onto the grating-period axis.
// Solving for λ given θ_i, θ_r, m:
//   λ_m = Λ (sin(θ_r) − sin(θ_i)) / m
// For each order in {-2,-1,1,2} we check whether λ_m lands in the visible
// range; if it does, we emit that color. In laser mode we instead gate by
// proximity to uLaserWavelengthUm.
// ----------------------------------------------------------------------------
vec3 runIridescent(vec3 viewTangent) {
  vec4 f = texture2D(uFront, vUv);
  vec4 b = texture2D(uBack, vUv);
  float mask = max(f.r, b.r); // pattern footprint — wherever there IS a grating

  // Project view + light directions onto the grating period axis (unit vec).
  vec2 axis = vec2(cos(uGratingOrientation), sin(uGratingOrientation));
  float sinViewProj  = dot(viewTangent.xy,  axis);
  float sinLightProj = dot(vLightDirTangent.xy, axis);

  float periodNm = uGratingPeriodUm * 1000.0; // Λ in nanometers

  vec3 rainbow = vec3(0.0);
  // Iterate a small fixed set of orders. In ambient broadband we accept every
  // valid λ; in laser we reward the angle where the laser's own λ satisfies.
  for (int m = -2; m <= 2; m++) {
    if (m == 0) continue;
    float lambda_nm = periodNm * (sinViewProj - sinLightProj) / float(m);
    if (lambda_nm < 380.0 || lambda_nm > 780.0) continue;
    if (uIllumination == 1) {
      // Laser: compute how close this order's required λ is to the user's
      // chosen laser wavelength; tint with the laser's color near the spot.
      float dist = abs(lambda_nm - uLaserWavelengthUm * 1000.0);
      float gate = smoothstep(40.0, 5.0, dist); // 5-40 nm falloff
      rainbow += uLaserColor * gate / float(m > 0 ? m : -m);
    } else {
      // Ambient: every valid λ contributes, weighted by 1/|m| (specular-ish).
      rainbow += wavelength_to_rgb(lambda_nm) / float(m > 0 ? m : -m);
    }
  }

  // The "base" gold reflection — without a grating there's just gold.
  vec3 goldBase = GOLD * (0.25 + 0.6 * max(0.0, vLightDirTangent.z));
  // Rainbow replaces part of the gold where the grating lives.
  vec3 color = mix(
    vec3(0.02, 0.02, 0.03),  // substrate void (no gold at this pixel)
    goldBase * 0.35 + rainbow * 0.9,
    mask
  );

  // Backlight: show transmission through the grating as faint colored rays.
  if (uIllumination == 2) {
    float t = 1.0 - mask;
    color = uBacklightColor * t * 0.9 + rainbow * mask * 0.6;
  }

  return color;
}

// ----------------------------------------------------------------------------
// Recipe 2: moire_interactive — sample front & back with physical parallax.
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
    // Ambient: reflected gold + a tiny transmission spill so the beat shows.
    vec3 goldShade = GOLD * reflected * (0.3 + 0.7 * max(0.0, vLightDirTangent.z));
    color = goldShade + vec3(0.04) * transmission;
    // Darken regions where both layers overlap (double gold); this is what
    // produces the classic moiré contrast.
    float overlap = frontGold * backGold;
    color *= (1.0 - 0.35 * overlap);
  } else if (uIllumination == 1) {
    // Laser: transmitted light shows moiré fringes brightly.
    color = uLaserColor * transmission * (0.45 + 0.55 * max(0.0, vLightDirTangent.z));
    color += GOLD * 0.12 * reflected;
  } else {
    // Backlight: classic moiré — only where both are open does light pass.
    color = uBacklightColor * transmission;
    color += GOLD_BACK * reflected * 0.25;
  }
  return color;
}

// ----------------------------------------------------------------------------
// Recipe 1: stereo_lenticular — pick between two pre-rendered scenes based on
// which side of the slit barrier the viewer's eye is on.
//
// A lenticular (or parallax-barrier) plate has a periodic slit array on the
// front; underneath lie two interlaced scenes. When the eye is off-axis by
// +angle the slits reveal one scene; by −angle the other. The switch angle
// is roughly ±arctan(p/2·t) where p = slit period, t = substrate thickness.
//
// We sample both view_a and view_b, Snell-shifted through the substrate, and
// blend them via the sign of the projected view angle on the slit-normal
// axis. A small smoothstep gives a soft crossfade near normal incidence so
// the transition isn't jarring. The front slit barrier still masks the
// result so the physical "louvre" look is preserved.
// ----------------------------------------------------------------------------
vec3 runStereoLenticular(vec3 viewTangent) {
  vec2 shift = parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);

  // Project the tangent-space view direction onto the slit-normal axis. The
  // sign of this scalar picks view_a vs view_b; the magnitude controls the
  // crossfade sharpness.
  vec2 slitNormal = vec2(cos(uSlitOrientation), sin(uSlitOrientation));
  float proj = dot(viewTangent.xy, slitNormal);

  // Crossfade width tuned so normal incidence (proj ≈ 0) softly mixes and
  // anything past ±0.15 is fully committed to one view. This is well inside
  // the switch angle arctan(p/2t) for typical (p, t) combos.
  float t = smoothstep(-0.15, 0.15, proj);

  float sA = texture2D(uViewA, vUv - shift).r;
  float sB = texture2D(uViewB, vUv - shift).r;
  float scene = mix(sA, sB, t);

  // Front slit barrier blocks the scene wherever gold covers the slit.
  float frontGold = texture2D(uFront, vUv).r;
  float transmission = (1.0 - frontGold) * scene;

  vec3 color;
  if (uIllumination == 0) {
    // Ambient: the exposed scene catches gold reflection; barrier is dark.
    vec3 sceneGold = GOLD * transmission * (0.35 + 0.7 * max(0.0, vLightDirTangent.z));
    vec3 barrier = GOLD * frontGold * (0.2 + 0.5 * max(0.0, vLightDirTangent.z)) * 0.6;
    color = sceneGold + barrier;
  } else if (uIllumination == 1) {
    // Laser: scene punches through bright; barrier absorbs.
    color = uLaserColor * transmission * (0.5 + 0.5 * max(0.0, vLightDirTangent.z));
    color += GOLD * 0.08 * frontGold;
  } else {
    // Backlight: transmitted light lives only where the scene is gold and
    // the slit is open — this is the classic lenticular viewing mode.
    color = uBacklightColor * transmission * 0.9;
    color += GOLD_BACK * frontGold * 0.18;
  }
  return color;
}

void main() {
  vec3 viewTangent = normalize(vViewDirTangent);

  vec3 color;
  if (uRecipe == 0) {
    color = runIridescent(viewTangent);
  } else if (uRecipe == 1) {
    color = runStereoLenticular(viewTangent);
  } else if (uRecipe == 2) {
    color = runMoireInteractive(viewTangent);
  } else {
    // Recipes 3, 4 not yet implemented — fall back to stylized_amplitude so
    // nothing regresses until phases D–E land.
    color = runStylized(viewTangent);
  }

  gl_FragColor = vec4(color, 1.0);
}
