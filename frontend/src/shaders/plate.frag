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
//   0 stereo_lenticular, 1 moire_interactive, 2 phase_shift_overlay.
uniform int uRecipe;

// --- stereo_lenticular uniforms (only read when uRecipe == 0) ---------------
uniform sampler2D uViewA;          // tilt-positive scene
uniform sampler2D uViewB;          // tilt-negative scene
uniform float uSlitOrientation;    // radians; slit-normal direction (0 = +X)
uniform float uSlitPeriodUm;       // slit period Λ_slit (μm)

// --- phase_shift_overlay uniforms (only read when uRecipe == 2) -------------
uniform float uSwitchAxis;         // radians; axis we project view onto
uniform float uCarrierPeriodUm;    // carrier stripe period (μm)

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

  vec2 slitNormal = vec2(cos(uSlitOrientation), sin(uSlitOrientation));
  float proj = dot(viewTangent.xy, slitNormal);

  float t = smoothstep(-0.15, 0.15, proj);

  float sA = texture2D(uViewA, vUv - shift).r;
  float sB = texture2D(uViewB, vUv - shift).r;
  float scene = mix(sA, sB, t);

  float frontGold = texture2D(uFront, vUv).r;
  float transmission = (1.0 - frontGold) * scene;

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

// ----------------------------------------------------------------------------
// Recipe 2: phase_shift_overlay — both layers carry image content, but the
// back's stripe carrier is offset by half a period from the front's. Parallax
// shift through the substrate slides the back layer; the sign of the view
// projection biases which carrier phase the eye samples. Head-on, the two
// interlace; tilt one way and the front (hummingbird) phase dominates,
// tilt the other and the back (globe) phase dominates.
// ----------------------------------------------------------------------------
vec3 runPhaseShiftOverlay(vec3 viewTangent) {
  vec2 shift = parallax_offset(viewTangent, uThicknessUm, uN, uExtentUm);

  // Project view direction onto the switch axis. Positive proj biases the
  // back-layer reveal; negative proj biases the front.
  vec2 axis = vec2(cos(uSwitchAxis), sin(uSwitchAxis));
  float proj = dot(viewTangent.xy, axis);
  float bias = smoothstep(-0.20, 0.20, proj);  // 0 = all front, 1 = all back

  float frontGold = texture2D(uFront, vUv).r;
  float backGold  = texture2D(uBack,  vUv - shift).r;

  // Per-pixel reveal weights. The 0.4..0.6 floor keeps the off-tilt image
  // faintly visible so the head-on view shows the interlaced texture rather
  // than vanishing into a single half-density mask.
  float wFront = mix(1.0, 0.35, bias);
  float wBack  = mix(0.35, 1.0, bias);
  float reflected = wFront * frontGold + wBack * backGold;

  // Transmission only opens where neither layer covers the pixel — that's
  // the "gap between stripes" of both carriers, which only exists when the
  // two phases align perfectly via parallax.
  float transmission = (1.0 - frontGold) * (1.0 - backGold);

  vec3 color;
  if (uIllumination == 0) {
    vec3 goldShade = GOLD * reflected * (0.3 + 0.7 * max(0.0, vLightDirTangent.z));
    color = goldShade + vec3(0.05) * transmission;
  } else if (uIllumination == 1) {
    color = uLaserColor * transmission * (0.45 + 0.55 * max(0.0, vLightDirTangent.z));
    color += GOLD * 0.12 * reflected;
  } else {
    color = uBacklightColor * transmission;
    color += GOLD_BACK * reflected * 0.25;
  }
  return color;
}

void main() {
  vec3 viewTangent = normalize(vViewDirTangent);

  vec3 color;
  if (uRecipe == 0) {
    color = runStereoLenticular(viewTangent);
  } else if (uRecipe == 2) {
    color = runPhaseShiftOverlay(viewTangent);
  } else {
    // Default + uRecipe == 1: moire_interactive.
    color = runMoireInteractive(viewTangent);
  }

  gl_FragColor = vec4(color, 1.0);
}
