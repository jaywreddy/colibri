precision highp float;

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

const vec3 GOLD = vec3(0.902, 0.737, 0.314);
const vec3 GOLD_BACK = vec3(0.4, 0.32, 0.12);

// Sample the back layer with parallax offset driven by the view direction
// refracted into the substrate.
vec4 sampleBack(vec3 viewTangent) {
  // Snell's law: sin(theta_in) = n * sin(theta_sub)
  vec2 lateral = viewTangent.xy;
  float cosV = max(0.05, viewTangent.z);
  float sinV = length(lateral);
  float sinSub = sinV / uN;
  float cosSub = sqrt(max(0.0, 1.0 - sinSub * sinSub));
  vec2 dirSub = lateral;
  if (sinV > 1e-4) dirSub = lateral * (sinSub / sinV);
  // Parallax shift in μm
  vec2 shiftUm = dirSub * (uThicknessUm / max(0.05, cosSub));
  vec2 shiftUv = shiftUm / uExtentUm;
  return texture2D(uBack, vUv - shiftUv);
}

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
  // Map the view angle into atlas coords. Atlas is 3 wavelengths side-by-side.
  // Each sub-image covers angle range symmetric around 0.
  float ax = clamp(viewTangent.x, -0.7, 0.7);  // sin of tilt-x
  float ay = clamp(viewTangent.y, -0.7, 0.7);
  vec2 local = vec2((ax + 0.7) / 1.4, (ay + 0.7) / 1.4);
  float slabW = 1.0 / 3.0;
  float slab = float(uWavelengthSlot);
  vec2 atlasUv = vec2(slab * slabW + local.x * slabW, local.y);
  return texture2D(uFftAtlas, atlasUv).rgb;
}

void main() {
  vec4 f = texture2D(uFront, vUv);
  vec4 b = sampleBack(normalize(vViewDirTangent));

  float frontGold = f.r;   // 1 = gold
  float backGold = b.r;

  vec3 backlight = (uIllumination == 2) ? uBacklightColor : vec3(0.0);
  vec3 laserTint = (uIllumination == 1) ? uLaserColor : vec3(1.0);

  // Composite: front gold on top; where front is transparent, back gold; else substrate (backlight).
  vec3 color = mix(
    mix(backlight, GOLD_BACK * laserTint, backGold),
    GOLD * laserTint,
    frontGold
  );

  if (uIllumination == 0) {
    color = ambientLit(color);
    // Small front highlight along grazing angles for realism
    color += (1.0 - vViewDirTangent.z) * vec3(0.08, 0.07, 0.04) * frontGold;
  } else if (uIllumination == 1) {
    // Laser: modulate by incidence for visibility contrast
    color *= 0.4 + 0.6 * max(0.0, vLightDirTangent.z);
    // Add diffraction halo where gold is absent (i.e., aperture is open)
    if (uUseFft > 0.5) {
      float aperture = (1.0 - frontGold) * (1.0 - backGold);
      color += fftHaloColor(vViewDirTangent) * aperture * 0.9;
    }
  } else {
    // Backlight: transmitted light only where both layers are open
    float t = (1.0 - frontGold) * (1.0 - backGold);
    color = backlight * t + GOLD_BACK * 0.15 * backGold + GOLD * 0.25 * frontGold;
  }

  gl_FragColor = vec4(color, 1.0);
}
