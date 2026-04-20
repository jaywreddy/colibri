// Snell-refracted lateral shift for sampling a layer sitting below a
// substrate of `thickness_um` and refractive index `n`, viewed in
// tangent-space along `viewTangent` (unit vector, +z = out of plate).
//
// Returns the UV offset to *add* to the front-face UV to sample the
// corresponding point on the back face. Multiple recipes need this
// (stereo_lenticular, moire_interactive, stylized_amplitude), so it lives
// here to keep the math in one place — notably, the stylized path used to
// compute this offset and then throw it away, which is why parallax never
// worked.
vec2 parallax_offset(vec3 viewTangent, float thicknessUm, float n, float extentUm) {
  // lateral shift through a slab: t · tan(θ_refracted)
  // with Snell's law sin(θ_refracted) = sin(θ_view) / n.
  vec2 lateral = viewTangent.xy;
  float sinV = length(lateral);
  if (sinV < 1e-4) return vec2(0.0);
  float sinSub = sinV / n;
  float cosSub = sqrt(max(0.0, 1.0 - sinSub * sinSub));
  vec2 dirSub = lateral / sinV;              // unit vector along the lateral axis
  vec2 shiftUm = dirSub * (thicknessUm * sinSub / max(0.05, cosSub));
  return shiftUm / extentUm;
}
