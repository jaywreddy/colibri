// Snell-refracted lateral shift for sampling a layer sitting below a
// substrate of `thickness_um` and refractive index `n`, viewed in
// tangent-space along `viewTangent` (unit vector, +z = out of plate).
//
// Returns the UV offset to *add* to the front-face UV to sample the
// corresponding point on the back face. Every recipe in the catalog uses
// this — moire_interactive, stereo_lenticular, and phase_shift_overlay all
// reduce to "sample two layers separated by a refracting slab", just with
// different downstream logic.
//
// `extentUm` is the plate's physical (width, height): UV u spans the width
// and UV v the height, so the um shift converts to UV per-axis. Ring-box
// wall plates are non-square — a scalar extent would distort the v-axis
// fringe motion by width/height.
vec2 parallax_offset(vec3 viewTangent, float thicknessUm, float n, vec2 extentUm) {
  // lateral shift through a slab: t · tan(θ_refracted)
  // with Snell's law sin(θ_refracted) = sin(θ_view) / n.
  vec2 lateral = viewTangent.xy;
  float sinV = length(lateral);
  if (sinV < 1e-4) return vec2(0.0);
  float sinSub = sinV / n;
  float cosSub = sqrt(max(0.0, 1.0 - sinSub * sinSub));
  vec2 dirSub = lateral / sinV;              // unit vector along the lateral axis
  vec2 shiftUm = dirSub * (thicknessUm * sinSub / max(0.05, cosSub));
  return shiftUm / extentUm;                 // componentwise um -> UV
}
