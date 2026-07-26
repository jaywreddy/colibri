// ITEM 3 — emit a WORLD-SPACE surface basis and let the fragment shader build the
// view/light directions PER PIXEL.
//
// This file used to normalize the world view and light directions here, project them
// onto the tangent basis, and interpolate the results as vViewDirTangent /
// vLightDirTangent. Both plate planes are PlaneGeometry(w, h) at 1x1 segments — FOUR
// vertices for an entire 50 mm face — so that was a bilinear interpolation of a
// normalized direction across a quad subtending a large solid angle at the 45° FOV
// working distance. It is a poor approximation, worst at close zoom and grazing
// angles, and it made the diffractionSheen hue ramp, ndl, ndv and the specular term
// all wrong IN SHAPE across a face, in a way that changed with nothing but the
// tessellation.
//
// The basis vectors themselves are genuinely constant across a flat quad, so
// interpolating THEM is exact; only the view direction needs per-pixel evaluation,
// and for that the fragment shader needs the world position. Emitting the basis
// instead of the projected directions also removes the latent flat-plate assumption
// from the vertex stage (a future non-planar plate would just work) and precomputes
// exactly what the deferred env-map IBL item needs.
//
// Varying budget after the swap: vUv (vec2) + 4 vec3 = 5 vec4 slots, well inside the
// WebGL1 guaranteed minimum of 8.
varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vNormalWorld;
varying vec3 vTangentWorld;
varying vec3 vBitangentWorld;

void main() {
  vUv = uv;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;

  // Tangent-space basis. Tangent = model +x, bitangent = normal x tangent. These are
  // constant over a flat plate, so the interpolation is exact; the fragment shader
  // re-normalizes anyway so a future curved plate stays correct.
  vec3 normalWorld = normalize(mat3(modelMatrix) * normal);
  vNormalWorld = normalWorld;
  vTangentWorld = normalize(mat3(modelMatrix) * vec3(1.0, 0.0, 0.0));
  vBitangentWorld = normalize(cross(normalWorld, vTangentWorld));

  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
