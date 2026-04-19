varying vec2 vUv;
varying vec3 vViewDirTangent;
varying vec3 vLightDirTangent;
varying vec3 vNormalWorld;

uniform vec3 uLightWorld;

void main() {
  vUv = uv;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vec3 normalWorld = normalize(mat3(modelMatrix) * normal);
  vNormalWorld = normalWorld;

  // Build tangent-space basis (assume plate is flat, tangent = +x, bitangent = +y in model space)
  vec3 tangentWorld = normalize(mat3(modelMatrix) * vec3(1.0, 0.0, 0.0));
  vec3 bitangentWorld = normalize(cross(normalWorld, tangentWorld));

  vec3 viewDirWorld = normalize(cameraPosition - worldPos.xyz);
  vec3 lightDirWorld = normalize(uLightWorld - worldPos.xyz);

  vViewDirTangent = vec3(
    dot(viewDirWorld, tangentWorld),
    dot(viewDirWorld, bitangentWorld),
    dot(viewDirWorld, normalWorld)
  );
  vLightDirTangent = vec3(
    dot(lightDirWorld, tangentWorld),
    dot(lightDirWorld, bitangentWorld),
    dot(lightDirWorld, normalWorld)
  );

  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
