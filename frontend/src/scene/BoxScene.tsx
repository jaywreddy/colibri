import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import vert from '../shaders/plate.vert';
import frag from '../shaders/plate.frag';
import { log } from '../logger';
import { useStore } from '../store';
import { RECIPE_IDS, type BoxManifest, type FaceId, FACE_IDS, type RenderRecipe } from '../api';

type FaceMaterial = {
  faceId: FaceId;
  material: THREE.ShaderMaterial;
  mesh: THREE.Mesh;
  textures: THREE.Texture[];  // tracked for disposal on rebind
};

/**
 * Six-face box viewer. Each face is its own PlaneMesh + ShaderMaterial bound
 * to that face's PlateManifest. We deliberately build six planes rather than
 * one BoxGeometry with materialIndex so we can:
 *   - Texture each face with its own front/back at the correct UVs without
 *     fighting BoxGeometry's per-face UV quirks.
 *   - Raycast individual faces for the click-to-edit selection.
 *
 * Flatten mode tweens the per-face rotation/position toward a 2×3 grid layout
 * so the user can review all faces head-on for fab inspection.
 */
export default function BoxScene({
  manifest,
  flat,
  onFaceClick,
}: {
  manifest: BoxManifest | null;
  flat: boolean;
  onFaceClick: (faceId: FaceId) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const threeRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    faces: Record<FaceId, FaceMaterial>;
    boxGroup: THREE.Group;
    flat: boolean;
  } | null>(null);
  const onFaceClickRef = useRef(onFaceClick);
  onFaceClickRef.current = onFaceClick;

  const illumination = useStore((s) => s.illumination);
  const laserColor = useStore((s) => s.laserColor);
  const lightAz = useStore((s) => s.lightAzimuthDeg);
  const lightEl = useStore((s) => s.lightElevationDeg);

  // --- one-time setup -------------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.position.set(2.2, 1.6, 2.4);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.8;
    controls.maxDistance = 8.0;
    controls.target.set(0, 0, 0);

    const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    blank.needsUpdate = true;

    const boxGroup = new THREE.Group();
    scene.add(boxGroup);

    const faces = {} as Record<FaceId, FaceMaterial>;
    for (const faceId of FACE_IDS) {
      const geom = new THREE.PlaneGeometry(1, 1, 1, 1);
      const material = new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        side: THREE.DoubleSide,
        uniforms: {
          uFront: { value: blank },
          uBack: { value: blank },
          uExtentUm: { value: 24000.0 },
          uThicknessUm: { value: 500.0 },
          uN: { value: 1.46 },
          uIllumination: { value: 0 },
          uLaserColor: { value: new THREE.Color(0x33ff88) },
          uBacklightColor: { value: new THREE.Color(0xffffff) },
          uAmbientColor: { value: new THREE.Color(0xffffff) },
          uLightWorld: { value: new THREE.Vector3(2, 3, 3) },
          uRecipe: { value: RECIPE_IDS.moire_interactive },
          uViewA: { value: blank },
          uViewB: { value: blank },
          uSlitOrientation: { value: 0.0 },
          uSlitPeriodUm: { value: 40.0 },
          uSwitchAxis: { value: 0.0 },
          uCarrierPeriodUm: { value: 20.0 },
        },
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.userData.faceId = faceId;
      boxGroup.add(mesh);
      faces[faceId] = { faceId, material, mesh, textures: [] };
    }
    layoutBox(faces, false, 1, 1, 1);

    threeRef.current = { scene, camera, renderer, controls, faces, boxGroup, flat: false };
    // Expose for E2E + debug. The plate-mode scene already takes window.__three;
    // box-mode keeps its own handle so both modes can coexist in dev tools.
    (window as unknown as { __box: typeof threeRef.current }).__box = threeRef.current;

    const canvas = renderer.domElement;
    const onContextLost = (e: Event) => {
      e.preventDefault();
      log('webgl_context_lost');
    };
    const onContextRestored = () => log('webgl_context_restored');
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);

    // --- raycast click → face select ---
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let dragMoved = false;
    let dragX = 0;
    let dragY = 0;
    const onPointerDown = (e: PointerEvent) => {
      dragMoved = false;
      dragX = e.clientX;
      dragY = e.clientY;
    };
    const onPointerMove = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - dragX, e.clientY - dragY) > 5) dragMoved = true;
    };
    const onPointerUp = (e: PointerEvent) => {
      if (dragMoved) return;
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(
        FACE_IDS.map((fid) => faces[fid].mesh)
      );
      if (hits.length > 0) {
        const fid = hits[0].object.userData.faceId as FaceId;
        log('face_clicked', { faceId: fid });
        onFaceClickRef.current(fid);
      }
    };
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);

    const onResize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    onResize();
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    let rafId = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      for (const fid of FACE_IDS) {
        for (const t of faces[fid].textures) t.dispose();
      }
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // --- bind manifest -> textures + recipe per face --------------------------
  useEffect(() => {
    const t = threeRef.current;
    if (!t || !manifest) return;

    // Box dimensions are in μm; project to scene units so the largest side
    // is ~1.4 (fits comfortably in the camera's default zoom).
    const dims = manifest.dimensions_um;
    const maxDim = Math.max(dims.width, dims.height, dims.depth);
    const sx = (dims.width / maxDim) * 1.4;
    const sy = (dims.height / maxDim) * 1.4;
    const sz = (dims.depth / maxDim) * 1.4;
    layoutBox(t.faces, t.flat, sx, sy, sz);

    const loader = new THREE.TextureLoader();
    for (const fid of FACE_IDS) {
      const fm = manifest.faces[fid];
      const face = t.faces[fid];
      if (!fm) continue;
      Promise.all([
        loader.loadAsync(fm.files.front_png),
        loader.loadAsync(fm.files.back_png),
      ]).then(([front, back]) => {
        for (const tex of [front, back]) {
          tex.colorSpace = THREE.LinearSRGBColorSpace;
          tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
          tex.magFilter = THREE.LinearFilter;
          tex.minFilter = THREE.LinearFilter;
          tex.generateMipmaps = false;
          tex.anisotropy = 8;
          tex.needsUpdate = true;
        }
        // Dispose old textures before swapping
        for (const old of face.textures) old.dispose();
        face.textures = [front, back];
        face.material.uniforms.uFront.value = front;
        face.material.uniforms.uBack.value = back;
        face.material.uniforms.uExtentUm.value = fm.extent_um[0];
        face.material.uniforms.uThicknessUm.value = fm.substrate.thickness_um;
        face.material.uniforms.uN.value = fm.substrate.n;
        const rawRecipe = fm.render_recipe;
        const recipe: RenderRecipe =
          rawRecipe && rawRecipe in RECIPE_IDS
            ? (rawRecipe as RenderRecipe)
            : 'moire_interactive';
        face.material.uniforms.uRecipe.value = RECIPE_IDS[recipe];
        log('face_texture_bound', {
          face: fid,
          slug: fm.spec.pattern_slug,
          recipe,
        });
      });
    }
  }, [manifest]);

  // --- toggle flatten -------------------------------------------------------
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    t.flat = flat;
    const dims = manifest?.dimensions_um;
    if (dims) {
      const maxDim = Math.max(dims.width, dims.height, dims.depth);
      const sx = (dims.width / maxDim) * 1.4;
      const sy = (dims.height / maxDim) * 1.4;
      const sz = (dims.depth / maxDim) * 1.4;
      layoutBox(t.faces, flat, sx, sy, sz);
    } else {
      layoutBox(t.faces, flat, 1, 1, 1);
    }
  }, [flat, manifest]);

  // --- illumination / light position ---------------------------------------
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    const illumId = illumination === 'ambient' ? 0 : illumination === 'laser' ? 1 : 2;
    const laserRgb: Record<string, number> = {
      red: 0xff3355,
      green: 0x44ff88,
      blue: 0x3388ff,
    };
    for (const fid of FACE_IDS) {
      t.faces[fid].material.uniforms.uIllumination.value = illumId;
      t.faces[fid].material.uniforms.uLaserColor.value.setHex(laserRgb[laserColor]);
    }
  }, [illumination, laserColor]);

  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    const az = (lightAz * Math.PI) / 180;
    const el = (lightEl * Math.PI) / 180;
    const r = 3.0;
    const x = r * Math.cos(el) * Math.sin(az);
    const y = r * Math.sin(el);
    const z = r * Math.cos(el) * Math.cos(az);
    for (const fid of FACE_IDS) {
      t.faces[fid].material.uniforms.uLightWorld.value.set(x, y, z);
    }
  }, [lightAz, lightEl]);

  return (
    <div
      ref={mountRef}
      style={{ width: '100%', height: '100%', minHeight: 0, minWidth: 0 }}
    />
  );
}

/**
 * Position + rotate the 6 face meshes to form either an assembled box or a
 * 2×3 flatten grid. The face dimensions ``sx, sy, sz`` are scene units —
 * front/back are sx×sy, top/bottom are sx×sz, left/right are sz×sy.
 */
function layoutBox(
  faces: Record<FaceId, FaceMaterial>,
  flat: boolean,
  sx: number,
  sy: number,
  sz: number
): void {
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;

  if (flat) {
    // 2×3 grid — keep each face's native aspect ratio, lay them flat in XY.
    const cellW = Math.max(sx, sz) + 0.15;
    const cellH = Math.max(sy, sz) + 0.15;
    const layout: [FaceId, number, number, number, number][] = [
      // [faceId, col(-1..1), row(-1..0..+1), width, height]
      ['top', -1, 1, sx, sz],
      ['front', 0, 1, sx, sy],
      ['bottom', 1, 1, sx, sz],
      ['left', -1, 0, sz, sy],
      ['back', 0, 0, sx, sy],
      ['right', 1, 0, sz, sy],
    ];
    for (const [fid, col, row, w, h] of layout) {
      const face = faces[fid];
      face.mesh.scale.set(w, h, 1);
      face.mesh.rotation.set(0, 0, 0);
      face.mesh.position.set(col * cellW, (row - 0.5) * cellH, 0);
    }
    return;
  }

  const place = (
    fid: FaceId,
    w: number,
    h: number,
    pos: [number, number, number],
    rot: [number, number, number]
  ) => {
    const face = faces[fid];
    face.mesh.scale.set(w, h, 1);
    face.mesh.position.set(...pos);
    face.mesh.rotation.set(...rot);
  };

  place('front', sx, sy, [0, 0, hz], [0, 0, 0]);
  place('back', sx, sy, [0, 0, -hz], [0, Math.PI, 0]);
  place('right', sz, sy, [hx, 0, 0], [0, Math.PI / 2, 0]);
  place('left', sz, sy, [-hx, 0, 0], [0, -Math.PI / 2, 0]);
  place('top', sx, sz, [0, hy, 0], [-Math.PI / 2, 0, 0]);
  place('bottom', sx, sz, [0, -hy, 0], [Math.PI / 2, 0, 0]);
}
