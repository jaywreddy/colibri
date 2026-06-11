import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import vert from '../shaders/plate.vert';
import frag from '../shaders/plate.frag';
import { log } from '../logger';
import { useStore } from '../store';
import { FACE_IDS, RECIPE_IDS, type FaceId, type RenderRecipe } from '../api';
import {
  FOIL_COLORS,
  cutList,
  hingeLayout,
  overlapUm,
  platePlacements,
  seamSegments,
  type CutPlate,
} from '../assembly';

// Scene is built in millimeters inside a root group scaled so the largest
// box dimension is ~1.4 scene units.
const TARGET_SCENE_SIZE = 1.4;
/** Pattern plane offset outside the slab's outer surface (mm). */
const EPS_PATTERN_MM = 0.03;
/** Foil strips sit just above the pattern plane (mm). */
const EPS_FOIL_MM = 0.09;
/** Visual thickness of tinned (no-bead) metallic edges (mm). */
const TIN_MM = 0.3;
const BRASS_COLOR = 0xb08d57;

type FaceRT = {
  faceId: FaceId;
  /** The dual-layer gold-pattern shader — persistent across rebuilds. */
  shader: THREE.ShaderMaterial;
  /** The fused-silica slab material — persistent across rebuilds. */
  glassMat: THREE.MeshPhysicalMaterial;
  /** Front/back PNG textures currently bound (tracked for disposal). */
  textures: THREE.Texture[];
};

type RebuildDisposables = {
  geoms: THREE.BufferGeometry[];
  mats: THREE.Material[];
  texs: THREE.Texture[];
};

type CamTween = {
  fromAz: number;
  toAz: number;
  t0: number;
  durMs: number;
};

type Ctx = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  root: THREE.Group;
  buildGroup: THREE.Group | null;
  lidPivot: THREE.Group | null;
  faces: Record<FaceId, FaceRT>;
  raycastTargets: THREE.Mesh[];
  rebuildDisposables: RebuildDisposables;
  keyLight: THREE.DirectionalLight;
  pmrem: THREE.PMREMGenerator;
  envTex: THREE.Texture;
  /** Background gradient + soft ground-shadow textures (created once). */
  bgTex: THREE.CanvasTexture;
  shadowTex: THREE.CanvasTexture;
  lidCurrentDeg: number;
  lidTargetDeg: number;
  bindToken: number;
  /** Gentle camera-azimuth tween toward the selected face (null = idle). */
  camTween: CamTween | null;
  /** True while the user is orbit-dragging — tweens must not fight it. */
  userDragging: boolean;
};

export type StudioHandle = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  root: THREE.Group;
  lidPivot: THREE.Group | null;
  faces: Record<FaceId, FaceRT>;
  store: typeof useStore;
  setLid: (deg: number) => void;
  getLidDeg: () => number;
  /** Mirrors the store's autoRotate flag (and OrbitControls.autoRotate). */
  autoRotate: boolean;
};

function makeBlankTexture(): THREE.DataTexture {
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  blank.needsUpdate = true;
  return blank;
}

/** Subtle vertical studio backdrop: very dark blue fading to near-black. */
function makeBackgroundTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 512;
  const c2d = canvas.getContext('2d')!;
  const grad = c2d.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#161d30');
  grad.addColorStop(0.55, '#0b0e16');
  grad.addColorStop(1, '#06070b');
  c2d.fillStyle = grad;
  c2d.fillRect(0, 0, 2, 512);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Radial soft-shadow blob laid flat under the box (no shadow mapping). */
function makeShadowTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const c2d = canvas.getContext('2d')!;
  const grad = c2d.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2
  );
  grad.addColorStop(0, 'rgba(0, 0, 0, 0.5)');
  grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.26)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  c2d.fillStyle = grad;
  c2d.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

/** Azimuth (OrbitControls theta) that faces each wall head-on. */
const FACE_AZIMUTH: Partial<Record<FaceId, number>> = {
  front: 0,
  right: Math.PI / 2,
  back: Math.PI,
  left: -Math.PI / 2,
};

const easeInOutQuad = (k: number): number =>
  k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;

function makePlateShader(blank: THREE.Texture): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: vert,
    fragmentShader: frag,
    side: THREE.DoubleSide,
    uniforms: {
      uFront: { value: blank },
      uBack: { value: blank },
      uExtentUm: { value: new THREE.Vector2(50000.0, 50000.0) },
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
}

function makeGlassMaterial(): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    transmission: 0.85,
    ior: 1.46,
    roughness: 0.06,
    metalness: 0.0,
    thickness: 0.02,
    side: THREE.DoubleSide,
    envMapIntensity: 1.0,
  });
}

/** Four flat foil strips framing a w x h plate at offset z (plate-local). */
function addFoilFrame(
  parent: THREE.Group,
  w: number,
  h: number,
  ov: number,
  z: number,
  foilMat: THREE.Material,
  geo: <T extends THREE.BufferGeometry>(g: T) => T
): void {
  const strips: [number, number, number, number][] = [
    // [cx, cy, strip_w, strip_h]
    [0, h / 2 - ov / 2, w, ov],
    [0, -(h / 2 - ov / 2), w, ov],
    [-(w / 2 - ov / 2), 0, ov, h - 2 * ov],
    [w / 2 - ov / 2, 0, ov, h - 2 * ov],
  ];
  for (const [cx, cy, sw, sh] of strips) {
    if (sw <= 0 || sh <= 0) continue;
    const m = new THREE.Mesh(geo(new THREE.PlaneGeometry(sw, sh)), foilMat);
    m.position.set(cx, cy, z);
    parent.add(m);
  }
}

function makeLabelSprite(text: string, D: RebuildDisposables): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const c2d = canvas.getContext('2d');
  if (c2d) {
    c2d.clearRect(0, 0, 256, 64);
    c2d.font = '600 36px system-ui, sans-serif';
    c2d.fillStyle = '#e8eaed';
    c2d.textAlign = 'center';
    c2d.textBaseline = 'middle';
    c2d.fillText(text.toUpperCase(), 128, 32);
  }
  const tex = new THREE.CanvasTexture(canvas);
  D.texs.push(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  D.mats.push(mat);
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(14, 3.5, 1);
  return sprite;
}

/**
 * Ring Box Studio 3D preview — the real object:
 *   - 6 fused-silica plates (MeshPhysicalMaterial slabs) each carrying its
 *     gold-on-quartz pattern surface (the existing plate.vert/plate.frag
 *     ShaderMaterial, textures bound from the box manifest).
 *   - Copper-foil overlap strips on outer + inner plate borders.
 *   - 8 solder seam beads (4 bottom + 4 vertical corners), tinned rims.
 *   - Brass tube-and-rod hinge along the back top edge; the lid + its foil
 *     + its tube segments live under a pivot group at the hinge axis and
 *     open by rotating about +X by -angle (front edge swings UP and BACK).
 *   - 'flat' layout: labeled 2x3 fab-inspection grid (seams/hinge hidden).
 *
 * All static geometry derives from src/assembly.ts so it updates instantly
 * on spec changes; manifest textures rebind only when the manifest changes.
 */
export default function BoxScene() {
  const mountRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Ctx | null>(null);

  const boxSpec = useStore((s) => s.boxSpec);
  const boxManifest = useStore((s) => s.boxManifest);
  const layout = useStore((s) => s.layout);
  const lidTargetDeg = useStore((s) => s.lidTargetDeg);
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const autoRotate = useStore((s) => s.autoRotate);
  const illumination = useStore((s) => s.illumination);
  const laserColor = useStore((s) => s.laserColor);
  const lightAz = useStore((s) => s.lightAzimuthDeg);
  const lightEl = useStore((s) => s.lightElevationDeg);

  // -- full static-geometry rebuild (cheap; pure assembly.ts math) -----------
  function rebuild(): void {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const spec = useStore.getState().boxSpec;
    const sceneLayout = useStore.getState().layout;

    if (ctx.buildGroup) {
      ctx.root.remove(ctx.buildGroup);
      for (const g of ctx.rebuildDisposables.geoms) g.dispose();
      for (const m of ctx.rebuildDisposables.mats) m.dispose();
      for (const t of ctx.rebuildDisposables.texs) t.dispose();
    }
    const D: RebuildDisposables = { geoms: [], mats: [], texs: [] };
    ctx.rebuildDisposables = D;
    ctx.raycastTargets = [];
    ctx.lidPivot = null;

    const geo = <T extends THREE.BufferGeometry>(g: T): T => {
      D.geoms.push(g);
      return g;
    };
    const mat = <T extends THREE.Material>(m: T): T => {
      D.mats.push(m);
      return m;
    };

    const mm = (um: number): number => um / 1000;
    const W = mm(spec.width_um);
    const Dep = mm(spec.depth_um);
    const H = mm(spec.height_um);
    const T = mm(spec.glass.thickness_um);
    // 2x3 fab grid cell size (also drives the flat-layout scale).
    const cellW = Math.max(W, Dep) + 8;
    const cellH = Math.max(H, Dep) + 12;
    const scale =
      sceneLayout === 'flat'
        ? 2.2 / Math.max(1e-6, 3 * cellW, 2 * cellH)
        : TARGET_SCENE_SIZE / Math.max(1e-6, W, Dep, H);
    ctx.root.scale.setScalar(scale);

    for (const fid of FACE_IDS) {
      const g = ctx.faces[fid].glassMat;
      g.ior = spec.glass.n;
      g.thickness = T * scale;
    }

    const finish = new THREE.Color(FOIL_COLORS[spec.foil.finish]);
    const foilMat = mat(
      new THREE.MeshStandardMaterial({
        color: finish,
        metalness: 1.0,
        roughness: 0.32,
        side: THREE.DoubleSide,
      })
    );
    const solderMat = mat(
      new THREE.MeshStandardMaterial({ color: finish, metalness: 1.0, roughness: 0.26 })
    );
    const tinMat = mat(
      new THREE.MeshStandardMaterial({ color: finish, metalness: 1.0, roughness: 0.3 })
    );
    const brassMat = mat(
      new THREE.MeshStandardMaterial({ color: BRASS_COLOR, metalness: 1.0, roughness: 0.34 })
    );

    const group = new THREE.Group();
    ctx.buildGroup = group;
    ctx.root.add(group);

    const cuts = {} as Record<FaceId, CutPlate>;
    for (const c of cutList(spec)) cuts[c.face] = c;
    const overlapMm = mm(overlapUm(spec));

    const buildPlate = (fid: FaceId): THREE.Group => {
      const cut = cuts[fid];
      const w = mm(cut.width_um);
      const h = mm(cut.height_um);
      const rt = ctx.faces[fid];
      const pg = new THREE.Group();
      // (a) the fused-silica slab
      const slab = new THREE.Mesh(geo(new THREE.BoxGeometry(w, h, T)), rt.glassMat);
      slab.userData.faceId = fid;
      pg.add(slab);
      ctx.raycastTargets.push(slab);
      // (b) the gold pattern surface, epsilon outside the outer face
      const plane = new THREE.Mesh(geo(new THREE.PlaneGeometry(w, h)), rt.shader);
      plane.position.z = T / 2 + EPS_PATTERN_MM;
      plane.userData.faceId = fid;
      pg.add(plane);
      ctx.raycastTargets.push(plane);
      // (c) copper foil overlap strips, outer AND inner borders
      const ov = Math.min(overlapMm, Math.min(w, h) / 2);
      if (ov > 1e-4) {
        addFoilFrame(pg, w, h, ov, T / 2 + EPS_FOIL_MM, foilMat, geo);
        addFoilFrame(pg, w, h, ov, -(T / 2 + EPS_FOIL_MM), foilMat, geo);
      }
      return pg;
    };

    if (sceneLayout === 'assembled') {
      // Soft ground shadow — a radial-gradient blob just under the box.
      const shadowSize = 1.9 * Math.max(W, Dep);
      const shadow = new THREE.Mesh(
        geo(new THREE.PlaneGeometry(shadowSize, shadowSize)),
        mat(
          new THREE.MeshBasicMaterial({
            map: ctx.shadowTex,
            transparent: true,
            depthWrite: false,
          })
        )
      );
      shadow.rotation.x = -Math.PI / 2;
      shadow.position.y = -H / 2 - 0.8;
      shadow.renderOrder = -1;
      group.add(shadow);
    }

    if (sceneLayout === 'flat') {
      // Labeled 2x3 fab-inspection grid — seams/hinge hidden.
      FACE_IDS.forEach((fid, i) => {
        const col = (i % 3) - 1;
        const row = i < 3 ? 0.5 : -0.5;
        const pg = buildPlate(fid);
        pg.position.set(col * cellW, row * cellH, 0);
        group.add(pg);
        const label = makeLabelSprite(fid, D);
        label.position.set(col * cellW, row * cellH - mm(cuts[fid].height_um) / 2 - 3, 0.5);
        group.add(label);
      });
    } else {
      const hinge = hingeLayout(spec);
      const pivotPos = new THREE.Vector3(0, mm(hinge.axis_y_um), mm(hinge.axis_z_um));
      const lidPivot = new THREE.Group();
      lidPivot.position.copy(pivotPos);
      lidPivot.rotation.x = -THREE.MathUtils.degToRad(ctx.lidCurrentDeg);
      group.add(lidPivot);
      ctx.lidPivot = lidPivot;

      // --- six plates; lid (top) is parented to the hinge pivot -------------
      for (const p of platePlacements(spec)) {
        const pg = buildPlate(p.face);
        pg.rotation.set(p.rotation[0], p.rotation[1], p.rotation[2]);
        const c = new THREE.Vector3(mm(p.center_um[0]), mm(p.center_um[1]), mm(p.center_um[2]));
        if (p.face === 'top') {
          pg.position.copy(c.sub(pivotPos));
          lidPivot.add(pg);
        } else {
          pg.position.copy(c);
          group.add(pg);
        }
      }

      // --- 8 solder seam beads (half-embedded cylinders) --------------------
      const beadR = mm(spec.foil.bead_um) / 2;
      for (const seam of seamSegments(spec)) {
        const a = new THREE.Vector3(mm(seam.start_um[0]), mm(seam.start_um[1]), mm(seam.start_um[2]));
        const b = new THREE.Vector3(mm(seam.end_um[0]), mm(seam.end_um[1]), mm(seam.end_um[2]));
        const len = a.distanceTo(b);
        const bead = new THREE.Mesh(geo(new THREE.CylinderGeometry(beadR, beadR, len, 16)), solderMat);
        bead.position.copy(a).add(b).multiplyScalar(0.5);
        if (seam.axis === 'x') bead.rotation.z = Math.PI / 2;
        else if (seam.axis === 'z') bead.rotation.x = Math.PI / 2;
        group.add(bead);
      }

      // --- tinned (no bead): wall top rim + lid edge faces -------------------
      const hw = W / 2;
      const hd = Dep / 2;
      const hh = H / 2;
      const rimY = hh - T;
      const wallRims: [number, number, number, number, number, number][] = [
        [0, rimY, hd - T / 2, W, TIN_MM, T],
        [0, rimY, -hd + T / 2, W, TIN_MM, T],
        [-hw + T / 2, rimY, 0, T, TIN_MM, Dep - 2 * T],
        [hw - T / 2, rimY, 0, T, TIN_MM, Dep - 2 * T],
      ];
      for (const [cx, cy, cz, sx, sy, sz] of wallRims) {
        const rim = new THREE.Mesh(geo(new THREE.BoxGeometry(sx, sy, sz)), tinMat);
        rim.position.set(cx, cy, cz);
        group.add(rim);
      }
      const lidY = hh - T / 2;
      const lidEdges: [number, number, number, number, number, number][] = [
        [0, lidY, hd, W, T, TIN_MM],
        [0, lidY, -hd, W, T, TIN_MM],
        [-hw, lidY, 0, TIN_MM, T, Dep],
        [hw, lidY, 0, TIN_MM, T, Dep],
      ];
      for (const [cx, cy, cz, sx, sy, sz] of lidEdges) {
        const edge = new THREE.Mesh(geo(new THREE.BoxGeometry(sx, sy, sz)), tinMat);
        edge.position.set(cx - pivotPos.x, cy - pivotPos.y, cz - pivotPos.z);
        lidPivot.add(edge);
      }

      // --- hinge: alternating tube segments + the rod ------------------------
      const tubeR = mm(hinge.tube_r_um);
      const rodR = mm(hinge.rod_r_um);
      for (const seg of hinge.segments) {
        const tube = new THREE.Mesh(
          geo(new THREE.CylinderGeometry(tubeR, tubeR, mm(seg.length_um), 20)),
          brassMat
        );
        tube.rotation.z = Math.PI / 2;
        if (seg.owner === 'lid') {
          tube.position.set(mm(seg.center_x_um), 0, 0);
          lidPivot.add(tube);
        } else {
          tube.position.set(mm(seg.center_x_um), pivotPos.y, pivotPos.z);
          group.add(tube);
        }
      }
      const rod = new THREE.Mesh(
        geo(new THREE.CylinderGeometry(rodR, rodR, mm(hinge.rod_length_um), 16)),
        brassMat
      );
      rod.rotation.z = Math.PI / 2;
      rod.position.set(0, pivotPos.y, pivotPos.z);
      group.add(rod);
    }

    const w = window as unknown as { __studio?: StudioHandle };
    if (w.__studio) w.__studio.lidPivot = ctx.lidPivot;
    log('box_scene_rebuilt', { layout: sceneLayout });
  }

  // --- one-time setup --------------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    // Subtle dark blue-to-black gradient backdrop (replaces flat 0x0b0d10).
    const bgTex = makeBackgroundTexture();
    scene.background = bgTex;
    const shadowTex = makeShadowTexture();

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.position.set(1.8, 1.3, 1.9);

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

    // Environment map — metals (foil, solder, brass) need one to read as
    // metal. Keep the dark background; the env only feeds reflections.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(2, 3, 3);
    scene.add(keyLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.8;
    controls.maxDistance = 8.0;
    controls.target.set(0, 0, 0);
    controls.autoRotate = useStore.getState().autoRotate;
    controls.autoRotateSpeed = 0.8; // slow turntable
    // Cancel any camera tween the moment the user starts orbiting.
    controls.addEventListener('start', () => {
      const c = ctxRef.current;
      if (!c) return;
      c.userDragging = true;
      c.camTween = null;
    });
    controls.addEventListener('end', () => {
      const c = ctxRef.current;
      if (c) c.userDragging = false;
    });

    const root = new THREE.Group();
    scene.add(root);

    const blank = makeBlankTexture();
    const faces = {} as Record<FaceId, FaceRT>;
    for (const fid of FACE_IDS) {
      faces[fid] = {
        faceId: fid,
        shader: makePlateShader(blank),
        glassMat: makeGlassMaterial(),
        textures: [],
      };
    }

    const ctx: Ctx = {
      scene,
      camera,
      renderer,
      controls,
      root,
      buildGroup: null,
      lidPivot: null,
      faces,
      raycastTargets: [],
      rebuildDisposables: { geoms: [], mats: [], texs: [] },
      keyLight,
      pmrem,
      envTex,
      bgTex,
      shadowTex,
      lidCurrentDeg: 0,
      lidTargetDeg: useStore.getState().lidTargetDeg,
      bindToken: 0,
      camTween: null,
      userDragging: false,
    };
    ctxRef.current = ctx;

    // E2E/debug surface — replaces the old window.__box / window.__three.
    const studio: StudioHandle = {
      scene,
      camera,
      renderer,
      controls,
      root,
      lidPivot: null,
      faces,
      store: useStore,
      setLid: (deg: number) => useStore.getState().setLidTargetDeg(deg),
      getLidDeg: () => ctxRef.current?.lidCurrentDeg ?? 0,
      autoRotate: useStore.getState().autoRotate,
    };
    (window as unknown as { __studio: StudioHandle }).__studio = studio;

    const canvas = renderer.domElement;
    const onContextLost = (e: Event) => {
      e.preventDefault();
      log('webgl_context_lost');
    };
    const onContextRestored = () => log('webgl_context_restored');
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);

    // --- raycast click -> face select ---
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
      // Cursor affordance: pointer over a selectable face, grab otherwise.
      const c = ctxRef.current;
      if (!c) return;
      if (c.userDragging) {
        canvas.style.cursor = 'grabbing';
        return;
      }
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(c.raycastTargets, false);
      canvas.style.cursor = hits.length > 0 ? 'pointer' : 'grab';
    };
    const onPointerUp = (e: PointerEvent) => {
      if (dragMoved) return;
      const c = ctxRef.current;
      if (!c) return;
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(c.raycastTargets, false);
      if (hits.length > 0) {
        const fid = hits[0].object.userData.faceId as FaceId;
        log('face_clicked', { faceId: fid });
        useStore.getState().setSelectedFace(fid);
      }
    };
    // Double-click on the box toggles the lid open/closed.
    const onDoubleClick = (e: MouseEvent) => {
      const c = ctxRef.current;
      if (!c) return;
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(c.raycastTargets, false);
      if (hits.length === 0) return;
      const st = useStore.getState();
      const next = st.lidTargetDeg > 0 ? 0 : 110;
      log('lid_changed', { deg: next, dblclick: true });
      st.setLidTargetDeg(next);
    };
    canvas.style.cursor = 'grab';
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('dblclick', onDoubleClick);

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
    let lastT = performance.now();
    const tick = (now: number) => {
      const c = ctxRef.current;
      rafId = requestAnimationFrame(tick);
      if (!c) return;
      const dt = Math.min(0.05, (now - lastT) / 1000);
      lastT = now;
      // Damped lid animation toward the target angle.
      const target = c.lidTargetDeg;
      let cur = c.lidCurrentDeg;
      cur += (target - cur) * Math.min(1, dt * 8);
      if (Math.abs(cur - target) < 0.01) cur = target;
      c.lidCurrentDeg = cur;
      if (c.lidPivot) {
        // VERIFIED sign: pivot sits behind the back face; rotating about +X
        // by -angle lifts the lid's front edge (z_rel > 0) UP and swings it
        // BACK over the hinge — never through the box.
        c.lidPivot.rotation.x = -THREE.MathUtils.degToRad(cur);
      }
      // Gentle camera-azimuth tween toward the selected face. Never runs
      // while the user is dragging (controls 'start' clears the tween).
      if (c.camTween && !c.userDragging) {
        const tw = c.camTween;
        const k = Math.min(1, (now - tw.t0) / tw.durMs);
        const az = tw.fromAz + (tw.toAz - tw.fromAz) * easeInOutQuad(k);
        const offset = camera.position.clone().sub(controls.target);
        const sph = new THREE.Spherical().setFromVector3(offset);
        sph.theta = az;
        offset.setFromSpherical(sph);
        camera.position.copy(controls.target).add(offset);
        if (k >= 1) c.camTween = null;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('dblclick', onDoubleClick);
      const c = ctxRef.current;
      if (c) {
        for (const g of c.rebuildDisposables.geoms) g.dispose();
        for (const m of c.rebuildDisposables.mats) m.dispose();
        for (const t of c.rebuildDisposables.texs) t.dispose();
        for (const fid of FACE_IDS) {
          const rt = c.faces[fid];
          for (const t of rt.textures) t.dispose();
          rt.shader.dispose();
          rt.glassMat.dispose();
        }
        c.envTex.dispose();
        c.pmrem.dispose();
        c.bgTex.dispose();
        c.shadowTex.dispose();
      }
      blank.dispose();
      ctxRef.current = null;
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- rebuild static geometry when spec geometry / layout changes ----------
  const geomKey = useMemo(
    () =>
      JSON.stringify({
        w: boxSpec.width_um,
        d: boxSpec.depth_um,
        h: boxSpec.height_um,
        glass: boxSpec.glass,
        foil: boxSpec.foil,
        hinge: boxSpec.hinge,
        layout,
      }),
    [boxSpec, layout]
  );
  useEffect(() => {
    rebuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geomKey]);

  // --- bind manifest -> textures + recipe uniforms per face -----------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !boxManifest) return;
    const token = ++ctx.bindToken;
    const loader = new THREE.TextureLoader();
    for (const fid of FACE_IDS) {
      const fm = boxManifest.faces[fid];
      if (!fm) continue;
      const rt = ctx.faces[fid];
      const rawRecipe = fm.render_recipe;
      const recipe: RenderRecipe =
        rawRecipe && rawRecipe in RECIPE_IDS
          ? (rawRecipe as RenderRecipe)
          : 'moire_interactive';
      // Recipe extras (slimmed recipe_data still carries scalar knobs +
      // the stereo view PNG urls — only frame_scene is stripped).
      const rd = fm.recipe_data ?? {};
      const viewAUrl =
        recipe === 'stereo_lenticular' && typeof rd.view_a_png === 'string'
          ? rd.view_a_png
          : null;
      const viewBUrl =
        recipe === 'stereo_lenticular' && typeof rd.view_b_png === 'string'
          ? rd.view_b_png
          : null;
      const loads = [loader.loadAsync(fm.files.front_png), loader.loadAsync(fm.files.back_png)];
      if (viewAUrl && viewBUrl) {
        loads.push(loader.loadAsync(viewAUrl), loader.loadAsync(viewBUrl));
      }
      Promise.all(loads)
        .then((texs) => {
          if (ctxRef.current !== ctx || ctx.bindToken !== token) {
            for (const tex of texs) tex.dispose();
            return;
          }
          const [front, back] = texs;
          const viewA: THREE.Texture | undefined = texs[2];
          const viewB: THREE.Texture | undefined = texs[3];
          for (const tex of texs) {
            tex.colorSpace = THREE.LinearSRGBColorSpace;
            tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.magFilter = THREE.LinearFilter;
            tex.minFilter = THREE.LinearFilter;
            tex.generateMipmaps = false;
            tex.anisotropy = 8;
            tex.needsUpdate = true;
          }
          for (const old of rt.textures) old.dispose();
          rt.textures = texs;
          const u = rt.shader.uniforms;
          u.uFront.value = front;
          u.uBack.value = back;
          // (width, height) — UV u spans the width, v the height; wall
          // plates are non-square so the parallax needs both axes.
          (u.uExtentUm.value as THREE.Vector2).set(fm.extent_um[0], fm.extent_um[1]);
          u.uThicknessUm.value = fm.substrate.thickness_um;
          u.uN.value = fm.substrate.n;
          u.uRecipe.value = RECIPE_IDS[recipe];
          if (recipe === 'stereo_lenticular') {
            u.uSlitOrientation.value = ((Number(rd.slit_axis_deg ?? 0) || 0) * Math.PI) / 180;
            u.uSlitPeriodUm.value = Number(rd.slit_period_um ?? 40.0) || 40.0;
            // Fall back to the front mask if the manifest lacks view urls.
            u.uViewA.value = viewA ?? front;
            u.uViewB.value = viewB ?? front;
          } else {
            u.uViewA.value = front;
            u.uViewB.value = front;
          }
          if (recipe === 'phase_shift_overlay') {
            u.uSwitchAxis.value = ((Number(rd.switch_axis_deg ?? 0) || 0) * Math.PI) / 180;
            u.uCarrierPeriodUm.value = Number(rd.carrier_period_um ?? 20.0) || 20.0;
          }
          log('face_texture_bound', {
            face: fid,
            slug: fm.spec.pattern_slug,
            recipe,
            stereo_views: !!(viewA && viewB),
          });
        })
        .catch((e) => {
          log('face_texture_failed', { face: fid, error: (e as Error).message });
        });
    }
  }, [boxManifest]);

  // --- auto-rotate (slow turntable) ------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    ctx.controls.autoRotate = autoRotate;
    const w = window as unknown as { __studio?: StudioHandle };
    if (w.__studio) w.__studio.autoRotate = autoRotate;
  }, [autoRotate]);

  // --- selected face highlight + gentle camera tween -------------------------
  const firstSelectRef = useRef(true);
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    for (const fid of FACE_IDS) {
      const g = ctx.faces[fid].glassMat;
      const sel = fid === selectedFaceId;
      g.emissive.setHex(sel ? 0x1d4a7a : 0x000000);
      g.emissiveIntensity = sel ? 0.5 : 0.0;
    }
    // Tween the camera azimuth toward the selected wall (~400 ms ease).
    // Skipped on mount, for top/bottom (no natural azimuth), in flat layout,
    // and while the user is orbit-dragging — never fight OrbitControls.
    if (firstSelectRef.current) {
      firstSelectRef.current = false;
      return;
    }
    const toAz = FACE_AZIMUTH[selectedFaceId];
    if (toAz === undefined) return;
    if (ctx.userDragging || useStore.getState().layout === 'flat') return;
    const offset = ctx.camera.position.clone().sub(ctx.controls.target);
    const fromAz = new THREE.Spherical().setFromVector3(offset).theta;
    // Take the short way around the circle.
    let to = toAz;
    while (to - fromAz > Math.PI) to -= 2 * Math.PI;
    while (to - fromAz < -Math.PI) to += 2 * Math.PI;
    ctx.camTween = { fromAz, toAz: to, t0: performance.now(), durMs: 400 };
  }, [selectedFaceId]);

  // --- lid target ------------------------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (ctx) ctx.lidTargetDeg = lidTargetDeg;
  }, [lidTargetDeg]);

  // --- illumination / laser color --------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const illumId = illumination === 'ambient' ? 0 : illumination === 'laser' ? 1 : 2;
    const laserRgb: Record<string, number> = {
      red: 0xff3355,
      green: 0x44ff88,
      blue: 0x3388ff,
    };
    for (const fid of FACE_IDS) {
      ctx.faces[fid].shader.uniforms.uIllumination.value = illumId;
      ctx.faces[fid].shader.uniforms.uLaserColor.value.setHex(laserRgb[laserColor]);
    }
  }, [illumination, laserColor]);

  // --- light direction ---------------------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const az = (lightAz * Math.PI) / 180;
    const el = (lightEl * Math.PI) / 180;
    const r = 3.0;
    const x = r * Math.cos(el) * Math.sin(az);
    const y = r * Math.sin(el);
    const z = r * Math.cos(el) * Math.cos(az);
    for (const fid of FACE_IDS) {
      ctx.faces[fid].shader.uniforms.uLightWorld.value.set(x, y, z);
    }
    ctx.keyLight.position.set(x, y, z);
  }, [lightAz, lightEl]);

  return (
    <div
      ref={mountRef}
      data-testid="box-scene"
      style={{ width: '100%', height: '100%', minHeight: 0, minWidth: 0 }}
    />
  );
}
