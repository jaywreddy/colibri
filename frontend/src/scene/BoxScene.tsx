import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { buildStudioEnvScene } from './studioEnv';
import {
  makeFoilMaps,
  makeSolderMaps,
  makeStripHeatColor,
  hashStr,
  setMetalTextureAnisotropy,
} from './metalTextures';
import { makeBeadGeometry, makeCornerBlob } from './solderBead';
import vert from '../shaders/plate.vert';
import frag from '../shaders/plate.frag';
import { log } from '../logger';
import { useStore } from '../store';
import { KIT, Button } from '../ui/kit';
import { FACE_IDS, RECIPE_IDS, type FaceId, type PlateManifest } from '../api';
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
/**
 * How long a lost GPU context gets to fire `webglcontextrestored` before we
 * stop waiting and offer a manual renderer reinit. Browsers normally restore
 * within a few hundred ms; a driver reset under memory pressure may never.
 */
const RESTORE_GRACE_MS = 5000;
/** Face mask loads: total attempts (1 initial + retries) and the retry delay. */
const FACE_TEXTURE_ATTEMPTS = 2;
const FACE_TEXTURE_RETRY_MS = 1200;
/**
 * Diffraction-LUT fetch: retry budget for a COLD BACKEND START.
 *
 * The renderer mounts with the page, but `just dev` needs ~40 s on this host to
 * import numpy/shapely/klayout and bind :8765 — until then the vite proxy
 * answers ECONNREFUSED (surfaced to the client as a 500). A 9 s window missed
 * that entirely and the spectral accent stayed silently dead for the session.
 * Backoff from 1 s to a 5 s cap over 15 attempts is ~68 s of cover, and each
 * miss costs one failed fetch. This runs once per renderer — nothing else
 * would ever retry it.
 */
const DIFF_LUT_ATTEMPTS = 15;
const DIFF_LUT_RETRY_BASE_MS = 1000;
const DIFF_LUT_RETRY_MAX_MS = 5000;
/**
 * Dirty-flag render loop.
 *
 * The scene is time-INVARIANT by contract (see CLAUDE.md renderer honesty), so
 * an idle frame is bit-identical to the one before it — and it is not cheap:
 * the glass slabs are MeshPhysicalMaterial with transmission > 0, which makes
 * three render the whole scene TWICE per frame into a transmission render
 * target. So the loop keeps running (damping, lid tween and camera tween all
 * need their per-frame integration) but only draws when something changed.
 *
 * DIRTY_FRAMES is the debt any change books: > 1 because a change can land
 * mid-frame and because a freshly bound texture / recompiled program may not be
 * resident on the first draw after it.
 *
 * IDLE_RENDER_MS is the safety valve. Test harnesses and the debug console
 * mutate the scene graph and uniforms directly through window.__studio, outside
 * any React effect that could mark the scene dirty (effectsHelpers'
 * scaleBackPlaneGap does exactly this). Those callers all
 * render explicitly before reading pixels, but a slow heartbeat means anything
 * that does NOT still shows up promptly, for ~2 fps instead of 60.
 */
const DIRTY_FRAMES = 3;
const IDLE_RENDER_MS = 500;

/**
 * Per-finish physically-based surface params. Foil tape, solder beads, and
 * tinned rims share the finish tint (FOIL_COLORS) but differ in microsurface:
 *   - foil is rolled tape (moderately smooth),
 *   - solder is a flowed-then-frozen bead — satin, rounder highlights, and a
 *     thin clearcoat that reads as the wet sheen real solder keeps,
 *   - patina is oxidized: it loses metallic reflectance (lower metalness) and
 *     scatters more (higher roughness), so it looks matte and dark.
 * envMapIntensity is > 1 because the studio env is the only thing a metalness=1
 * surface reflects — at 1.0 against a dark background the metals crush to black.
 *
 * metalness is deliberately held < 1.0 even for the "clean" finishes. A perfect
 * conductor has NO diffuse term, so a metalness=1 facet shows ONLY its env
 * reflection; the small foil frame strips reflect toward directions the studio
 * lightbox does not fill, so they crushed to flat black off the key axis and
 * the lightest finish (bright) rendered DARKER than the dark finishes (patina,
 * gunmetal) whose lower metalness gave them a diffuse floor — a finish
 * inversion. Real rolled foil tape and flowed solder carry a thin diffuse
 * oxide/scatter component anyway, so a metalness in the ~0.8 range keeps them
 * unmistakably metallic while letting the three-point rig light their tint from
 * every angle. Verified: at az=28/el=20, bright's foil-strip luminance now
 * clearly exceeds patina's and gunmetal's (was 0 vs 22/16 — inverted).
 */
type FinishKey = keyof typeof FOIL_COLORS;
type FinishPbr = {
  foilRough: number;
  solderRough: number;
  tinRough: number;
  metalness: number;
  clearcoat: number;
  clearcoatRough: number;
  env: number;
  /** 0 = clean rolled tape, 1 = heavily oxidised — drives texture mottle. */
  oxidation: number;
  /** Solder micro-relief strength (mm). */
  bumpScale: number;
  /**
   * Brushed-metal ANISOTROPY of the rolled foil (MeshPhysicalMaterial
   * .anisotropy): rolled copper tape has a strongly directional micro-groove
   * structure, so its highlight stretches along the roll axis — the single
   * strongest "this is real rolled tape" cue. Oxidation buries the grooves,
   * so the dark finishes carry much less.
   */
  foilAniso: number;
};
const FINISH_PBR: Record<FinishKey, FinishPbr> = {
  bright: {
    foilRough: 0.26,
    solderRough: 0.34,
    tinRough: 0.3,
    // < 1.0 so the diffuse floor carries the silver tint on off-axis strips
    // (see FINISH_PBR docstring — a perfect conductor crushed these to black).
    metalness: 0.82,
    clearcoat: 0.4,
    clearcoatRough: 0.45,
    env: 2.4,
    oxidation: 0.12,
    bumpScale: 0.012,
    foilAniso: 0.55,
  },
  copper: {
    foilRough: 0.36,
    solderRough: 0.42,
    tinRough: 0.38,
    metalness: 0.82,
    clearcoat: 0.3,
    clearcoatRough: 0.5,
    env: 2.2,
    oxidation: 0.3,
    bumpScale: 0.016,
    foilAniso: 0.45,
  },
  patina: {
    foilRough: 0.62,
    solderRough: 0.6,
    tinRough: 0.58,
    metalness: 0.7,
    clearcoat: 0.12,
    clearcoatRough: 0.7,
    env: 1.7,
    oxidation: 0.95,
    bumpScale: 0.02,
    foilAniso: 0.15,
  },
  gold: {
    foilRough: 0.22,
    solderRough: 0.3,
    tinRough: 0.26,
    metalness: 0.85,
    clearcoat: 0.45,
    clearcoatRough: 0.4,
    env: 2.6,
    oxidation: 0.08,
    bumpScale: 0.01,
    foilAniso: 0.55,
  },
  rose: {
    foilRough: 0.3,
    solderRough: 0.38,
    tinRough: 0.32,
    metalness: 0.83,
    clearcoat: 0.35,
    clearcoatRough: 0.45,
    env: 2.4,
    oxidation: 0.18,
    bumpScale: 0.012,
    foilAniso: 0.5,
  },
  gunmetal: {
    foilRough: 0.44,
    solderRough: 0.5,
    tinRough: 0.46,
    metalness: 0.8,
    clearcoat: 0.2,
    clearcoatRough: 0.5,
    env: 2.0,
    oxidation: 0.5,
    bumpScale: 0.016,
    foilAniso: 0.35,
  },
};

/**
 * Linear conductor response per litho metal (renderer-audit item 5). The masks
 * are metal-agnostic; only the preview shading follows this. `gold` is the
 * legacy plate.frag constants EXACTLY (GOLD / GOLD_BACK / GOLD_F0), so a gold
 * box renders bit-identically to the pre-uniform build and the @effects gates
 * see the same pixels. `chrome` is bright standard chrome (measured Cr F0 —
 * near-neutral, the platinum-line read); `chrome-ar` the low-reflective
 * AR-coated mask grade (ink-black linework, a few percent reflectance).
 */
type MetalLook = {
  albedo: [number, number, number];
  back: [number, number, number];
  f0: [number, number, number];
  /** Broad-lobe weight — a near-mirror film puts less energy here. */
  body: number;
  /** Half-vector lobe weight. */
  spec: number;
  /** Specular exponent (smoothness proxy). */
  gloss: number;
  /** Reflectance approached at grazing: 1.0 bare conductor, lower for AR. */
  grazing: number;
  /** Diffraction-accent efficiency, scaled by the film's reflectance. */
  sheen: number;
  /**
   * Environment-reflection weight — how much of the film's return is a mirror
   * image of the room rather than broad scatter. 0 keeps a metal on the
   * single-light model it was calibrated against (gold); a smooth mask chrome
   * needs this to read as metal at all.
   */
  env: number;
};

/** Hemisphere surround the plate metal reflects — matches the scene's
 * HemisphereLight so plates and metalwork share one room. */
const ENV_SKY: [number, number, number] = [0.72, 0.79, 0.9];
const ENV_GROUND: [number, number, number] = [0.11, 0.09, 0.07];

const METAL_LOOKS: Record<'gold' | 'chrome' | 'chrome-ar', MetalLook> = {
  // Evaporated Au on quartz — the legacy constants; body/spec/gloss/grazing/
  // sheen are gold's ORIGINAL hardcoded values, so this path is unchanged.
  gold: {
    albedo: [0.791, 0.503, 0.08],
    back: [0.133, 0.084, 0.013],
    f0: [1.0, 0.766, 0.336],
    body: 1.0,
    spec: 0.5,
    gloss: 80,
    grazing: 1.0,
    sheen: 1.0,
    env: 0.0,
  },
  // Mask-grade Cr on polished glass: R0 ~0.55 neutral and a very smooth film,
  // so it is far more MIRROR than gold — most of its energy belongs in a tight
  // bright lobe, not the body. Rendering it with gold's split is what made it
  // read as grey paint.
  chrome: {
    albedo: [0.42, 0.427, 0.432],
    back: [0.071, 0.072, 0.073],
    f0: [0.549, 0.556, 0.554],
    body: 0.62,
    spec: 1.45,
    gloss: 190,
    grazing: 1.0,
    sheen: 0.79,
    env: 0.55,
  },
  // Low-reflective (AR chrome-oxide) mask grade: ~5-8% reflectance, and the
  // coating exists specifically to KILL the specular return — so it is a dark
  // near-matte absorber with a real grazing ceiling (see uMetalGrazing).
  'chrome-ar': {
    albedo: [0.048, 0.051, 0.055],
    back: [0.01, 0.011, 0.012],
    f0: [0.06, 0.065, 0.072],
    body: 1.0,
    spec: 0.12,
    gloss: 32,
    grazing: 0.32,
    sheen: 0.09,
    env: 0.05,
  },
};

/** Max rocking angle (deg, each axis) in face-inspection mode. */
const INSPECT_MAX_DEG = 8;
/** Drag sensitivity in inspection mode (deg per CSS pixel). */
const INSPECT_DEG_PER_PX = 0.045;

type FaceRT = {
  faceId: FaceId;
  /**
   * TWO real surfaces per plate (see makePlateShader / buildPlate). Each runs the
   * plate shader for ONE layer (uLayer) and binds its OWN mask to uFront — no
   * cross-layer sampling. `shader` drives the OUTER plane (front layer, z=+T/2),
   * `shaderBack` the INNER plane (back layer, z=-T/2). Cross-layer illusions
   * emerge from the perspective projection of the two planes. `shader` keeps its
   * name (not `shaderFront`) so the demo-box shot harness's `faces.*.shader`
   * probe still resolves. Persistent across rebuilds.
   */
  shader: THREE.ShaderMaterial;
  shaderBack: THREE.ShaderMaterial;
  /** The fused-silica slab material — persistent across rebuilds. */
  glassMat: THREE.MeshPhysicalMaterial;
  /** Front/back PNG textures currently bound (tracked for disposal). */
  textures: THREE.Texture[];
  /**
   * Manifest URLs of the masks currently bound, or null when this face has
   * nothing real on it yet (fresh renderer, refused recipe, failed load). The
   * bind pass compares against these and skips the fetch+decode+upload when a
   * new manifest resolves to the same two PNGs — a one-face edit used to reload
   * all twelve.
   */
  boundFront: string | null;
  boundBack: string | null;
};

type RebuildDisposables = {
  geoms: THREE.BufferGeometry[];
  mats: THREE.Material[];
  texs: THREE.Texture[];
};

/**
 * The metalwork materials whose look is a pure function of `spec.foil.finish`.
 *
 * Held on the ctx so a finish change can RESTYLE them in place instead of
 * rebuilding ~90 BufferGeometries that the finish cannot possibly affect (see
 * the geomKey / finishKey split). `strips` records the two flags and the seed
 * each per-strip material needs to re-derive its own maps.
 */
type FinishMats = {
  // Physical (not Standard) so the rolled-tape anisotropy is expressible.
  foil: THREE.MeshPhysicalMaterial;
  solder: THREE.MeshPhysicalMaterial;
  tin: THREE.MeshPhysicalMaterial;
  strips: {
    mat: THREE.MeshPhysicalMaterial;
    heat: boolean;
    isVertical: boolean;
    seed: number;
  }[];
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
  /**
   * ITEM 7 — the soft ground-shadow blob of the CURRENT build, so the light-direction
   * effect can drive it. Null in flat layout, which has no ground plane.
   */
  groundShadow: THREE.Mesh | null;
  faces: Record<FaceId, FaceRT>;
  raycastTargets: THREE.Mesh[];
  rebuildDisposables: RebuildDisposables;
  /** Finish-driven materials of the CURRENT build (null before the first). */
  finishMats: FinishMats | null;
  keyLight: THREE.DirectionalLight;
  pmrem: THREE.PMREMGenerator;
  envTex: THREE.Texture;
  /** Baked diffraction colour table (null until the fetch lands). */
  diffLut: THREE.DataTexture | null;
  /**
   * The 1x1 placeholder every sampler starts on. Held here so the bind path can
   * point a sampler BACK at it when a layer turns out to carry no chrome —
   * leaving the disposed previous texture bound is a use-after-free on the GPU.
   */
  blank: THREE.DataTexture;
  /** Background gradient + soft ground-shadow textures (created once). */
  bgTex: THREE.CanvasTexture;
  /** Backdrop presets incl. the backlight light-table field (created once). */
  backdropTexs: Record<'studio' | 'velvet' | 'daylight' | 'lighttable', THREE.CanvasTexture>;
  shadowTex: THREE.CanvasTexture;
  lidCurrentDeg: number;
  lidTargetDeg: number;
  bindToken: number;
  /** Gentle camera-azimuth tween toward the selected face (null = idle). */
  camTween: CamTween | null;
  /** True while the user is orbit-dragging — tweens must not fight it. */
  userDragging: boolean;
  /** Frames the render loop still owes (dirty-flag loop; see DIRTY_FRAMES). */
  renderDebt: number;
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
  /**
   * Book `frames` more draws with the dirty-flag render loop. Anything that
   * mutates the scene through this handle rather than through the store should
   * call it (or render explicitly, as the pixel harnesses do).
   */
  requestRender: (frames?: number) => void;
};

/**
 * Render the studio lightbox to a PMREM environment map.
 *
 * Metals (foil, solder, brass) need an env map to read as metal at all —
 * RoomEnvironment is too dim and crushes them to black, so we render our own
 * bright lightbox (see studioEnv) and throw the source scene away; the visible
 * background stays the dark gradient and the env only feeds reflections.
 *
 * Callable more than once ON PURPOSE. The env map is a render TARGET with no
 * CPU-side source, so unlike every image/canvas/data texture in the scene
 * three.js cannot re-upload it after a GPU context loss — it comes back dead and
 * every metal (metalness ~0.8-1.0, see FINISH_PBR) crushes to black for the rest
 * of the session. The context-restore handler calls this to build a fresh one.
 */
function makeStudioEnv(renderer: THREE.WebGLRenderer): {
  pmrem: THREE.PMREMGenerator;
  envTex: THREE.Texture;
} {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = buildStudioEnvScene();
  const envTex = pmrem.fromScene(envScene, 0.02).texture;
  envScene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    const kill = (x: THREE.Material) => {
      (x as THREE.MeshBasicMaterial).map?.dispose();
      x.dispose();
    };
    if (Array.isArray(mat)) mat.forEach(kill);
    else if (mat) kill(mat);
  });
  return { pmrem, envTex };
}

function makeBlankTexture(): THREE.DataTexture {
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  blank.needsUpdate = true;
  return blank;
}

/**
 * Load one mode-L PNG as a SINGLE-CHANNEL LOD-0 texture — the loader for the
 * literal fabricated-chrome rasters and their period maps.
 *
 * Why not `TextureLoader`: it hands the browser's decoded `<img>` straight to
 * `texImage2D`, which uploads RGBA. Twelve 2048² rasters plus period maps at
 * 4 bytes/texel is ~200 MB of VRAM for data that is one byte wide; decoding to
 * `RedFormat` here makes it ~50 MB. The shader only ever reads `.r`, so nothing
 * downstream changes.
 *
 * NO MIPMAPS, deliberately. These used to be mipmapped + anisotropic, on the
 * reasoning that the mip level the GPU picks IS the eye's integration over the
 * pixel footprint. That is true of ONE layer in isolation and false of the pair:
 * what the eye integrates is the transmission of the STACK, and the stack's
 * coverage 1 - (1-A)(1-B) is a product, so pre-averaging each layer and combining
 * afterwards computes <1-A>·<B> instead of <(1-A)·B>. Those differ exactly where
 * the two layers are correlated across the footprint — i.e. on every cross-layer
 * effect the renderer exists to show — and at the default camera (~1 screen pixel
 * per 99 µm carrier period) the pre-averaged form flattens the barrier switches,
 * the moiré and the fringes to a uniform quarter tone. plate.frag now supersamples
 * the PRODUCT at raster resolution instead (runLiteralLayer), so every tap must be
 * an honest LOD-0 read: no mip chain, LinearFilter both ways (a mipmap minFilter
 * without a chain is an incomplete texture), and anisotropy is meaningless without
 * mips. The level-coded procedural masks stay NEAREST for their own reason —
 * interpolating region CODES invents codes that never existed.
 *
 * Returns `null` for an all-zero raster (a blank face, or the back of a
 * single-ply one): the caller drops that layer rather than uploading 4 MB of
 * zeros.
 *
 * Rows are flipped here, not by `flipY`: three leaves `flipY` false on a
 * DataTexture, and matching the `TextureLoader` convention (image row 0 at
 * v = 1) in the copy is deterministic across drivers.
 */
async function loadCoverageTexture(url: string): Promise<THREE.DataTexture | null> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''}`);
  const bitmap = await createImageBitmap(await res.blob(), {
    // Raw bytes, not a colour-managed image: this is a coverage map.
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const w = bitmap.width;
  const h = bitmap.height;
  let rgba: Uint8ClampedArray;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const c2d = canvas.getContext('2d', { willReadFrequently: true });
    if (!c2d) throw new Error('2d context unavailable');
    c2d.drawImage(bitmap, 0, 0);
    rgba = c2d.getImageData(0, 0, w, h).data;
  } finally {
    bitmap.close();
  }
  const red = new Uint8Array(w * h);
  let peak = 0;
  for (let y = 0; y < h; y++) {
    const src = y * w * 4;
    const dst = (h - 1 - y) * w; // flipY, done in the copy
    for (let x = 0; x < w; x++) {
      const v = rgba[src + x * 4];
      red[dst + x] = v;
      if (v > peak) peak = v;
    }
  }
  if (peak === 0) return null; // no chrome anywhere on this layer
  const tex = new THREE.DataTexture(red, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.NoColorSpace; // coverage, not colour — never transfer-decoded
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter; // no mip chain — see the doc comment
  tex.generateMipmaps = false;
  tex.anisotropy = 1; // irrelevant without mips
  tex.unpackAlignment = 1; // one byte per texel: rows are not 4-aligned
  // The honesty harness identifies a bound mask by the URL of its image; a
  // DataTexture's image is a bare {data,width,height}, so record the source
  // here (same contract an HTMLImageElement gives for the level-coded masks).
  (tex.image as unknown as { src?: string }).src = url;
  tex.needsUpdate = true;
  return tex;
}

/** Vertical two-stop gradient backdrop texture (sRGB). */
function makeGradientTexture(stops: [number, string][]): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 512;
  const c2d = canvas.getContext('2d')!;
  const grad = c2d.createLinearGradient(0, 0, 0, 512);
  for (const [at, color] of stops) grad.addColorStop(at, color);
  c2d.fillStyle = grad;
  c2d.fillRect(0, 0, 2, 512);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Backdrop presets (renderer-audit items 2/6): what the transmissive faces
 * read AGAINST is half of how every effect looks. 'studio' is the classic dark
 * gradient; 'velvet' a deep warm jeweler's ground; 'daylight' a soft bright
 * cool field that turns the gold linework into silhouette-and-glint.
 * 'lighttable' is not user-selectable — it is what the backlight illumination
 * mode swaps in: a bright warm-white diffuse field (the mask-inspection view),
 * so the gold reads as dark silhouette lines in transmission, which is exactly
 * how you would proof these masks on a real light table.
 */
function makeBackdropTextures(): Record<
  'studio' | 'velvet' | 'daylight' | 'lighttable',
  THREE.CanvasTexture
> {
  return {
    studio: makeGradientTexture([
      [0, '#161d30'],
      [0.55, '#0b0e16'],
      [1, '#06070b'],
    ]),
    velvet: makeGradientTexture([
      [0, '#301218'],
      [0.5, '#180a0e'],
      [1, '#0a0406'],
    ]),
    daylight: makeGradientTexture([
      [0, '#dde5ef'],
      [0.6, '#b9c5d4'],
      [1, '#97a4b5'],
    ]),
    lighttable: makeGradientTexture([
      [0, '#f6f3ea'],
      [0.6, '#efe9dc'],
      [1, '#ddd6c6'],
    ]),
  };
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

/**
 * ITEM 7 — pose the ground-shadow blob from the key-light direction.
 *
 * There is no shadow mapping anywhere in the scene (see the dropped-items note in the
 * plan: `plate.frag` has no `<shadowmap_pars_fragment>` so the plates could not
 * RECEIVE shadows, the depth pass ignores `alphaToCoverage` so the pattern planes
 * would CAST as solid opaque rectangles, and the transmissive slab would cast solid
 * black). The blob is the honest cheap stand-in — but it used to be completely INERT
 * to the light: you could swing the light-azimuth slider all the way around and the
 * shadow would not move, which reads as obviously fake and left the light control
 * decoupled from the scene's grounding.
 *
 * This projects the light direction onto the ground plane and drives three things:
 *   - offset, away from the light, growing with cot(elevation);
 *   - elongation along that same azimuth (a low light rakes the blob out);
 *   - opacity, from the elevation (a high light gives a tight dark contact shadow;
 *     a low one gives a long faint smear).
 *
 * `casterH` is the effective caster height (half the box height) and `reach` bounds
 * the offset so a near-horizon light cannot fling the blob out of frame.
 *
 * Static per light setting — no time term, no history — so time-invariance,
 * determinism, the lid round-trip and the turntable-off gates are all unaffected. The
 * blob also sits below the box, outside every plate ROI (facePlateROI projects the
 * OUTER PLANE's geometry bbox), with renderOrder -1 and depthWrite false.
 */
function poseGroundShadow(
  shadow: THREE.Mesh,
  lightAzDeg: number,
  lightElDeg: number,
  casterH: number,
  reach: number
): void {
  const az = (lightAzDeg * Math.PI) / 180;
  // Clamp the elevation away from the horizon so cot() stays finite.
  const el = THREE.MathUtils.clamp((lightElDeg * Math.PI) / 180, 0.14, Math.PI / 2);
  const cot = Math.cos(el) / Math.sin(el);

  // Ground-plane offset, directly away from the light's horizontal bearing.
  const dist = Math.min(casterH * cot, reach);
  shadow.position.x = -Math.sin(az) * dist;
  shadow.position.z = -Math.cos(az) * dist;

  // In-plane rotation so the blob's local +X runs along that same bearing. The mesh is
  // already flattened by rotation.x = -PI/2 (local +X -> world +X, local +Y -> world
  // -Z); with three's default 'XYZ' Euler order the z term is applied first, i.e.
  // about the plane's own normal, so it is a genuine in-plane spin. Solving
  // (cos t, 0, -sin t) = (-sin az, 0, -cos az) gives t = az + PI/2.
  shadow.rotation.set(-Math.PI / 2, 0, az + Math.PI / 2);

  // Rake it out along that bearing as the light drops; keep the transverse axis fixed.
  shadow.scale.set(1 + 0.55 * Math.min(cot, 3.0), 1, 1);

  // A high light gives a tight dark contact shadow, a low one a long faint smear.
  const m = shadow.material as THREE.MeshBasicMaterial;
  m.opacity = THREE.MathUtils.clamp(0.3 + 0.7 * Math.sin(el), 0.18, 1.0);
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

function makePlateShader(blank: THREE.Texture, layer: number): THREE.ShaderMaterial {
  const m = new THREE.ShaderMaterial({
    vertexShader: vert,
    fragmentShader: frag,
    side: THREE.DoubleSide,
    // foliage_moire outputs per-fragment gold coverage as alpha. Rendered in the
    // OPAQUE pass with alphaToCoverage (the renderer runs MSAA) instead of
    // blending, so: (a) the fine gratings get MSAA-smoothed edges without the old
    // in-shader supersampling, (b) each plate writes real depth → correct
    // occlusion between the six walls, and (c) the planes land in the glass
    // slab's transmission backdrop, so the inner (back) plane is visible THROUGH
    // the slab from outside. Blending would break all three.
    transparent: false,
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
      // Composed plates only ever run foliage_moire (the bind path refuses anything
      // else), so that is also the pre-bind default: with the blank mask it yields
      // zero gold coverage → invisible planes until a manifest binds, instead of the
      // banned single-plane model rendering opaque dark plates for a few frames.
      // Not read by plate.frag (it implements exactly one recipe now) — this
      // records WHICH recipe the bind path accepted for this face, which is what
      // the @effects suite asserts on and what makes a refused manifest visible
      // in a dump.
      uRecipe: { value: RECIPE_IDS.foliage_moire },
      uSlitPeriodUm: { value: 40.0 },
      uSwitchAxis: { value: 0.0 },
      uCarrierPeriodUm: { value: 20.0 },
      uCarrierAngle: { value: 0.0 },
      uSlitAngle: { value: 0.0 },
      uGratingDuty: { value: 0.5 },
      uCenterPeriodUm: { value: 220.0 },
      // Centerpiece art-box uv rect: gates the barrier-interlace comb (the comb
      // spans the whole box). (0,0) → shader falls back to treating the whole
      // face as the art box; the bind path logs it.
      uArtBoxHalfUv: { value: new THREE.Vector2(0.0, 0.0) },
      uArtBoxCenterUv: { value: new THREE.Vector2(0.5, 0.5) },
      // Per-motif frame-band angle bucket encoding (foliage_moire).
      uFrameBucket0: { value: 96.0 / 255.0 },
      uFrameBucketStep: { value: 14.0 / 255.0 },
      uFrameBucketCount: { value: 6.0 },
      // Backend ships 3.5 deg (frame_angle_span_deg); the bind path overwrites
      // this, but a default that disagrees with the producer is a trap for any
      // face that renders before its manifest lands.
      uFrameAngleSpan: { value: (3.5 * Math.PI) / 180.0 },
      // 0 = OUTER plane (front layer), 1 = INNER plane (back layer).
      uLayer: { value: layer },
      // Litho-metal conductor response (renderer-audit item 5). Defaults are
      // the legacy GOLD constants; the metal effect rebinds them per spec.metal.
      uMetalAlbedo: { value: new THREE.Color().setRGB(...METAL_LOOKS.gold.albedo, THREE.LinearSRGBColorSpace) },
      uMetalAlbedoBack: { value: new THREE.Color().setRGB(...METAL_LOOKS.gold.back, THREE.LinearSRGBColorSpace) },
      uMetalF0: { value: new THREE.Color().setRGB(...METAL_LOOKS.gold.f0, THREE.LinearSRGBColorSpace) },
      uMetalBody: { value: METAL_LOOKS.gold.body },
      uMetalSpec: { value: METAL_LOOKS.gold.spec },
      uMetalGloss: { value: METAL_LOOKS.gold.gloss },
      uMetalGrazing: { value: METAL_LOOKS.gold.grazing },
      uMetalSheen: { value: METAL_LOOKS.gold.sheen },
      uMetalEnv: { value: METAL_LOOKS.gold.env },
      // Baked diffraction table (backend app/diffraction.py). uDiffReady stays
      // 0 until the fetch lands, so a failed/slow load simply shows no accent
      // rather than a wrong colour or a dangling sampler.
      uDiffLut: { value: blank },
      uDiffUMax: { value: 10.0 },
      uDiffReady: { value: 0.0 },
      uRainbowAngleRad: { value: Math.PI / 4 },
      uRainbowZeroOrder: { value: 0.25 },
      uSkyColor: { value: new THREE.Color().setRGB(...ENV_SKY, THREE.LinearSRGBColorSpace) },
      uGroundColor: { value: new THREE.Color().setRGB(...ENV_GROUND, THREE.LinearSRGBColorSpace) },
      // Pattern Scale (Task 1b): multiplies the preview-MAGNIFIED period family
      // (frame carrier + louvre) on both planes. The centerpiece barrier pitch is
      // excluded — scaling it would scale the switch tilt angle, since the T/n
      // plane gap does not scale with it.
      uPatternScale: { value: 1.0 },
      // Barrier-interlace tilt switch: 1 on the two-ply exemplar that still has
      // one (globe-duo-phase), plus the SOLVED lattice phase the backend
      // publishes. No production wall sets it — every wall is single-ply.
      uSwitchInterlace: { value: 0.0 },
      uSwitchBarrierPhaseUm: { value: 0.0 },
      // LITERAL fabricated-geometry path. 1 = uFront carries a raster of the
      // real chrome and the shader just samples it (no procedural gratings);
      // 0 = the level-coded procedural path. Bound from recipe_data.literal.
      uLiteral: { value: 0.0 },
      // Per-pixel sub-grating pitch (files.period_front), front plane only.
      uPeriodMap: { value: blank },
      uPeriodReady: { value: 0.0 },
      // The INNER layer's chrome raster, bound on the OUTER plane's material: a
      // literal face composites both layers in ONE pass there (the eye integrates
      // their PRODUCT, which two independently filtered planes cannot express —
      // see plate.frag::runLiteralLayer), and its inner pattern plane is hidden.
      uBackCoverage: { value: blank },
      uBackCoverageReady: { value: 0.0 },
      // Raster dimensions, so the composite can size its subsample grid to at
      // most one texel per step.
      uCoverageSizePx: { value: new THREE.Vector2(2048.0, 2048.0) },
      // LIVE outer→inner plane separation (µm), refreshed every frame from the
      // real mesh positions by the outer plane's onBeforeRender (see buildPlate).
      // Design value = the paraxial air gap T/n; read from the GEOMETRY so the
      // honesty harness's scaleBackPlaneGap still collapses the cross-layer
      // effect on a literal face, whose inner plane no longer draws.
      uInnerGapUm: { value: 0.0 },
    },
  });
  m.alphaToCoverage = true;
  return m;
}

function makeGlassMaterial(): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    // ITEM 1 — substrate honesty. The inner gold plane is only ever seen THROUGH
    // this material, so its transmission sampling is the LAST low-pass filter
    // applied to the back layer; no shader-side work can recover detail that has
    // already been smeared here.
    //
    // `roughness` sets the transmission blur LOD directly (three's
    // transmission_pars_fragment):
    //     lod = log2(transmissionSamplerSize.x) * roughness * clamp(2*ior - 2, 0, 1)
    // At roughness 0.06 / ior 1.46 (factor 0.92) on a ~1400 px viewport that is
    // lod ~= 0.58 — the back carrier arrives bicubic-blurred across ~1.5 device
    // pixels. At 0.012 it is lod ~= 0.115. 0.06 describes GROUND glass; a
    // lambda/10-polished fused-silica window is an order of magnitude smoother,
    // so the new value is also the more physical one.
    //
    // `transmission` 0.85 left 15% of a white-diffuse-lit slab composited OVER the
    // back plane as a milky veil that flattened back-layer contrast globally. Bare
    // polished fused silica transmits ~92%.
    //
    // Both are set here only — the rebuild loop overwrites `ior` from spec.glass.n
    // but never these two, so this single edit persists across rebuilds.
    transmission: 0.94,
    ior: 1.46,
    roughness: 0.012,
    metalness: 0.0,
    // No volume: the T/n inner-plane placement is the only refractive
    // displacement in the scene (see the rebuild loop's g.thickness note).
    thickness: 0,
    side: THREE.DoubleSide,
    // Renderer-audit item 6: at 1.0 the broad studio bounce panels paint a
    // milky white sheen across the whole low-roughness slab and the plates
    // read as frosted acrylic. Halving the GLASS env pickup (metals keep their
    // own per-finish envMapIntensity) keeps the crisp hero-streak highlights
    // while letting the pattern layers, not the sheen, carry the face.
    envMapIntensity: 0.55,
  });
}

/**
 * Scene-graph census tags. Every plate surface and every piece of metalwork
 * carries `userData.kind` so a test can COUNT the real hardware in the scene
 * graph instead of trusting a screenshot: the visual gate asserts 48 foil
 * strips, 8 seam beads, 8 corner blobs, 8 tinned rims, `spec.hinge.segments`
 * tubes and 1 rod in the assembled layout — and their absence in flat layout
 * (`frontend/tests/e2e/visualSignatures.spec.ts`, expectations derived from
 * assembly.ts, not magic numbers).
 *
 * These strings are a TEST CONTRACT: renaming one, or adding a mesh in one of
 * these families without tagging it, silently shrinks what the gate covers.
 * `userData.faceId` (set on the raycast targets) is a separate, older contract
 * used by click-to-select — do not fold the two together.
 */
export type SceneMeshKind =
  | 'plate-outer'
  | 'plate-inner'
  | 'foil-strip'
  | 'seam-bead'
  | 'seam-corner'
  | 'tin-rim'
  | 'hinge-tube'
  | 'hinge-rod';

/** Stamp a census tag on a mesh and return it (so it can wrap an expression). */
function tag<T extends THREE.Object3D>(o: T, kind: SceneMeshKind): T {
  o.userData.kind = kind;
  return o;
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
    const m = tag(new THREE.Mesh(geo(new THREE.PlaneGeometry(sw, sh)), foilMat), 'foil-strip');
    m.position.set(cx, cy, z);
    parent.add(m);
  }
}

/**
 * Foil frame for the assembled box: each of the four strips gets its own
 * material so (a) the brushed roughness streaks run along the strip's physical
 * long axis, and (b) a per-strip heat-patina colour map darkens the outer long
 * edge (the edge welded to the plate join) and fades inward.
 *
 * `mkStripMat` builds a strip material given the shared foil rough map, the
 * per-strip heat colour texture, and whether the strip is vertical (long axis
 * = plate Y, needs the rough map rotated 90deg).
 */
function addFoilFrameRealistic(
  parent: THREE.Group,
  w: number,
  h: number,
  ov: number,
  z: number,
  outward: boolean,
  build: (isVertical: boolean, seed: number) => THREE.Material,
  geo: <T extends THREE.BufferGeometry>(g: T) => T,
  mat: <T extends THREE.Material>(m: T) => T
): void {
  // [cx, cy, sw, sh, isVertical]
  const strips: [number, number, number, number, boolean][] = [
    [0, h / 2 - ov / 2, w, ov, false], // top
    [0, -(h / 2 - ov / 2), w, ov, false], // bottom
    [-(w / 2 - ov / 2), 0, ov, h - 2 * ov, true], // left
    [w / 2 - ov / 2, 0, ov, h - 2 * ov, true], // right
  ];
  strips.forEach(([cx, cy, sw, sh, vert], idx) => {
    if (sw <= 0 || sh <= 0) return;
    const seed = hashStr(`strip:${outward ? 'o' : 'i'}:${idx}:${w.toFixed(1)}:${h.toFixed(1)}`);
    const stripMat = mat(build(vert, seed));
    const m = tag(new THREE.Mesh(geo(new THREE.PlaneGeometry(sw, sh)), stripMat), 'foil-strip');
    m.position.set(cx, cy, z);
    parent.add(m);
  });
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
 * Write every finish-derived property onto the metalwork materials.
 *
 * The ONLY place the foil finish reaches the scene. Called once at the end of
 * each geometry rebuild (materials fresh, nothing compiled yet) and again on a
 * finish-only change, where it is the whole update — no geometry is touched.
 * Both paths run identical code so the two look the same by construction.
 *
 * Every map comes from the memoized builders in metalTextures, so a finish the
 * user has already visited costs three Map lookups instead of ~720k noise
 * samples. Those textures are module-owned: they must never land in the
 * per-rebuild disposal list.
 */
function applyFinishMats(mats: FinishMats, finishKey: FinishKey): void {
  const pbr = FINISH_PBR[finishKey];
  const tint = FOIL_COLORS[finishKey];
  const finish = new THREE.Color(tint);
  const foilMaps = makeFoilMaps(finishKey, { oxidation: pbr.oxidation, tint });
  const solderMaps = makeSolderMaps(finishKey, { oxidation: pbr.oxidation, tint });

  // Rolled copper tape: metal with brushed roughness streaks + colour mottle.
  // roughness stays 1.0 — the roughnessMap scales it.
  mats.foil.map = foilMaps.color;
  mats.foil.roughnessMap = foilMaps.rough;
  mats.foil.metalness = pbr.metalness;
  mats.foil.envMapIntensity = pbr.env;
  // Rolled-tape anisotropy: the flat-layout shared foil brushes along U
  // (horizontal); per-strip materials orient it below.
  mats.foil.anisotropy = pbr.foilAniso;
  mats.foil.anisotropyRotation = 0;

  // Solder bead: flowed metal with a satin clearcoat sheen + blotchy roughness
  // and a micro-relief bump — the clearcoat + blotch is what separates a real
  // solder joint from a chrome rod.
  mats.solder.map = solderMaps.color;
  mats.solder.roughnessMap = solderMaps.rough;
  mats.solder.bumpMap = solderMaps.bump;
  mats.solder.bumpScale = pbr.bumpScale;
  mats.solder.metalness = pbr.metalness;
  mats.solder.clearcoat = pbr.clearcoat;
  mats.solder.clearcoatRoughness = pbr.clearcoatRough;
  mats.solder.envMapIntensity = pbr.env;

  mats.tin.color.copy(finish);
  mats.tin.metalness = pbr.metalness;
  mats.tin.roughness = pbr.tinRough;
  mats.tin.envMapIntensity = pbr.env;
  // A tinned wipe is re-flowed, not rolled — barely directional.
  mats.tin.anisotropy = 0.12;

  // Base tint as 0..255 RGB for the heat-patina colour maps.
  const tintRgb: [number, number, number] = [
    Math.round(finish.r * 255),
    Math.round(finish.g * 255),
    Math.round(finish.b * 255),
  ];
  for (const s of mats.strips) {
    // Vertical strips get the pre-rotated rough map so the brush streaks run
    // along the strip's long axis (rotating the shared Texture in place would
    // affect every user of it).
    s.mat.roughnessMap = s.isVertical ? foilMaps.roughRotated : foilMaps.rough;
    if (s.heat) {
      // Outer long edge nearest the plate join gets patina. Horizontal strips:
      // the outer edge is the top/bottom (v-edges); vertical strips (rotated):
      // the outer edge is the far U end. We tint both long edges lightly so any
      // seam-adjacent border reads warm-oxidised, fading in.
      const seamEdges = s.isVertical
        ? { v0: false, v1: false, u0: true, u1: true }
        : { v0: true, v1: true, u0: false, u1: false };
      s.mat.map = makeStripHeatColor(tintRgb, seamEdges, s.seed);
    } else {
      s.mat.map = foilMaps.color;
    }
    s.mat.metalness = pbr.metalness;
    s.mat.envMapIntensity = pbr.env;
    // Brushed highlight stretches along the strip's PHYSICAL long axis: the
    // roll direction is U for horizontal strips, V (rotate 90°) for vertical
    // ones — matching the pre-rotated roughness streak maps above.
    s.mat.anisotropy = pbr.foilAniso;
    s.mat.anisotropyRotation = s.isVertical ? Math.PI / 2 : 0;
  }
}

/**
 * Presentation props (renderer-audit item 2): a velvet cushion pair and a
 * REAL-DIMENSIONED ring standing in the slot between them. Pure display
 * objects — no fab meaning, never raycast targets, no census tags — but the
 * ring's FIXED real size (size-7: 17.3 mm bore, 1.7 mm band, ~20.7 mm OD)
 * makes it an honest fit check: when the interior can't give a standing ring
 * its headroom, the pose falls back to lying flat, exactly the compromise the
 * real box would force. Default OFF (store.showRing) so the pixel-metric
 * harnesses keep seeing the exact scene they always did.
 */
function addPresentationProps(
  group: THREE.Group,
  W: number,
  Dep: number,
  H: number,
  wallMm: number,
  geo: <T extends THREE.BufferGeometry>(g: T) => T,
  mat: <T extends THREE.Material>(m: T) => T
): void {
  const iw = W - 2 * wallMm;
  const id = Dep - 2 * wallMm;
  const ih = H - 2 * wallMm;
  if (iw < 12 || id < 12 || ih < 4) return; // no room for any prop at all

  // Ring: size-7 band in mm — REAL dimensions, never scaled with the box.
  const bore = 17.3;
  const bandTube = 0.85; // circular approximation of a 1.7 mm band
  const R = bore / 2 + bandTube; // torus major radius (9.5)
  const ringOD = 2 * (R + bandTube); // 20.7
  const stoneGirdle = 2.2;
  const standingH = ringOD + 2.6; // band + stone crown headroom

  const velvet = mat(
    new THREE.MeshPhysicalMaterial({
      color: 0x5c1622,
      roughness: 0.95,
      metalness: 0.0,
      sheen: 1.0,
      sheenColor: new THREE.Color(0xb26775),
      sheenRoughness: 0.55,
    })
  );
  const bandGold = mat(
    new THREE.MeshPhysicalMaterial({
      color: 0xc9a24b,
      metalness: 1.0,
      roughness: 0.16,
      clearcoat: 0.35,
      clearcoatRoughness: 0.35,
      envMapIntensity: 2.2,
    })
  );
  const gem = mat(
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 0.0,
      roughness: 0.03,
      transmission: 0.9,
      ior: 2.417,
      thickness: 2.2,
      envMapIntensity: 3.0,
      flatShading: true,
    })
  );
  // Chromatic dispersion (the fire) shipped in newer three; harmless to skip.
  if ('dispersion' in gem) (gem as unknown as { dispersion: number }).dispersion = 0.2;

  const floorY = -H / 2 + wallMm;
  const standing = ih >= standingH + 2.5; // needs cushion sink + lid clearance
  const cushH = standing
    ? THREE.MathUtils.clamp(ih - standingH - 1.0, 2.5, 12)
    : THREE.MathUtils.clamp(ih * 0.35, 2.0, 8);

  // Cushion pair with the classic ring slot between the halves.
  const slot = 2.6;
  const cw = iw * 0.96;
  const cd = (id * 0.96 - slot) / 2;
  if (cd > 2) {
    const r = Math.min(1.4, cushH / 2.5, cd / 2.5);
    for (const sz of [-1, 1]) {
      const half = new THREE.Mesh(geo(new RoundedBoxGeometry(cw, cushH, cd, 3, r)), velvet);
      half.position.set(0, floorY + cushH / 2, sz * (slot / 2 + cd / 2));
      half.userData.prop = true;
      group.add(half);
    }
  }

  // Ring group: band torus + a small bezel + faceted stone at the top.
  const ring = new THREE.Group();
  const band = new THREE.Mesh(geo(new THREE.TorusGeometry(R, bandTube, 24, 64)), bandGold);
  ring.add(band);
  const bezel = new THREE.Mesh(geo(new THREE.TorusGeometry(stoneGirdle * 0.72, 0.38, 12, 24)), bandGold);
  bezel.rotation.x = Math.PI / 2;
  bezel.position.y = R + bandTube + 0.15;
  ring.add(bezel);
  const crown = new THREE.Mesh(
    geo(new THREE.CylinderGeometry(stoneGirdle * 0.55, stoneGirdle, 0.95, 8, 1)),
    gem
  );
  crown.position.y = R + bandTube + 1.05;
  ring.add(crown);
  const pavilion = new THREE.Mesh(
    geo(new THREE.CylinderGeometry(stoneGirdle, 0.02, 1.8, 8, 1)),
    gem
  );
  pavilion.position.y = R + bandTube - 0.35;
  ring.add(pavilion);
  for (const o of ring.children) o.userData.prop = true;

  if (standing) {
    // Standing in the slot, sunk ~2.2 mm into the cushion, facing the front.
    ring.position.set(0, floorY + cushH - 2.2 + R + bandTube, 0);
  } else {
    // Not enough headroom: the honest fallback — the ring lies flat.
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, floorY + cushH + bandTube + 0.1, 0);
  }
  group.add(ring);
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
 *
 * GPU health is user-visible: a lost WebGL context raises an overlay over the
 * mount and (if the browser never restores it) offers a renderer reinit, and a
 * face whose masks fail to load retries once before saying so on screen.
 */
export default function BoxScene() {
  const mountRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Ctx | null>(null);
  /** Suppresses the select-face camera tween on mount and on a renderer reinit. */
  const firstSelectRef = useRef(true);

  const boxSpec = useStore((s) => s.boxSpec);
  const boxManifest = useStore((s) => s.boxManifest);
  const layout = useStore((s) => s.layout);
  const lidTargetDeg = useStore((s) => s.lidTargetDeg);
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const autoRotate = useStore((s) => s.autoRotate);
  const patternScale = useStore((s) => s.patternScale);
  const illumination = useStore((s) => s.illumination);
  const laserColor = useStore((s) => s.laserColor);
  const lightAz = useStore((s) => s.lightAzimuthDeg);
  const lightEl = useStore((s) => s.lightElevationDeg);
  const showRing = useStore((s) => s.showRing);
  const backdrop = useStore((s) => s.backdrop);
  const inspectMode = useStore((s) => s.inspectMode);

  // GPU / texture health, surfaced over the canvas. 'lost' = context died and
  // the browser owes us a restore; 'stalled' = the restore never came, so the
  // user gets a manual reinit. A frozen canvas with no message reads as a hung
  // app, and on this host a driver reset mid-session is plausible.
  const [gpuStatus, setGpuStatus] = useState<'ok' | 'lost' | 'stalled'>('ok');
  /** Bumped to tear down and rebuild the whole renderer ('Reload view'). */
  const [reinitTick, setReinitTick] = useState(0);
  /** Bumped to re-run the manifest -> texture bind pass ('Retry' on failures). */
  const [rebindTick, setRebindTick] = useState(0);
  /** Faces whose masks failed every load attempt (see bind effect). */
  const [failedFaces, setFailedFaces] = useState<FaceId[]>([]);
  /** Live tilt readout for the inspection HUD (deg, face-local axes). */
  const [inspectTilt, setInspectTilt] = useState({ x: 0, y: 0 });
  /** Camera rig of the active face inspection (null = not inspecting). */
  const inspectRig = useRef<{
    center: THREE.Vector3;
    normal: THREE.Vector3;
    right: THREE.Vector3;
    up: THREE.Vector3;
    dist: number;
    savedPos: THREE.Vector3;
    savedTarget: THREE.Vector3;
    tilt: { x: number; y: number };
  } | null>(null);

  const markFaceFailed = (fid: FaceId): void =>
    setFailedFaces((prev) => (prev.includes(fid) ? prev : [...prev, fid]));
  const clearFaceFailed = (fid: FaceId): void =>
    setFailedFaces((prev) => (prev.includes(fid) ? prev.filter((f) => f !== fid) : prev));

  /** Tear the renderer down and build it again (offered when a restore stalls). */
  const reloadView = (): void => {
    log('webgl_view_reloaded');
    setGpuStatus('ok');
    firstSelectRef.current = true;
    setReinitTick((n) => n + 1);
  };

  /** Re-run the whole manifest bind pass after mask loads failed. */
  const retryMasks = (): void => {
    log('face_texture_retry', { face: 'all', manual: true });
    // Forget what is bound first. The bind pass skips a face whose mask URLs it
    // has already loaded, and 'Retry' must not be a no-op for a face that looks
    // bound but is showing an evicted or stale image.
    const ctx = ctxRef.current;
    if (ctx) {
      for (const fid of FACE_IDS) {
        ctx.faces[fid].boundFront = null;
        ctx.faces[fid].boundBack = null;
      }
    }
    setFailedFaces([]);
    setRebindTick((n) => n + 1);
  };

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
    ctx.groundShadow = null;

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
      // Refractive displacement of the back layer is carried GEOMETRICALLY by the
      // inner pattern plane sitting at the paraxial T/n gap (see buildPlate). The
      // slab's volume must therefore add NO second displacement of its own: three's
      // screen-space transmission ray offsets the backdrop by `thickness` (in world
      // units — it multiplies by the model-matrix scale), and the inner plane is
      // inside that backdrop. `T * scale` used to be assigned here, which the root
      // scale then squared to ~0.0006 mm — accidentally harmless, and the reason
      // nobody noticed. Zero states the intent: no volumetric refraction, no
      // attenuation, one displacement only. Do NOT "fix" this to T.
      g.thickness = 0;
    }

    // --- metalwork materials -------------------------------------------------
    // Structural properties only; every finish-derived property (maps, tint,
    // metalness, clearcoat, env) is written by applyFinishMats at the bottom of
    // this function, which is also the whole of the finish-only restyle path.
    // The maps come from the memoized builders and are module-owned, so nothing
    // here goes into D.texs.
    //
    // The base foilMat is used for the flat layout; the assembled layout builds
    // per-strip materials so the brush direction and heat-patina align to each
    // strip's long axis (see addFoilFrameRealistic). Double-sided so the
    // inner-border strips read from inside the open box too.
    const finishMats: FinishMats = {
      foil: mat(
        new THREE.MeshPhysicalMaterial({
          color: 0xffffff, // tint carried by the colour map
          roughness: 1.0, // scaled by the roughnessMap
          side: THREE.DoubleSide,
        })
      ),
      solder: mat(
        new THREE.MeshPhysicalMaterial({
          color: 0xffffff,
          roughness: 1.0,
        })
      ),
      tin: mat(new THREE.MeshPhysicalMaterial({})),
      strips: [],
    };
    ctx.finishMats = finishMats;
    const foilMat = finishMats.foil;
    const solderMat = finishMats.solder;
    const tinMat = finishMats.tin;

    const buildFoilStripMat = (heat: boolean) => (isVertical: boolean, seed: number) => {
      const m = new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        roughness: 1.0,
        side: THREE.DoubleSide,
      });
      finishMats.strips.push({ mat: m, heat, isVertical, seed });
      return m;
    };

    // Brass hinge hardware: warm metal, lightly lacquered (thin clearcoat).
    // Drawn brass tube is machined ALONG its axis, so the highlight stretches
    // lengthwise — cylinder UVs run V along the axis, hence rotation π/2.
    const brassMat = mat(
      new THREE.MeshPhysicalMaterial({
        color: BRASS_COLOR,
        metalness: 1.0,
        roughness: 0.28,
        clearcoat: 0.25,
        clearcoatRoughness: 0.4,
        envMapIntensity: 1.5,
        anisotropy: 0.45,
        anisotropyRotation: Math.PI / 2,
      })
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
      // (a) the glass — ONE slab. (The bonded two-ply stack, with its inset
      // inner ply and its stepped edge, went on 2026-09-16 along with the
      // nested-shell cut list it was built from.)
      {
        const slab = new THREE.Mesh(geo(new THREE.BoxGeometry(w, h, T)), rt.glassMat);
        slab.userData.faceId = fid;
        pg.add(slab);
        ctx.raycastTargets.push(slab);
      }
      // (b) TWO real gold-pattern surfaces — the physical second-surface object.
      // OUTER plane just outside the front face carries the front layer; INNER
      // plane just outside the inner face carries the back layer. They are
      // separated by the true slab thickness T, so a two-ply exemplar's shading
      // moiré and its barrier interlace both emerge from the perspective
      // projection of these two real surfaces (no in-shader parallax). The back
      // mask is authored in the same uv frame as the front (registers when viewed
      // from OUTSIDE — the primary switch view); from inside it reads as genuine
      // second-surface art (laterally reversed, as any inner-face deposition is).
      const outer = tag(
        new THREE.Mesh(geo(new THREE.PlaneGeometry(w, h)), rt.shader),
        'plate-outer'
      );
      // Bonded: the front chrome physically sits at the bond line, one ply
      // below the outer surface. Both layers' apparent depths shift by the
      // same paraxial amount, so the LAYER-TO-LAYER gap — the quantity every
      // moiré beat and switch crossing depends on, and what the @effects suite
      // scales — is T/n in both constructions. We keep the front plane
      // at the stack surface (its burial only affects parallax against the
      // glass edge, not against the back layer) and place the back plane T/n
      // below it, exactly as in the single-plate build.
      const surfaceZ = T / 2 + EPS_PATTERN_MM;
      outer.position.z = surfaceZ;
      outer.userData.faceId = fid;
      outer.renderOrder = 2;
      pg.add(outer);
      ctx.raycastTargets.push(outer);
      const inner = tag(
        new THREE.Mesh(geo(new THREE.PlaneGeometry(w, h)), rt.shaderBack),
        'plate-inner'
      );
      // TASK 2 — apparent-depth gap. The back gold layer physically sits one
      // GLASS thickness below the front layer (the far surface of the plate),
      // but refraction lifts its APPARENT position toward the viewer: a
      // paraxial ray exits the glass as if the back layer were only T/n below
      // the front. Placing the inner plane at that paraxial-equivalent air gap
      // (separation T/n below the outer plane, not the full T) makes the
      // straight-ray parallax the camera sees match the physical Snell rate
      // (~5.98 µm/deg through 500 µm fused silica at n=1.46), so a barrier
      // switch crosses at its true tilt angle. The glass slab geometry is
      // unchanged; only the pattern plane moves.
      const nGlass = spec.glass.n > 1.0 ? spec.glass.n : 1.46;
      const outerZ = surfaceZ;
      inner.position.z = outerZ - T / nGlass;
      inner.userData.faceId = fid;
      inner.renderOrder = 0;
      pg.add(inner);
      ctx.raycastTargets.push(inner);
      // LITERAL faces composite BOTH layers on the outer plane (plate.frag::
      // runLiteralLayer) and hide this inner pattern plane, so the plane gap stops
      // being expressed by the scene graph and becomes a shader parameter. Publish
      // it from the REAL mesh separation every frame rather than recomputing T/n:
      // the honesty harness manipulates the substrate by moving this very mesh
      // (effectsHelpers::scaleBackPlaneGap mutates inner.position.z directly, with
      // no uniform to update), and the anti-cheat only stays live if the composite
      // reads the geometry. Scene units are mm and mm = µm/1000, so ×1000 converts;
      // both meshes are siblings in `pg`, so their local z difference is the gap.
      // onBeforeRender runs immediately before this material's draw call, in every
      // pass (including the glass transmission backdrop), so the value is never stale.
      outer.onBeforeRender = () => {
        rt.shader.uniforms.uInnerGapUm.value = (outer.position.z - inner.position.z) * 1000;
      };
      // (c) copper foil overlap strips, outer AND inner borders.
      const zOut = T / 2 + EPS_FOIL_MM;
      const zIn = -zOut;
      const iw2 = w;
      const ih2 = h;
      const ov = Math.min(overlapMm, Math.min(iw2, ih2) / 2);
      if (ov > 1e-4) {
        if (sceneLayout === 'assembled') {
          // Outer frame: brushed + heat-patina near the welded edge.
          addFoilFrameRealistic(
            pg, w, h, ov, zOut, true,
            buildFoilStripMat(true), geo, mat
          );
          // Inner frame: brushed only (no patina inside the box).
          addFoilFrameRealistic(
            pg, iw2, ih2, ov, zIn, false,
            buildFoilStripMat(false), geo, mat
          );
        } else {
          addFoilFrame(pg, w, h, ov, zOut, foilMat, geo);
          addFoilFrame(pg, iw2, ih2, ov, zIn, foilMat, geo);
        }
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
      shadow.position.y = -H / 2 - 0.8;
      shadow.renderOrder = -1;
      group.add(shadow);
      // ITEM 7 — pose it from the CURRENT light immediately (rotation.x included), so a
      // fresh build is never briefly wrong before the light effect first runs.
      ctx.groundShadow = shadow;
      // Light-table (backlight) mode hides the blob — a contact shadow on a
      // luminous field reads as a smudge on the light box.
      shadow.visible = useStore.getState().illumination !== 'backlight';
      // Carry the two build-time scalars poseGroundShadow needs so the light effect,
      // which has no access to this closure, can re-pose the blob on its own.
      shadow.userData.casterH = H / 2;
      shadow.userData.reach = 0.75 * shadowSize;
      const lightSt = useStore.getState();
      poseGroundShadow(
        shadow,
        lightSt.lightAzimuthDeg,
        lightSt.lightElevationDeg,
        H / 2,
        0.75 * shadowSize
      );
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

      // --- presentation props (cushion + ring) — display only ----------------
      if (useStore.getState().showRing) {
        addPresentationProps(group, W, Dep, H, T, geo, mat);
      }

      // --- organic solder seam beads + corner junction blobs ----------------
      // Each seam is a tube-of-revolution with seeded radius undulation, blobby
      // spherical end caps, and a slightly asymmetric fillet cross-section
      // (see solderBead.ts). The 8 box corners get a lumpy accumulation blob so
      // meeting seams read as flowed-together solder, not clean rod ends. The
      // per-seam seed is derived from the seam id so it never flickers across
      // rebuilds. Triangle budget: ~14 radial * ~(len/0.35 + caps) rings per
      // seam + 8 blobs at subdiv 2 -> comfortably < 60k total.
      const beadR = mm(spec.foil.bead_um) / 2;
      const cornerKeys = new Set<string>();
      const cornerAt: THREE.Vector3[] = [];
      const noteCorner = (v: THREE.Vector3) => {
        const k = `${v.x.toFixed(2)},${v.y.toFixed(2)},${v.z.toFixed(2)}`;
        if (!cornerKeys.has(k)) {
          cornerKeys.add(k);
          cornerAt.push(v.clone());
        }
      };
      for (const seam of seamSegments(spec)) {
        const a = new THREE.Vector3(mm(seam.start_um[0]), mm(seam.start_um[1]), mm(seam.start_um[2]));
        const b = new THREE.Vector3(mm(seam.end_um[0]), mm(seam.end_um[1]), mm(seam.end_um[2]));
        const len = a.distanceTo(b);
        const seed = hashStr(`seam:${seam.id}`);
        const beadGeo = geo(
          makeBeadGeometry({ radius: beadR, length: len, seed, undulation: 0.09 })
        );
        const bead = tag(new THREE.Mesh(beadGeo, solderMat), 'seam-bead');
        bead.userData.seamId = seam.id;
        bead.position.copy(a).add(b).multiplyScalar(0.5);
        // bead runs along local +Y; orient to the seam axis
        if (seam.axis === 'x') bead.rotation.z = Math.PI / 2;
        else if (seam.axis === 'z') bead.rotation.x = Math.PI / 2;
        // vertical (y) seams already run along +Y — no rotation
        group.add(bead);
        noteCorner(a);
        noteCorner(b);
      }
      // Corner blobs: a touch larger than the bead so the joint looks pooled.
      for (const c of cornerAt) {
        const blobSeed = hashStr(`corner:${c.x.toFixed(2)},${c.y.toFixed(2)},${c.z.toFixed(2)}`);
        const blob = tag(
          new THREE.Mesh(geo(makeCornerBlob(beadR * 1.35, blobSeed)), solderMat),
          'seam-corner'
        );
        blob.position.copy(c);
        group.add(blob);
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
        const rim = tag(new THREE.Mesh(geo(new THREE.BoxGeometry(sx, sy, sz)), tinMat), 'tin-rim');
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
        const edge = tag(new THREE.Mesh(geo(new THREE.BoxGeometry(sx, sy, sz)), tinMat), 'tin-rim');
        edge.position.set(cx - pivotPos.x, cy - pivotPos.y, cz - pivotPos.z);
        lidPivot.add(edge);
      }

      // --- hinge: alternating tube segments + the rod ------------------------
      const tubeR = mm(hinge.tube_r_um);
      const rodR = mm(hinge.rod_r_um);
      for (const seg of hinge.segments) {
        const tube = tag(
          new THREE.Mesh(
            geo(new THREE.CylinderGeometry(tubeR, tubeR, mm(seg.length_um), 32)),
            brassMat
          ),
          'hinge-tube'
        );
        // Which half of the hinge this knuckle belongs to. The census asserts the
        // lid-owned tubes really hang off lidPivot (they must swing with the lid).
        tube.userData.owner = seg.owner;
        tube.rotation.z = Math.PI / 2;
        if (seg.owner === 'lid') {
          tube.position.set(mm(seg.center_x_um), 0, 0);
          lidPivot.add(tube);
        } else {
          tube.position.set(mm(seg.center_x_um), pivotPos.y, pivotPos.z);
          group.add(tube);
        }
      }
      const rod = tag(
        new THREE.Mesh(
          geo(new THREE.CylinderGeometry(rodR, rodR, mm(hinge.rod_length_um), 24)),
          brassMat
        ),
        'hinge-rod'
      );
      rod.rotation.z = Math.PI / 2;
      rod.position.set(0, pivotPos.y, pivotPos.z);
      group.add(rod);
      // Peened rod ends — the real assembly caps the rod so it cannot walk.
      // Deliberately UNTAGGED: the census pins exactly one 'hinge-rod'.
      for (const sx of [-1, 1]) {
        const cap = new THREE.Mesh(
          geo(new THREE.SphereGeometry(rodR * 1.5, 16, 12)),
          brassMat
        );
        cap.scale.x = 0.55; // squashed dome, like a peened head
        cap.position.set(sx * (mm(hinge.rod_length_um) / 2), pivotPos.y, pivotPos.z);
        group.add(cap);
      }
    }

    // Finish pass — the ONLY writer of finish-derived material state, shared
    // with the restyle path. Runs before the first draw of these materials, so
    // populating the map slots here costs no extra program compile.
    applyFinishMats(finishMats, spec.foil.finish);

    const w = window as unknown as { __studio?: StudioHandle };
    if (w.__studio) w.__studio.lidPivot = ctx.lidPivot;
    requestRender();
    log('box_scene_rebuilt', { layout: sceneLayout });
  }

  /**
   * Coalesce geometry rebuilds to at most one per animation frame.
   *
   * A dimension slider fires 30-60 store updates a second and each one used to
   * run the whole of rebuild() — ~90 BufferGeometries and ~28 materials — on the
   * event, so the work piled up ahead of the frame it was for. The effect below
   * cancels a pending rebuild before scheduling the next, which is what makes
   * this latest-wins rather than a queue.
   */
  const rebuildRafRef = useRef(0);
  const scheduleRebuild = (): void => {
    if (rebuildRafRef.current) return;
    rebuildRafRef.current = requestAnimationFrame(() => {
      rebuildRafRef.current = 0;
      rebuild();
    });
  };

  /** Book `frames` more draws with the dirty-flag render loop (see DIRTY_FRAMES). */
  const requestRender = (frames: number = DIRTY_FRAMES): void => {
    const ctx = ctxRef.current;
    if (ctx && ctx.renderDebt < frames) ctx.renderDebt = frames;
  };

  // --- renderer setup (once per mount, again on 'Reload view') ---------------
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    // Backdrop preset set; the classic dark studio gradient is the default and
    // doubles as ctx.bgTex (legacy name kept for the restore paths).
    const backdropTexs = makeBackdropTextures();
    const bgTex = backdropTexs.studio;
    scene.background = bgTex;
    const shadowTex = makeShadowTexture();

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.position.set(1.8, 1.3, 1.9);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance',
      });
    } catch (e) {
      // A GPU that is gone for good throws here instead of firing contextlost —
      // most likely on the 'Reload view' retry path. Surface it in the overlay
      // (which keeps offering the retry) instead of letting the throw escape a
      // React effect and take the whole app down.
      log('webgl_init_failed', { error: (e as Error).message });
      setGpuStatus('stalled');
      bgTex.dispose();
      shadowTex.dispose();
      return;
    }
    // ITEM 2e — cap the device pixel ratio at 2. `transmission` makes three render
    // the scene TWICE per frame (the backdrop RT, then the beauty pass), so an
    // uncapped 3x DPR is ~18x the fill of a 1x single pass — and items 3-5 make the
    // plate fragment shader materially more expensive. 2x is past the point of
    // visible return for the fine litho masks.
    //
    // Deliberately NOT a store field with UI: a store-dependent pixel-ratio effect
    // can re-assert the app ratio mid-probe and silently invalidate a
    // zoomForMicroPatterns capture. Note also that onResize() below calls setSize()
    // only, never setPixelRatio — that is load-bearing, because it is what stops a
    // ResizeObserver tick from clobbering the ratio zoomForMicroPatterns saves and
    // restores itself.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // ITEM 2a — tone mapping. The plate rig sums key 2.1 + fill 0.8 + rim 1.3 +
    // frontFill 1.05 + ambient 0.5 ~= 5.7 of irradiance, so with no tone map the
    // highlights HARD-CLIP — and clipping is what destroys hue exactly where a
    // conductor is diagnostic: gold peaks near linear (1.19, 0.97, 0.41) and clamps
    // to (1.0, 0.97, 0.41), where R and G collapse to near-equal and every bright
    // gold highlight reads yellow-white (the single biggest reason the metal used to
    // read as chrome-ish plastic).
    //
    // NeutralToneMapping specifically, NOT ACES or AgX: it returns `color` UNCHANGED
    // below its StartCompression = 0.76 peak, so mid-range differences are preserved
    // EXACTLY. ACES multiplies by exposure/0.6 and reshapes the whole range; AgX
    // log2's the full domain. Both compress mid-tone contrast globally, which attacks
    // every changedFrac floor in the @effects suite — most dangerously the tightest,
    // opposite-tilt >= 0.06.
    //
    // Note the RENDER-TARGET GUARD that makes this safe for the two-plane renderer:
    // three applies tone mapping + the sRGB OETF only when _currentRenderTarget is
    // null. The plates therefore write LINEAR, un-tone-mapped values into the glass
    // slab's transmission backdrop RT (where the inner plane lives) and tone-mapped
    // sRGB values into the default framebuffer. The inner plane is tone-mapped
    // exactly once, by the glass fragment that composites it; the outer plane exactly
    // once, directly. There is no double-tone-map to fear.
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.0;
    // Let the foil/solder maps use the GPU's real anisotropy limit (commonly 16) rather
    // than the 8/4 they were hardcoded to. Set before any build runs, so no map is ever
    // built at the stale default.
    setMetalTextureAnisotropy(renderer.capabilities.getMaxAnisotropy());
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    mount.appendChild(renderer.domElement);

    // Environment map — metals (foil, solder, brass) need one to read as metal
    // (see makeStudioEnv).
    const { pmrem, envTex } = makeStudioEnv(renderer);
    scene.environment = envTex;

    // Global env bounce. The studio env is bright, so keep this ~1; per-finish
    // envMapIntensity in FINISH_PBR does the metal-specific lift.
    scene.environmentIntensity = 1.0;

    // Three-point rig: warm key, cool fill from the opposite side so the metals
    // carry a second highlight, and a bright back/rim light that catches the
    // top edges of the solder beads and hinge. Brightened so direct-lit facets
    // read clearly on top of the env reflections.
    const keyLight = new THREE.DirectionalLight(0xfff2e0, 2.1);
    keyLight.position.set(2, 3, 3);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xbcd0ff, 0.8);
    fillLight.position.set(-2.5, 1.0, -1.5);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 1.3);
    rimLight.position.set(-1.0, 2.5, -3.0);
    scene.add(rimLight);
    // A soft frontal fill from the camera hemisphere. The foil frame strips are
    // near-metal (metalness ~0.8) MeshStandardMaterials whose diffuse floor is
    // the ONLY thing lighting the strips whose mirror reflection misses the
    // studio env. Front-facing strips (normal toward the viewer) get no direct
    // light from the key/fill/rim rig, so without this they read dark grey even
    // for the light silver finish. This frontal fill lights their tint from the
    // camera side; because diffuse = albedo * light, the light finishes (bright,
    // gold) come out far brighter than the dark ones (patina, gunmetal),
    // restoring the finish ordering the verifier found inverted.
    const frontFill = new THREE.DirectionalLight(0xf2f4f8, 1.05);
    frontFill.position.set(0.6, 0.9, 3.2);
    scene.add(frontFill);
    // Ambient floor + a hemisphere pair (fidelity pass): the old flat 0.5
    // ambient lifted every facet identically, which is exactly what makes CG
    // metal look like paint — real ambient light is sky-tinted from above and
    // ground-tinted from below. Splitting the same total irradiance into a
    // 0.28 floor + a 0.45 hemisphere keeps off-axis foil strips out of the
    // black crush (the reason the flat lift existed) while giving every curved
    // metal surface (beads, hinge, ring band) a vertical color gradient to
    // read its shape by. The plate planes ignore scene lights entirely (their
    // shader owns its own light model), so the @effects surfaces see only the
    // small change in the glass slab's 6% diffuse term.
    scene.add(new THREE.AmbientLight(0xffffff, 0.28));
    scene.add(new THREE.HemisphereLight(0xdde6f5, 0x2e2620, 0.45));

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
        shader: makePlateShader(blank, 0),
        shaderBack: makePlateShader(blank, 1),
        glassMat: makeGlassMaterial(),
        textures: [],
        boundFront: null,
        boundBack: null,
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
      groundShadow: null,
      faces,
      raycastTargets: [],
      rebuildDisposables: { geoms: [], mats: [], texs: [] },
      finishMats: null,
      keyLight,
      pmrem,
      envTex,
      diffLut: null,
      blank,
      bgTex,
      backdropTexs,
      shadowTex,
      lidCurrentDeg: 0,
      lidTargetDeg: useStore.getState().lidTargetDeg,
      bindToken: 0,
      camTween: null,
      userDragging: false,
      renderDebt: DIRTY_FRAMES,
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
      requestRender: (frames?: number) => requestRender(frames),
    };
    (window as unknown as { __studio: StudioHandle }).__studio = studio;

    // --- baked diffraction table ------------------------------------------
    // Fetched once per renderer and shared by all twelve plate materials. The
    // physics lives in the backend (app/diffraction.py, unit-tested against
    // closed forms); this is a dumb upload of the result.
    (async () => {
      // BOUNDED RETRY. The renderer mounts as soon as the page loads, which can
      // beat the backend to listening — and this fetch runs exactly once per
      // renderer, so a single lost race used to leave uDiffReady at 0 and the
      // spectral accent silently dead for the whole session. Same shape as the
      // mask-bind retry (FACE_TEXTURE_ATTEMPTS), for the same reason: a
      // transient startup blip must not be indistinguishable from "this design
      // has no accent".
      const attempt = async (): Promise<Response> => {
        let lastErr: unknown = null;
        for (let i = 0; i < DIFF_LUT_ATTEMPTS; i++) {
          try {
            const res = await fetch('/sim/diffraction/lut?duty=0.5&size=1024&u_max_um=10');
            if (res.ok) return res;
            lastErr = new Error(`HTTP ${res.status}`);
          } catch (e) {
            lastErr = e;
          }
          const wait = Math.min(
            DIFF_LUT_RETRY_MAX_MS,
            DIFF_LUT_RETRY_BASE_MS * Math.pow(1.5, i)
          );
          await new Promise((r) => setTimeout(r, wait));
        }
        throw lastErr instanceof Error ? lastErr : new Error('diffraction LUT unavailable');
      };
      try {
        const r = await attempt();
        const payload = (await r.json()) as { size: number; u_max_um: number; rgb: number[] };
        const n = payload.size;
        // RGBA float: RGB float textures are not universally filterable.
        const data = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
          data[i * 4] = payload.rgb[i * 3];
          data[i * 4 + 1] = payload.rgb[i * 3 + 1];
          data[i * 4 + 2] = payload.rgb[i * 3 + 2];
          data[i * 4 + 3] = 1;
        }
        const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat, THREE.FloatType);
        tex.colorSpace = THREE.LinearSRGBColorSpace; // values are already linear
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.generateMipmaps = false;
        tex.needsUpdate = true;
        const c = ctxRef.current;
        if (!c) {
          tex.dispose();
          return;
        }
        c.diffLut = tex;
        for (const fid of FACE_IDS) {
          for (const sh of [c.faces[fid].shader, c.faces[fid].shaderBack]) {
            sh.uniforms.uDiffLut.value = tex;
            sh.uniforms.uDiffUMax.value = payload.u_max_um;
            sh.uniforms.uDiffReady.value = 1.0;
          }
        }
        requestRender();
        log('diffraction_lut_loaded', { size: n, u_max_um: payload.u_max_um });
      } catch (e) {
        // No accent rather than a wrong one — uDiffReady stays 0.
        log('diffraction_lut_failed', { error: (e as Error).message });
      }
    })();

    const canvas = renderer.domElement;
    // GPU context loss. preventDefault() opts into browser restoration; three's
    // own listeners (registered in the WebGLRenderer ctor, so they run before
    // these) skip rendering while lost and re-init GL state on restore. What
    // three CANNOT do is regenerate the PMREM env render target — we do that
    // here, or every metal stays black for the rest of the session.
    let restoreTimer = 0;
    const onContextLost = (e: Event) => {
      e.preventDefault();
      log('webgl_context_lost');
      setGpuStatus('lost');
      window.clearTimeout(restoreTimer);
      restoreTimer = window.setTimeout(() => {
        log('webgl_restore_timeout', { waited_ms: RESTORE_GRACE_MS });
        setGpuStatus('stalled');
      }, RESTORE_GRACE_MS);
    };
    const onContextRestored = () => {
      window.clearTimeout(restoreTimer);
      const c = ctxRef.current;
      if (c) {
        // A fresh context may report a different anisotropy limit; re-assert it before
        // the rebuild below re-derives the metal maps.
        setMetalTextureAnisotropy(c.renderer.capabilities.getMaxAnisotropy());
        c.envTex.dispose();
        c.pmrem.dispose();
        c.diffLut?.dispose();
        const env = makeStudioEnv(c.renderer);
        c.pmrem = env.pmrem;
        c.envTex = env.envTex;
        c.scene.environment = env.envTex;
      }
      // Static geometry is cheap pure assembly math — rebuild it rather than
      // trust GL objects that lived through a context death.
      rebuild();
      log('webgl_context_restored');
      setGpuStatus('ok');
    };
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
      // Inspection mode owns the pointer: a sub-5px drag must not re-select a
      // face out from under the locked camera.
      if (useStore.getState().inspectMode) return;
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
      if (useStore.getState().inspectMode) return;
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
      requestRender();
    };
    onResize();
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    let rafId = 0;
    let lastT = performance.now();
    let lastRenderT = 0;
    const tick = (now: number) => {
      const c = ctxRef.current;
      rafId = requestAnimationFrame(tick);
      if (!c) return;
      const dt = Math.min(0.05, (now - lastT) / 1000);
      lastT = now;
      // Damped lid animation toward the target angle. The integration itself
      // always runs — only the DRAW is dirty-flagged — so the lid still reaches
      // its target while the scene is otherwise idle.
      const target = c.lidTargetDeg;
      let cur = c.lidCurrentDeg;
      if (cur !== target) c.renderDebt = Math.max(c.renderDebt, DIRTY_FRAMES);
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
        c.renderDebt = Math.max(c.renderDebt, DIRTY_FRAMES);
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
      // OrbitControls.update() reports whether the camera actually moved, which
      // covers damping decay, the turntable, an in-flight drag AND a camera
      // placed straight onto the object from outside React (setCameraAzEl in the
      // e2e helpers) — it diffs against its own last known transform.
      if (controls.update() || c.userDragging) {
        c.renderDebt = Math.max(c.renderDebt, DIRTY_FRAMES);
      }
      // Idle heartbeat — the safety valve described at IDLE_RENDER_MS.
      if (c.renderDebt <= 0 && now - lastRenderT >= IDLE_RENDER_MS) c.renderDebt = 1;
      if (c.renderDebt > 0) {
        c.renderDebt -= 1;
        lastRenderT = now;
        renderer.render(scene, camera);
      }
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      window.clearTimeout(restoreTimer);
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
          rt.shaderBack.dispose();
          rt.glassMat.dispose();
        }
        c.envTex.dispose();
        c.pmrem.dispose();
        c.diffLut?.dispose();
        for (const t of Object.values(c.backdropTexs)) t.dispose();
        c.shadowTex.dispose();
      }
      blank.dispose();
      ctxRef.current = null;
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // reinitTick: 'Reload view' tears the whole renderer down and rebuilds it
    // after a context loss the browser never restored. Every effect below that
    // seeds ctx/uniform state carries the same dep so the fresh scene comes back
    // fully configured instead of at material defaults.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reinitTick]);

  // --- rebuild static geometry when spec geometry / layout changes ----------
  // `foil.finish` is deliberately NOT in this key: it cannot move a vertex, only
  // the metalwork materials, so it gets its own restyle effect below. Keeping it
  // here made every click on a finish swatch rebuild the whole scene graph.
  const geomKey = useMemo(
    () =>
      JSON.stringify({
        w: boxSpec.width_um,
        d: boxSpec.depth_um,
        h: boxSpec.height_um,
        glass: boxSpec.glass,
        foil: {
          tape_width_um: boxSpec.foil.tape_width_um,
          safety_um: boxSpec.foil.safety_um,
          bead_um: boxSpec.foil.bead_um,
        },
        hinge: boxSpec.hinge,
        layout,
        // Presentation props are built in rebuild(), so the toggle re-keys it.
        ring: showRing,
      }),
    [boxSpec, layout, showRing]
  );
  useEffect(() => {
    scheduleRebuild();
    return () => {
      // Cancel-then-schedule is what makes a burst of slider updates collapse to
      // ONE rebuild of the latest key instead of a queue of stale ones.
      if (rebuildRafRef.current) {
        cancelAnimationFrame(rebuildRafRef.current);
        rebuildRafRef.current = 0;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geomKey, reinitTick]);

  // --- foil finish: restyle the metalwork in place, no geometry rebuild ------
  // Skipped when there is no build yet (finishMats null) — the rebuild above
  // applies the finish itself, so a fresh scene is never left unstyled.
  const finishKey = boxSpec.foil.finish;
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !ctx.finishMats) return;
    applyFinishMats(ctx.finishMats, finishKey);
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finishKey, reinitTick]);

  // --- bind manifest -> textures + recipe uniforms per face -----------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !boxManifest) return;
    const token = ++ctx.bindToken;
    const loader = new THREE.TextureLoader();
    const retryTimers: number[] = [];

    /**
     * Bind one LITERAL face: the backend published a raster of the FABRICATED
     * CHROME per layer, so the shader synthesizes no gratings at all (see
     * plate.frag::runLiteralLayer). Moiré, switches and shimmer emerge from the
     * geometric separation of the two layers across the paraxial T/n gap, exactly
     * as they do on the procedural path — only the source of the geometry, and
     * where the two layers are combined, changes.
     *
     * ONE PASS, ON THE OUTER PLANE. Both rasters bind to the outer material and
     * the inner pattern plane is hidden. That is not a shortcut: the eye
     * integrates the transmission of the STACK over its resolution cell, so the
     * PRODUCT of the two layers has to be formed at raster resolution and only
     * then averaged. Two separately filtered planes alpha-blended afterwards
     * compute <1-A>·<B> rather than <(1-A)·B>, and at the default camera (~1
     * screen pixel per 99 µm carrier period) that flattened every cross-layer
     * effect on the box to a uniform quarter tone. The inner GLASS ply and the
     * real plane separation are untouched — the gap is still read from the scene
     * graph (see buildPlate's uInnerGapUm sync).
     *
     * A layer whose raster carries NO chrome (loadCoverageTexture returns null:
     * a blank face, or the back of a single-ply one) gets no texture upload and
     * drops out of the composite (B = 0); a face with no FRONT chrome draws
     * nothing at all. `recipe_data.single_ply` / `.blank` say the same thing, but
     * the raster is the primary source — believing the flag over the file would
     * put a face's appearance one manifest field away from its fabrication. The
     * glass slabs are untouched, so a blank face still reads as the bare quartz
     * wall it is.
     */
    const bindLiteralFace = (fid: FaceId, fm: PlateManifest, rt: FaceRT): void => {
      const rd = fm.recipe_data ?? {};
      const frontUrl = fm.files.literal_front;
      const backUrl = fm.files.literal_back;
      const periodUrl = fm.files.period_front;
      if (!frontUrl) {
        // recipe_data claims literal but no raster was published, so this face's
        // geometry does not exist. REFUSE it rather than falling back to reading
        // front.png procedurally: on a blank or halftone face the level codes
        // mean nothing, and that fallback would synthesize gratings and present
        // them as the fabricated part.
        log('face_recipe_data_incomplete', {
          face: fid,
          slug: fm.spec.pattern_slug,
          missing: ['literal_front'],
        });
        rt.shader.visible = false;
        rt.shaderBack.visible = false;
        rt.boundFront = null;
        rt.boundBack = null;
        requestRender();
        return;
      }
      const backKey = backUrl ?? null;
      // Content-addressed under data/plates/<hash>/, so the same pair of URLs IS
      // the same composed plate and every uniform below is a pure function of
      // it — nothing to re-decode. (Same reasoning as the procedural path's
      // `alreadyBound`; 'Retry' clears these first so it stays a real reload.)
      if (rt.boundFront === frontUrl && rt.boundBack === backKey) return;

      const attemptLiteral = (attempt: number): void => {
        Promise.all([
          loadCoverageTexture(frontUrl),
          backUrl ? loadCoverageTexture(backUrl) : Promise.resolve(null),
          periodUrl ? loadCoverageTexture(periodUrl) : Promise.resolve(null),
        ])
          .then(([front, back, period]) => {
            if (ctxRef.current !== ctx || ctx.bindToken !== token) {
              for (const t of [front, back, period]) t?.dispose();
              return;
            }
            for (const old of rt.textures) old.dispose();
            rt.textures = [front, back, period].filter(
              (t): t is THREE.DataTexture => t !== null
            );
            const applyShared = (u: Record<string, THREE.IUniform>) => {
              (u.uExtentUm.value as THREE.Vector2).set(fm.extent_um[0], fm.extent_um[1]);
              u.uThicknessUm.value = fm.substrate.thickness_um;
              u.uN.value = fm.substrate.n;
              u.uRecipe.value = RECIPE_IDS.foliage_moire;
              // THE switch: sample the fabricated raster, draw no gratings.
              u.uLiteral.value = 1.0;
            };
            applyShared(rt.shader.uniforms);
            applyShared(rt.shaderBack.uniforms);
            // SINGLE-PASS COMPOSITE. The OUTER plane material carries BOTH
            // rasters — uFront the outer chrome, uBackCoverage the inner — and
            // plate.frag forms their product per subsample before averaging over
            // the pixel footprint. Two independently mip-filtered planes could not:
            // the eye integrates the transmission of the STACK, and
            // <(1-A)·B> != <1-A>·<B> wherever the layers correlate across a
            // footprint, which is every cross-layer effect on the box. So the inner
            // pattern plane draws nothing (below); the inner glass ply is untouched.
            rt.shader.uniforms.uFront.value = front ?? ctx.blank;
            rt.shader.uniforms.uBack.value = back ?? ctx.blank;
            rt.shader.uniforms.uBackCoverage.value = back ?? ctx.blank;
            rt.shader.uniforms.uBackCoverageReady.value = back ? 1.0 : 0.0;
            // Subsample sizing: the raster's own texel grid. Both layers of a face
            // are published at the same resolution; front is the one that always
            // exists (a missing front already refused the face above).
            (rt.shader.uniforms.uCoverageSizePx.value as THREE.Vector2).set(
              front?.image.width ?? 2048,
              front?.image.height ?? 2048
            );
            // The inner material keeps its own bindings — it is inert here, but a
            // dangling sampler and a silently-unbound face are both worse than a
            // bound invisible one, and `face_layer_empty` below stays diagnosable.
            rt.shaderBack.uniforms.uFront.value = back ?? ctx.blank;
            rt.shaderBack.uniforms.uBack.value = front ?? ctx.blank;
            rt.shaderBack.uniforms.uBackCoverage.value = ctx.blank;
            rt.shaderBack.uniforms.uBackCoverageReady.value = 0.0;
            // Sub-grating period map: the composite plane only (the backend
            // publishes period_front alone). On single-ply faces this map also
            // carries the garland's leaf diffraction gratings, not just the
            // photo's colour-zone periods. The two now speak one vocabulary:
            // the leaves are written at one PERIOD per motif family off the
            // photographs' own 4.15-6.02 um colour ladder, so a leaf and a
            // colour zone beside it flash the same hue at the same tilt. The
            // map carries the ladder's MEAN over every leaf texel
            // (recipe_data.single_ply_leaf_period_um) rather than the family
            // split, and plate.frag's diffractionSheen() lights every leaf at
            // the same fixed angle — the one non-literal term here.
            rt.shader.uniforms.uPeriodMap.value = period ?? ctx.blank;
            rt.shader.uniforms.uPeriodReady.value = period ? 1.0 : 0.0;
            rt.shaderBack.uniforms.uPeriodMap.value = ctx.blank;
            rt.shaderBack.uniforms.uPeriodReady.value = 0.0;
            // The composite plane draws when there is any chrome at all; the inner
            // pattern plane NEVER draws on a literal face — drawing the back layer
            // again on its own plane would paint it twice.
            rt.shader.visible = front !== null;
            rt.shaderBack.visible = false;
            for (const [layer, tex] of [
              ['front', front],
              ['back', back],
            ] as const) {
              if (tex === null) {
                log('face_layer_empty', {
                  face: fid,
                  slug: fm.spec.pattern_slug,
                  layer,
                  single_ply: rd.single_ply === true,
                  blank: rd.blank === true,
                });
              }
            }
            rt.boundFront = frontUrl;
            rt.boundBack = backKey;
            requestRender();
            log('face_texture_bound', {
              face: fid,
              slug: fm.spec.pattern_slug,
              recipe: 'foliage_moire',
              // The @effects suite reads this event as proof a new mask landed;
              // `literal` says WHICH geometry source it was.
              literal: true,
            });
            clearFaceFailed(fid);
          })
          .catch((e) => {
            if (ctxRef.current !== ctx || ctx.bindToken !== token) return;
            const message = (e as Error).message;
            if (attempt + 1 < FACE_TEXTURE_ATTEMPTS) {
              log('face_texture_retry', {
                face: fid,
                slug: fm.spec.pattern_slug,
                attempt: attempt + 1,
                error: message,
              });
              retryTimers.push(
                window.setTimeout(() => {
                  if (ctxRef.current === ctx && ctx.bindToken === token) {
                    attemptLiteral(attempt + 1);
                  }
                }, FACE_TEXTURE_RETRY_MS)
              );
              return;
            }
            log('face_texture_failed', {
              face: fid,
              slug: fm.spec.pattern_slug,
              attempts: attempt + 1,
              error: message,
            });
            markFaceFailed(fid);
          });
      };
      attemptLiteral(0);
    };

    for (const fid of FACE_IDS) {
      const fm = boxManifest.faces[fid];
      if (!fm) continue;
      const rt = ctx.faces[fid];
      // A composed box plate is ALWAYS foliage_moire — plates.py forces it, and it
      // is the only recipe the two-plane renderer implements. Anything else (a
      // manifest cached before the recipe was forced, an unknown string, a missing
      // field) is REFUSED, not fallen back on: the old fallback bound
      // moire_interactive to BOTH plane materials, which is the banned single-plane
      // model — in-shader parallax_offset instead of the real T/n gap, opaque (so
      // the inner plane never shows) and with uFront/uBack swapped on the inner
      // plane. It also mislabelled itself in face_texture_bound as if the manifest
      // had asked for it. Leave the planes unbound and invisible: a bare glass face
      // is a visible, greppable failure; a plausible-looking wrong physics is not.
      if (fm.render_recipe !== 'foliage_moire') {
        log('face_recipe_unsupported', {
          face: fid,
          slug: fm.spec.pattern_slug,
          recipe: fm.render_recipe ?? null,
          expected: 'foliage_moire',
        });
        rt.shader.visible = false;
        rt.shaderBack.visible = false;
        // Nothing honest is bound any more: forget the URLs so a later manifest
        // that DOES declare foliage_moire binds this face instead of skipping it
        // as already-current.
        rt.boundFront = null;
        rt.boundBack = null;
        requestRender();
        continue;
      }
      // Recipe extras (slimmed recipe_data still carries the scalar knobs — only
      // frame_scene is stripped).
      const rd = fm.recipe_data ?? {};
      // LITERAL faces take the other material path entirely: their geometry
      // arrives as a raster of the fabricated chrome, so none of the procedural
      // scalar knobs below (bucket encoding, preview periods, barrier phase)
      // apply — there is no lattice to reconstruct.
      if (rd.literal === true) {
        bindLiteralFace(fid, fm, rt);
        continue;
      }
      // recipe_data keys whose absence changes the GEOMETRY rather than a shade:
      // the art-box rect gates the barrier comb (a (0,0) fallback stretches it to
      // the whole face), the preview periods carry the grating pitch, and the
      // barrier phase carries the switch registration.
      // Degrade loudly — the fallbacks below still keep the face renderable.
      const missing = [
        'water_art_half_uv',
        'preview_carrier_period_um',
        'preview_slit_period_um',
        'fab_center_period_um',
        ...(rd.switch_interlace ? ['switch_barrier_phase_um'] : []),
      ].filter((k) => rd[k] == null);
      if (missing.length > 0) {
        log('face_recipe_data_incomplete', {
          face: fid,
          slug: fm.spec.pattern_slug,
          missing,
        });
      }
      const frontUrl = fm.files.front_png;
      const backUrl = fm.files.back_png;
      // A manifest change usually moves ONE face, but this pass used to re-fetch,
      // re-decode and re-upload all twelve mask PNGs. Mask paths are content
      // addressed (data/plates/<hash>/...), so an unchanged pair IS the same
      // composed plate: reuse the bound textures and only re-apply the recipe
      // uniforms. Nothing (re)binds, so no face_texture_bound is logged — the
      // e2e suite reads that event as proof a NEW mask landed. 'Retry' clears
      // these URLs first (see retryMasks) so it stays a real reload.
      const alreadyBound =
        rt.boundFront === frontUrl && rt.boundBack === backUrl && rt.textures.length === 2;
      // Bounded retry: a transient blip — a network hiccup, or the plate cache
      // evicted between the manifest write and this fetch — used to leave the
      // face on the 1x1 blank (or the PREVIOUS design's masks) forever, with the
      // only trace a console.debug line. Retry once, then fail loudly:
      // face_texture_failed plus a badge over the scene, because one dark wall is
      // otherwise indistinguishable from the user's own design.
      const attemptBind = (attempt: number): void => {
        const loads = alreadyBound
          ? [Promise.resolve(rt.textures[0]), Promise.resolve(rt.textures[1])]
          : [loader.loadAsync(frontUrl), loader.loadAsync(backUrl)];
        Promise.all(loads)
          .then((texs) => {
            if (ctxRef.current !== ctx || ctx.bindToken !== token) {
              if (!alreadyBound) for (const tex of texs) tex.dispose();
              return;
            }
            const [front, back] = texs;
            if (!alreadyBound) {
              for (const tex of texs) {
                tex.colorSpace = THREE.LinearSRGBColorSpace;
                tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
                tex.generateMipmaps = false;
                tex.needsUpdate = true;
              }
              // The front/back masks (texs[0], texs[1]) are LABEL/CODE textures, not
              // continuous tone: runFoliageMoireLayer decodes r through hard region
              // thresholds (FRAME_MIN/RAINBOW_MIN/ART_MIN) and a quantized angle-bucket
              // floor(). Bilinear interpolation across any boundary manufactures
              // intermediate codes that never existed in the mask, decoding to the
              // wrong bucket / wrong region in a thin edge band — which reads as a
              // "traced" outline on every motif and a black stripe wherever the ramp
              // dips below the gold threshold. Sample them NEAREST. (anisotropy is a
              // no-op without a mipmap chain, so it is intentionally dropped here.)
              for (const mask of [front, back]) {
                mask.magFilter = THREE.NearestFilter;
                mask.minFilter = THREE.NearestFilter;
              }
              for (const old of rt.textures) old.dispose();
              rt.textures = texs;
            }
            // Shared recipe uniforms are written to BOTH plane materials; the ONLY
            // per-plane differences are the mask bound to uFront (front vs back
            // PNG) and the fixed uLayer set at material creation.
            const applyShared = (u: Record<string, THREE.IUniform>) => {
              // (width, height) — UV u spans the width, v the height; wall plates
              // are non-square so the physical grating axes need both.
              (u.uExtentUm.value as THREE.Vector2).set(fm.extent_um[0], fm.extent_um[1]);
              u.uThicknessUm.value = fm.substrate.thickness_um;
              u.uN.value = fm.substrate.n;
              u.uRecipe.value = RECIPE_IDS.foliage_moire;
              // Procedural path: level-coded masks, gratings synthesized below.
              // Explicit (not merely defaulted) so a face that was literal on the
              // previous manifest and is not on this one flips back.
              u.uLiteral.value = 0.0;
              u.uPeriodReady.value = 0.0;
              u.uPeriodMap.value = ctx.blank;
              // Same reason: the single-pass composite belongs to the literal
              // path only. The procedural path keeps the TWO-PLANE construction,
              // where each plane draws its own layer and the beat emerges from
              // perspective across the real gap.
              u.uBackCoverageReady.value = 0.0;
              u.uBackCoverage.value = ctx.blank;
              // Two-plane geometric renderer. The FRONT (outer) plane draws the
              // foliage louvre (slit period/angle) + the centerpiece's front
              // geometry; the BACK (inner) plane draws the uniform carrier + the
              // centerpiece's back geometry. The leaf
              // moiré and the switch EMERGE from the two real planes — NOT from a
              // single-plane beat formula. Preview periods must resolve at
              // >=2-3 px/period at default zoom or the real planes alias (the old
              // 70 µm carrier was sub-pixel — see the two-plane rig matrix).
              // NOTE: uCarrierPeriodUm / uSlitPeriodUm are bound ONLY from the
              // preview_* / fab_*-derived values below — the manifest's
              // advertised true pitch never drives the shader directly.
              u.uCarrierAngle.value = ((Number(rd.carrier_angle_deg ?? 0) || 0) * Math.PI) / 180;
              u.uSlitAngle.value = ((Number(rd.slit_axis_deg ?? 3) || 3) * Math.PI) / 180;
              u.uGratingDuty.value = Number(rd.grating_duty ?? 0.5) || 0.5;
              u.uFrameBucket0.value = (Number(rd.frame_bucket0 ?? 96) || 96) / 255;
              u.uFrameBucketStep.value = (Number(rd.frame_bucket_step ?? 14) || 14) / 255;
              u.uFrameBucketCount.value = Number(rd.frame_bucket_count ?? 6) || 6;
              u.uFrameAngleSpan.value =
                ((Number(rd.frame_angle_span_deg ?? 3.5) || 3.5) * Math.PI) / 180;
              // --- GRATING PITCH preview periods (Tasks 1 + 2) ------------------
              // The frame back carrier + leaf louvre are drawn at the USER-TUNED
              // grating pitch, MAGNIFIED for on-screen resolvability (see
              // plates.PREVIEW_PITCH_MAGNIFY): the raw fab pitch can be the 4 µm
              // litho floor, which is deeply sub-pixel on the two real planes even
              // at Pattern Scale 4× and would collapse to flat gold. The preview
              // period is proportional to the real pitch, so finer pitch reads as
              // finer, livelier fringes; the manifest's carrier_period_um / fab_*
              // stay the TRUE pitch (advertised + baked). uPatternScale multiplies
              // on top. Fallbacks read the true pitch × the magnify factor so a
              // stale manifest (no preview_* fields) still resolves. The
              // centerpiece 60 µm switch/comb below is NOT part of this family: it
              // is drawn at exact fab pitch and is NOT scaled (see uPatternScale).
              u.uCarrierPeriodUm.value =
                Number(rd.preview_carrier_period_um ??
                  (Number(rd.fab_back_period_um ?? 22.0) || 22.0) * 5.0) || 110.0;
              u.uSlitPeriodUm.value =
                Number(rd.preview_slit_period_um ??
                  (Number(rd.fab_front_period_um ?? 22.0 * 1.09) || 22.0 * 1.09) * 5.0) ||
                110.0 * 1.09;
              // Centerpiece barrier pitch: the fab 60 µm value
              // → ~5° crossing at the T/n air gap. On a barrier-interlace face the
              // canonical field is switch_interlace_period_um (comb AND lanes share
              // that ONE period by construction); fab_center_period_um is the same
              // number for every other face.
              const centerPeriodRd = rd.switch_interlace
                ? (rd.switch_interlace_period_um ?? rd.fab_center_period_um)
                : rd.fab_center_period_um;
              u.uCenterPeriodUm.value = Number(centerPeriodRd ?? 60.0) || 60.0;
              u.uSwitchAxis.value = ((Number(rd.switch_axis_deg ?? 0) || 0) * Math.PI) / 180;
              // Barrier-interlace faces (the globe-duo-phase exemplar) plus the
              // SOLVED registration phase, so the preview comb + lanes ride the same
              // lattice the fab bake and the generators do instead of the old
              // hardcoded face-edge assumption.
              u.uSwitchInterlace.value = rd.switch_interlace ? 1.0 : 0.0;
              u.uSwitchBarrierPhaseUm.value = Number(rd.switch_barrier_phase_um ?? 0) || 0;
              // Live Pattern Scale (Task 1b) — the store may have changed it
              // before this manifest bound; keep the freshly-bound uniforms in sync.
              u.uPatternScale.value = useStore.getState().patternScale;
              // The centrepiece ART BOX in uv — the window the barrier interlace
              // is written inside. `water_art_*` is the backend's legacy key name
              // for it (plates.py); the water scanimation that named it is gone.
              {
                const half = (rd.water_art_half_uv ?? [0, 0]) as number[];
                const ctr = (rd.water_art_center_uv ?? [0.5, 0.5]) as number[];
                (u.uArtBoxHalfUv.value as THREE.Vector2).set(
                  Number(half[0]) || 0,
                  Number(half[1]) || 0
                );
                (u.uArtBoxCenterUv.value as THREE.Vector2).set(
                  Number(ctr[0]) || 0.5,
                  Number(ctr[1]) || 0.5
                );
              }
              // The FABRICATED accent grating's orientation + zeroth-order share,
              // straight from recipe_data (which reads the same constants the mask
              // is baked with). Fallbacks are the shipping values, so a manifest
              // cached before these keys existed still sheens at the right angle.
              // The accent's PITCH is per-pixel now and rides the period map on the
              // literal path; the procedural path draws no accent at all.
              u.uRainbowAngleRad.value =
                ((Number(rd.rainbow_angle_deg ?? 45) || 45) * Math.PI) / 180;
              u.uRainbowZeroOrder.value = Number(rd.rainbow_zero_order ?? 0.25) || 0.25;
            };
            applyShared(rt.shader.uniforms);
            applyShared(rt.shaderBack.uniforms);
            // Per-plane masks: OUTER plane samples the FRONT mask, INNER the BACK.
            // uBack is kept bound (legacy single-plane recipes read it); the
            // foliage_moire path ignores it and reads only uFront (this layer).
            rt.shader.uniforms.uFront.value = front;
            rt.shader.uniforms.uBack.value = back;
            rt.shaderBack.uniforms.uFront.value = back;
            rt.shaderBack.uniforms.uBack.value = front;
            // Bound and honest — undo any earlier refusal on this face.
            rt.shader.visible = true;
            rt.shaderBack.visible = true;
            rt.boundFront = frontUrl;
            rt.boundBack = backUrl;
            requestRender();
            if (alreadyBound) return;
            log('face_texture_bound', {
              face: fid,
              slug: fm.spec.pattern_slug,
              recipe: 'foliage_moire',
            });
            clearFaceFailed(fid);
          })
          .catch((e) => {
            if (ctxRef.current !== ctx || ctx.bindToken !== token) return;
            const message = (e as Error).message;
            if (attempt + 1 < FACE_TEXTURE_ATTEMPTS) {
              log('face_texture_retry', {
                face: fid,
                slug: fm.spec.pattern_slug,
                attempt: attempt + 1,
                error: message,
              });
              retryTimers.push(
                window.setTimeout(() => {
                  if (ctxRef.current === ctx && ctx.bindToken === token) attemptBind(attempt + 1);
                }, FACE_TEXTURE_RETRY_MS)
              );
              return;
            }
            log('face_texture_failed', {
              face: fid,
              slug: fm.spec.pattern_slug,
              attempts: attempt + 1,
              error: message,
            });
            markFaceFailed(fid);
          });
      };
      attemptBind(0);
    }
    return () => {
      for (const h of retryTimers) window.clearTimeout(h);
    };
    // rebindTick: the 'Retry' button on the mask-failure badge. reinitTick: a
    // rebuilt renderer needs every face bound again from scratch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boxManifest, rebindTick, reinitTick]);

  // --- auto-rotate (slow turntable) ------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    ctx.controls.autoRotate = autoRotate;
    const w = window as unknown as { __studio?: StudioHandle };
    if (w.__studio) w.__studio.autoRotate = autoRotate;
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRotate, reinitTick]);

  // --- selected face highlight + gentle camera tween -------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    for (const fid of FACE_IDS) {
      const g = ctx.faces[fid].glassMat;
      const sel = fid === selectedFaceId;
      g.emissive.setHex(sel ? 0x1d4a7a : 0x000000);
      g.emissiveIntensity = sel ? 0.5 : 0.0;
    }
    requestRender();
    // Tween the camera azimuth toward the selected wall (~400 ms ease).
    // Skipped on mount (and on a renderer reinit, which resets the flag — the
    // highlight must be re-applied to the fresh materials but a surprise camera
    // move on 'Reload view' is not wanted), for top/bottom (no natural
    // azimuth), in flat layout, and while the user is orbit-dragging — never
    // fight OrbitControls.
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFaceId, reinitTick]);

  // --- lid target ------------------------------------------------------------
  // No requestRender here: the loop books its own debt for every frame the lid
  // is still travelling toward the target.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (ctx) ctx.lidTargetDeg = lidTargetDeg;
  }, [lidTargetDeg, reinitTick]);

  // --- pattern scale (Task 1b) -----------------------------------------------
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    for (const fid of FACE_IDS) {
      ctx.faces[fid].shader.uniforms.uPatternScale.value = patternScale;
      ctx.faces[fid].shaderBack.uniforms.uPatternScale.value = patternScale;
    }
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patternScale, reinitTick]);

  // --- litho metal (renderer-audit item 5) ------------------------------------
  const metalKey = (boxSpec.metal ?? 'gold') as keyof typeof METAL_LOOKS;
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const look = METAL_LOOKS[metalKey] ?? METAL_LOOKS.gold;
    for (const fid of FACE_IDS) {
      for (const sh of [ctx.faces[fid].shader, ctx.faces[fid].shaderBack]) {
        (sh.uniforms.uMetalAlbedo.value as THREE.Color).setRGB(
          look.albedo[0], look.albedo[1], look.albedo[2], THREE.LinearSRGBColorSpace);
        (sh.uniforms.uMetalAlbedoBack.value as THREE.Color).setRGB(
          look.back[0], look.back[1], look.back[2], THREE.LinearSRGBColorSpace);
        (sh.uniforms.uMetalF0.value as THREE.Color).setRGB(
          look.f0[0], look.f0[1], look.f0[2], THREE.LinearSRGBColorSpace);
        sh.uniforms.uMetalBody.value = look.body;
        sh.uniforms.uMetalSpec.value = look.spec;
        sh.uniforms.uMetalGloss.value = look.gloss;
        sh.uniforms.uMetalGrazing.value = look.grazing;
        sh.uniforms.uMetalSheen.value = look.sheen;
        sh.uniforms.uMetalEnv.value = look.env;
      }
    }
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metalKey, reinitTick]);

  // --- backdrop preset / light-table field -------------------------------------
  // Backlight illumination overrides the user backdrop with the bright
  // light-table field (mask-inspection view) and hides the contact-shadow blob
  // (a shadow ON a light box reads as a smudge).
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    ctx.scene.background =
      illumination === 'backlight' ? ctx.backdropTexs.lighttable : ctx.backdropTexs[backdrop];
    if (ctx.groundShadow) ctx.groundShadow.visible = illumination !== 'backlight';
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backdrop, illumination, reinitTick]);

  // --- face tilt-inspection (renderer-audit item 4) ----------------------------
  // Head-on lock on the selected face; dragging rocks the view +/-INSPECT_MAX_DEG
  // about the face axes while the HUD reads out tilt, the paraxial back-layer
  // shift it produces, and where this face's effects peak. Pure camera work over
  // the honest geometry -- nothing in the shading path changes.
  const applyInspectPose = (): void => {
    const ctx = ctxRef.current;
    const rig = inspectRig.current;
    if (!ctx || !rig) return;
    const tx = THREE.MathUtils.degToRad(rig.tilt.x);
    const ty = THREE.MathUtils.degToRad(rig.tilt.y);
    const dir = rig.normal.clone().applyAxisAngle(rig.up, tx).applyAxisAngle(rig.right, -ty);
    ctx.camera.position.copy(rig.center).addScaledVector(dir, rig.dist);
    ctx.camera.up.copy(rig.up);
    ctx.camera.lookAt(rig.center);
    requestRender();
  };

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const restore = (): void => {
      const rig = inspectRig.current;
      if (!rig) return;
      ctx.camera.position.copy(rig.savedPos);
      ctx.camera.up.set(0, 1, 0);
      ctx.controls.target.copy(rig.savedTarget);
      ctx.controls.enabled = true;
      inspectRig.current = null;
      requestRender();
    };
    if (!inspectMode || layout !== 'assembled') {
      restore();
      return;
    }
    const spec = useStore.getState().boxSpec;
    const p = platePlacements(spec).find((x) => x.face === selectedFaceId);
    if (!p) return;
    // The lid swings the top face away -- close it before locking head-on.
    if (selectedFaceId === 'top' && useStore.getState().lidTargetDeg !== 0) {
      useStore.getState().setLidTargetDeg(0);
    }
    const sc = ctx.root.scale.x;
    const mmv = (um: number): number => (um / 1000) * sc;
    const center = new THREE.Vector3(mmv(p.center_um[0]), mmv(p.center_um[1]), mmv(p.center_um[2]));
    const euler = new THREE.Euler(p.rotation[0], p.rotation[1], p.rotation[2], 'XYZ');
    const right = new THREE.Vector3(1, 0, 0).applyEuler(euler);
    const up = new THREE.Vector3(0, 1, 0).applyEuler(euler);
    const normal = new THREE.Vector3(p.outward[0], p.outward[1], p.outward[2]);
    const span = (Math.max(p.width_um, p.height_um) / 1000) * sc;
    const dist = Math.max(0.9, span * 1.5);
    inspectRig.current = {
      center,
      normal,
      right,
      up,
      dist,
      savedPos: ctx.camera.position.clone(),
      savedTarget: ctx.controls.target.clone(),
      tilt: { x: 0, y: 0 },
    };
    setInspectTilt({ x: 0, y: 0 });
    ctx.controls.enabled = false;
    ctx.camTween = null;
    applyInspectPose();
    log('inspect_mode', { face: selectedFaceId, on: true });

    const canvas = ctx.renderer.domElement;
    let dragging = false;
    let sx = 0;
    let sy = 0;
    let bx = 0;
    let by = 0;
    const down = (e: PointerEvent): void => {
      dragging = true;
      sx = e.clientX;
      sy = e.clientY;
      const rig = inspectRig.current;
      if (rig) {
        bx = rig.tilt.x;
        by = rig.tilt.y;
      }
    };
    const move = (e: PointerEvent): void => {
      if (!dragging) return;
      const rig = inspectRig.current;
      if (!rig) return;
      rig.tilt.x = THREE.MathUtils.clamp(
        bx + (e.clientX - sx) * INSPECT_DEG_PER_PX, -INSPECT_MAX_DEG, INSPECT_MAX_DEG);
      rig.tilt.y = THREE.MathUtils.clamp(
        by + (e.clientY - sy) * INSPECT_DEG_PER_PX, -INSPECT_MAX_DEG, INSPECT_MAX_DEG);
      setInspectTilt({ x: rig.tilt.x, y: rig.tilt.y });
      applyInspectPose();
    };
    const upHandler = (): void => {
      dragging = false;
    };
    const keyHandler = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') useStore.getState().setInspectMode(false);
    };
    canvas.addEventListener('pointerdown', down);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', upHandler);
    window.addEventListener('keydown', keyHandler);
    return () => {
      canvas.removeEventListener('pointerdown', down);
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', upHandler);
      window.removeEventListener('keydown', keyHandler);
      restore();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspectMode, selectedFaceId, layout, geomKey, reinitTick]);

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
      for (const s of [ctx.faces[fid].shader, ctx.faces[fid].shaderBack]) {
        s.uniforms.uIllumination.value = illumId;
        s.uniforms.uLaserColor.value.setHex(laserRgb[laserColor]);
      }
    }
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [illumination, laserColor, reinitTick]);

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
      ctx.faces[fid].shaderBack.uniforms.uLightWorld.value.set(x, y, z);
    }
    ctx.keyLight.position.set(x, y, z);
    // ITEM 7 — the ground shadow follows the key light. Null in flat layout, which has
    // no ground plane.
    if (ctx.groundShadow) {
      const casterH = Number(ctx.groundShadow.userData.casterH) || 1;
      const reach = Number(ctx.groundShadow.userData.reach) || 1;
      poseGroundShadow(ctx.groundShadow, lightAz, lightEl, casterH, reach);
    }
    requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lightAz, lightEl, reinitTick]);

  // Inspection HUD numbers: paraxial back-layer shift for the current tilt
  // (Snell, matching sim2d/_axis_shift_um), plus where THIS face's effects peak
  // — derived from the manifest's fab periods and the spec's real glass, so the
  // HUD always states the numbers of the box being edited, never defaults.
  const inspectHud = useMemo(() => {
    if (!inspectMode) return null;
    const t = boxSpec.glass.thickness_um;
    const n = boxSpec.glass.n > 1 ? boxSpec.glass.n : 1.46;
    const shiftUm = (deg: number): number => {
      const sinSub = Math.sin((Math.abs(deg) * Math.PI) / 180) / n;
      const cosSub = Math.sqrt(Math.max(0, 1 - sinSub * sinSub));
      return (t * sinSub) / Math.max(0.05, cosSub);
    };
    const tiltForShift = (sUm: number): number | null => {
      const v = n * Math.sin(Math.atan2(sUm, t));
      if (v >= 1) return null;
      return (Math.asin(v) * 180) / Math.PI;
    };
    const rd = (boxManifest?.faces?.[selectedFaceId]?.recipe_data ?? {}) as Record<string, unknown>;
    const num = (k: string): number => Number(rd[k] ?? 0) || 0;
    const lines: string[] = [];
    if (rd.switch_interlace) {
      const pp = num('switch_interlace_period_um') || num('fab_center_period_um');
      const swap = pp > 0 ? tiltForShift(pp / 4) : null;
      const alias = pp > 0 ? tiltForShift(pp) : null;
      if (swap != null) lines.push(`A↔B swap peaks at ±${swap.toFixed(1)}°`);
      if (alias != null) lines.push(`replays every ~${alias.toFixed(1)}°`);
    } else {
      const pp = num('carrier_period_um');
      const peak = pp > 0 ? tiltForShift(pp / 2) : null;
      if (peak != null) lines.push(`carrier reveal peaks at ±${peak.toFixed(1)}°`);
    }
    return { shiftUm, lines };
  }, [inspectMode, boxSpec, boxManifest, selectedFaceId]);

  return (
    // The mount div keeps its own box so ResizeObserver still measures the
    // canvas area; the status layers are absolutely positioned siblings, never
    // React children of the mount (three appends the canvas there imperatively).
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 0, minWidth: 0 }}>
      <div
        ref={mountRef}
        data-testid="box-scene"
        style={{ width: '100%', height: '100%', minHeight: 0, minWidth: 0 }}
      />
      {gpuStatus !== 'ok' && (
        <div
          data-testid="gpu-overlay"
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 10,
            background: 'rgba(6,7,11,0.86)',
            color: KIT.text,
            fontSize: 13,
            textAlign: 'center',
            padding: 16,
            // The canvas is frozen anyway; let orbit drags through so the view
            // is not hostage to the overlay if a restore is slow.
            pointerEvents: 'none',
          }}
        >
          <div>
            {gpuStatus === 'lost'
              ? '3D preview lost the GPU — restoring…'
              : '3D preview lost the GPU — it did not come back.'}
          </div>
          {gpuStatus === 'stalled' && (
            <>
              <div style={{ opacity: 0.7, fontSize: 12, maxWidth: 320 }}>
                Reloading rebuilds the renderer; your design and the cut list are untouched.
              </div>
              <div style={{ pointerEvents: 'auto' }}>
                <Button onClick={reloadView} testId="gpu-reload-view">
                  Reload view
                </Button>
              </div>
            </>
          )}
        </div>
      )}
      {gpuStatus === 'ok' && inspectMode && inspectHud && (
        <div
          data-testid="inspect-hud"
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 3,
            padding: '8px 12px',
            borderRadius: 6,
            border: `1px solid ${KIT.border}`,
            background: 'rgba(12,15,21,0.88)',
            color: KIT.text,
            fontSize: 12,
            pointerEvents: 'none',
            textAlign: 'center',
          }}
        >
          <div style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, fontSize: 11 }}>
            Inspecting {selectedFaceId}
          </div>
          <div style={{ fontVariantNumeric: 'tabular-nums' }}>
            tilt {inspectTilt.x.toFixed(1)}° / {inspectTilt.y.toFixed(1)}°
            {' · '}
            back-layer shift {inspectHud.shiftUm(inspectTilt.x).toFixed(1)} /{' '}
            {inspectHud.shiftUm(inspectTilt.y).toFixed(1)} µm
          </div>
          {inspectHud.lines.map((l) => (
            <div key={l} style={{ opacity: 0.75 }}>{l}</div>
          ))}
          <div style={{ opacity: 0.5, fontSize: 11 }}>
            drag rocks ±{INSPECT_MAX_DEG}° · Esc exits
          </div>
        </div>
      )}
      {gpuStatus === 'ok' && failedFaces.length > 0 && (
        <div
          data-testid="face-texture-warning"
          style={{
            position: 'absolute',
            left: 10,
            bottom: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 8px',
            borderRadius: 4,
            border: `1px solid ${KIT.error}`,
            background: 'rgba(15,18,24,0.92)',
            color: KIT.text,
            fontSize: 11,
          }}
        >
          <span>
            Mask load failed: {failedFaces.join(', ')} — these faces show blank or previous
            patterns.
          </span>
          <Button onClick={retryMasks} testId="face-texture-retry">
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}
