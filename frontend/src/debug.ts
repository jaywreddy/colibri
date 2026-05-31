/**
 * window.__debug — stable surface for headless inspection of the live app.
 *
 * Designed for an external Playwright client (Claude_Preview MCP, the e2e
 * harness, or @playwright/mcp) that wants to drive a specific scene and
 * read back rendered state in one round-trip per call. Each method is a
 * thin wrapper over the same primitives the e2e helpers use:
 * `window.__three` for the renderer/scene/material, `window.__store` for
 * the Zustand state, `window.__log` for structured events.
 *
 * Self-registers on import. See tools/preview_inspect.md for the workflow.
 */
import * as THREE from 'three';
import { useStore, type Illumination } from './store';
import { getDefault } from './api';
import { log, type LogEvent } from './logger';

export type LaserColor = 'red' | 'green' | 'blue';

/**
 * Bundle of every URL-driveable scene knob. `pattern` triggers a select +
 * texture-bind await; the rest are synchronous Zustand pushes that take
 * effect in the next RAF.
 */
export type ScenePreset = {
  pattern?: string;
  illumination?: Illumination;
  laserColor?: LaserColor;
  cameraAzEl?: [number, number];
  light?: [number, number];
  params?: Record<string, unknown>;
};

type ThreeCtx = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  material: THREE.ShaderMaterial;
  controls: {
    target?: { x: number; y: number; z: number };
    update?: () => void;
    getAzimuthalAngle?: () => number;
    getPolarAngle?: () => number;
  };
};

function getThree(): ThreeCtx | null {
  return (window as unknown as { __three?: ThreeCtx }).__three ?? null;
}

function getBuffer(): LogEvent[] {
  return ((window as unknown as { __log?: LogEvent[] }).__log ?? []);
}

/**
 * Yield long enough for React/Zustand to flush a state push into uniforms.
 * Drains the microtask queue (so React 18 effects run) plus one macrotask
 * (so any deferred work also completes). Avoids requestAnimationFrame
 * because background tabs throttle it (often to 0 Hz in Claude_Preview).
 * Callers that need a fresh framebuffer use quadrantBrightness or
 * captureCanvas — both call renderer.render() directly.
 */
async function settleFrames(): Promise<void> {
  await Promise.resolve();
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function waitForThreeContext(timeoutMs = 8_000): Promise<ThreeCtx> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const t = getThree();
    if (t?.material) return t;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error('window.__debug: __three not initialized within timeout');
}

/**
 * Poll `window.__log` for a matching event pushed AFTER this call. Searches
 * back-to-front and stops when timestamps cross the call time, so the cost
 * is bounded by recently-pushed events, not the whole 500-event buffer.
 */
async function waitForLog(
  type: string,
  predicate?: (e: LogEvent) => boolean,
  timeoutMs = 10_000
): Promise<LogEvent> {
  const startTime = performance.now();
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const buf = getBuffer();
    for (let i = buf.length - 1; i >= 0; i--) {
      const e = buf[i];
      if (e.t < startTime) break;
      if (e.type === type && (!predicate || predicate(e))) return e;
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  const tail = JSON.stringify(getBuffer().slice(-20), null, 2);
  throw new Error(
    `window.__debug.waitForLog: no '${type}' within ${timeoutMs}ms. Recent log:\n${tail}`
  );
}

function readLogTail(n?: number): LogEvent[] {
  const buf = getBuffer();
  return n != null ? buf.slice(-n) : [...buf];
}

function clearLog(): void {
  (window as unknown as { __log?: LogEvent[] }).__log = [];
}

/**
 * Wait for the scene to reach steady state: any pending pattern select
 * commits + textures bind + two RAFs so the new uniforms make it onto
 * the screen.
 */
async function waitForRender(opts: { timeoutMs?: number } = {}): Promise<void> {
  const timeoutMs = opts.timeoutMs ?? 12_000;
  const state = useStore.getState();
  if (state.pendingSlug && state.pendingSlug !== state.activeSlug) {
    await waitForLog('texture_bound', (e) => e.slug === state.pendingSlug, timeoutMs);
  }
  await settleFrames();
}

async function selectPattern(slug: string, timeoutMs = 15_000): Promise<void> {
  const startTime = performance.now();
  const s = useStore.getState();
  s.beginSelect(slug);
  log('pattern_select_requested', { slug, debug: true });
  const m = await getDefault(slug);
  const committed = s.selectPatternIfCurrent(slug, m);
  log('pattern_selected', { slug, variant: m.variant, committed, debug: true });
  await waitForLog(
    'texture_bound',
    (e) => e.slug === slug && e.t >= startTime,
    timeoutMs
  );
  await settleFrames();
}

function setIllumination(mode: Illumination, color?: LaserColor): void {
  const s = useStore.getState();
  s.setIllumination(mode);
  if (color) s.setLaserColor(color);
}

function setLight(azDeg: number, elDeg: number): void {
  useStore.getState().setLight(azDeg, elDeg);
}

function setParam(key: string, value: number | string | boolean): void {
  useStore.getState().patchParams({ [key]: value });
}

/**
 * Programmatic OrbitControls camera placement. Replicates the spherical
 * math from frontend/tests/e2e/helpers.ts:setCameraAzEl so debug captures
 * and visual-signature captures land at identical viewpoints.
 *
 * Elevation is measured from +Y: 0° = camera looking straight down at the
 * plate, 90° = camera in the plate plane.
 */
function setCamera(azDeg: number, elDeg: number): void {
  const t = getThree();
  if (!t?.camera || !t?.controls) return;
  const camera = t.camera;
  const controls = t.controls;
  const target = controls.target ?? { x: 0, y: 0, z: 0 };
  const azRad = (azDeg * Math.PI) / 180;
  const phi = ((90 - elDeg) * Math.PI) / 180;
  const dx = camera.position.x - target.x;
  const dy = camera.position.y - target.y;
  const dz = camera.position.z - target.z;
  const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.8;
  const sinPhi = Math.sin(phi);
  const nx = target.x + radius * sinPhi * Math.sin(azRad);
  const ny = target.y + radius * Math.cos(phi);
  const nz = target.z + radius * sinPhi * Math.cos(azRad);
  camera.position.set(nx, ny, nz);
  camera.lookAt(target.x, target.y, target.z);
  if (typeof controls.update === 'function') controls.update();
}

/**
 * One-shot scene setup. Awaits texture binding when `pattern` is set, then
 * applies the synchronous knobs, then waits for one render cycle so the
 * caller can immediately captureCanvas() and trust the result.
 *
 * Idempotent: passing the same preset twice is a no-op except for the
 * trailing render wait.
 */
async function applyScene(preset: ScenePreset): Promise<void> {
  await waitForThreeContext();
  if (preset.pattern && useStore.getState().activeSlug !== preset.pattern) {
    await selectPattern(preset.pattern);
  }
  if (preset.illumination) setIllumination(preset.illumination, preset.laserColor);
  else if (preset.laserColor) useStore.getState().setLaserColor(preset.laserColor);
  if (preset.light) setLight(preset.light[0], preset.light[1]);
  if (preset.params) useStore.getState().patchParams(preset.params);
  if (preset.cameraAzEl) setCamera(preset.cameraAzEl[0], preset.cameraAzEl[1]);
  await waitForRender();
  api.lastScene = preset;
}

/** Snapshot of the Zustand state. Cheap — Zustand's getState returns the live object. */
function readState(): Record<string, unknown> {
  const s = useStore.getState() as unknown as Record<string, unknown>;
  // Prune function fields so JSON.stringify in eval responses stays small.
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(s)) {
    if (typeof v !== 'function') out[k] = v;
  }
  return out;
}

/**
 * Every shader uniform, JSON-serializable. Vectors are unwrapped to arrays,
 * colors to hex ints, textures to a {texture,w,h} stub. Any uniform we
 * can't classify is stringified — never returned as a live Three object,
 * which would blow up `JSON.stringify` in `preview_eval`.
 */
function readUniforms(): Record<string, unknown> {
  const t = getThree();
  if (!t?.material) return {};
  const u = t.material.uniforms ?? {};
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(u)) {
    const v = u[k]?.value;
    if (v == null) {
      out[k] = null;
    } else if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') {
      out[k] = v;
    } else if ((v as THREE.Vector2).isVector2 || (v as THREE.Vector3).isVector3 || (v as THREE.Vector4).isVector4) {
      out[k] = (v as THREE.Vector3).toArray();
    } else if ((v as THREE.Color).isColor) {
      out[k] = '#' + (v as THREE.Color).getHexString();
    } else if ((v as THREE.Texture).isTexture) {
      const img = (v as THREE.Texture).image as { width?: number; height?: number } | null;
      out[k] = { texture: true, w: img?.width ?? 0, h: img?.height ?? 0 };
    } else if (Array.isArray(v)) {
      out[k] = v;
    } else {
      out[k] = String(v);
    }
  }
  return out;
}

type PixelSample = { x: number; y: number; r: number; g: number; b: number; a: number };

/** Atomic render + readPixels for many points. Avoids per-call evaluate cost. */
function sampleMany(pts: [number, number][]): PixelSample[] {
  const t = getThree();
  if (!t) return [];
  t.renderer.render(t.scene, t.camera);
  const gl = t.renderer.getContext() as WebGLRenderingContext;
  const buf = new Uint8Array(4);
  return pts.map(([x, y]) => {
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    return { x, y, r: buf[0], g: buf[1], b: buf[2], a: buf[3] };
  });
}

type Quadrants = { ul: number; ur: number; ll: number; lr: number; total: number };

/**
 * 64×64 grid luminance sums per quadrant. Used to detect energy distribution
 * regressions (e.g. carrier-shift drop on far-field hologram, focal spot
 * drift on the muzo zone plate). Returns sums, not averages, so ratios
 * stay meaningful for mostly-dark scenes.
 */
function quadrantBrightness(): Quadrants {
  const t = getThree();
  if (!t) return { ul: 0, ur: 0, ll: 0, lr: 0, total: 0 };
  t.renderer.render(t.scene, t.camera);
  const gl = t.renderer.getContext() as WebGLRenderingContext;
  const canvas = t.renderer.domElement;
  const W = canvas.width;
  const H = canvas.height;
  const N = 64;
  const buf = new Uint8Array(4);
  let ul = 0, ur = 0, ll = 0, lr = 0;
  for (let iy = 0; iy < N; iy++) {
    for (let ix = 0; ix < N; ix++) {
      const x = Math.floor((ix + 0.5) * (W / N));
      const yGlFromTop = Math.floor((iy + 0.5) * (H / N));
      const y = H - 1 - yGlFromTop;
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      const L = (buf[0] + buf[1] + buf[2]) / 3;
      const top = iy < N / 2;
      const left = ix < N / 2;
      if (top && left) ul += L;
      else if (top && !left) ur += L;
      else if (!top && left) ll += L;
      else lr += L;
    }
  }
  return { ul, ur, ll, lr, total: ul + ur + ll + lr };
}

/** Atomic render + canvas.toDataURL — same path as captureScene in helpers.ts. */
function captureCanvas(): string {
  const t = getThree();
  if (!t) throw new Error('captureCanvas: __three not initialized');
  t.renderer.render(t.scene, t.camera);
  return t.renderer.domElement.toDataURL('image/png');
}

/**
 * Parse a ScenePreset from URL search params. Schema:
 *   ?pattern=<slug>&illum=ambient|laser|backlight&laser=red|green|blue
 *   &cam=<az>,<el>&light=<az>,<el>&param.<name>=<value>
 * Unknown params are ignored. Numeric param.* values are coerced when the
 * raw string parses cleanly; otherwise passed through as a string.
 */
export function parseSceneFromUrl(href = window.location.href): ScenePreset {
  const u = new URL(href);
  const p = u.searchParams;
  const preset: ScenePreset = {};
  const pattern = p.get('pattern');
  if (pattern) preset.pattern = pattern;
  const illum = p.get('illum');
  if (illum === 'ambient' || illum === 'laser' || illum === 'backlight') {
    preset.illumination = illum;
  }
  const laser = p.get('laser');
  if (laser === 'red' || laser === 'green' || laser === 'blue') {
    preset.laserColor = laser;
  }
  const cam = p.get('cam');
  if (cam) {
    const parts = cam.split(',').map(Number);
    if (parts.length === 2 && parts.every(Number.isFinite)) {
      preset.cameraAzEl = [parts[0], parts[1]];
    }
  }
  const light = p.get('light');
  if (light) {
    const parts = light.split(',').map(Number);
    if (parts.length === 2 && parts.every(Number.isFinite)) {
      preset.light = [parts[0], parts[1]];
    }
  }
  const params: Record<string, unknown> = {};
  for (const [k, v] of p.entries()) {
    if (!k.startsWith('param.')) continue;
    const name = k.slice('param.'.length);
    if (!name) continue;
    const num = Number(v);
    params[name] = Number.isFinite(num) && /^-?\d+(\.\d+)?$/.test(v) ? num : v;
  }
  if (Object.keys(params).length) preset.params = params;
  return preset;
}

export type DebugApi = {
  selectPattern: typeof selectPattern;
  setIllumination: typeof setIllumination;
  setCamera: typeof setCamera;
  setLight: typeof setLight;
  setParam: typeof setParam;
  applyScene: typeof applyScene;
  readState: typeof readState;
  readUniforms: typeof readUniforms;
  readLog: typeof readLogTail;
  clearLog: typeof clearLog;
  quadrantBrightness: typeof quadrantBrightness;
  sampleMany: typeof sampleMany;
  captureCanvas: typeof captureCanvas;
  waitForRender: typeof waitForRender;
  waitForLog: typeof waitForLog;
  parseSceneFromUrl: typeof parseSceneFromUrl;
  lastScene: ScenePreset | null;
};

const api: DebugApi = {
  selectPattern,
  setIllumination,
  setCamera,
  setLight,
  setParam,
  applyScene,
  readState,
  readUniforms,
  readLog: readLogTail,
  clearLog,
  quadrantBrightness,
  sampleMany,
  captureCanvas,
  waitForRender,
  waitForLog,
  parseSceneFromUrl,
  lastScene: null,
};

declare global {
  interface Window {
    __debug?: DebugApi;
  }
}

if (typeof window !== 'undefined' && import.meta.env.DEV) {
  window.__debug = api;
}

export { api as debugApi };
