import * as THREE from 'three';

/**
 * Procedural microsurface maps for the copper-foil / solder metals.
 *
 * Everything here is generated from a small 2D canvas (256px) — no external
 * image assets, no fetches. The maps give the otherwise dead-flat
 * PBR metals the low-frequency variation a real hand-soldered copper-foil box
 * has: rolled-tape streaks, colour mottle, fingerprint-blotchy solder, and a
 * heat-patina darkening on the foil right next to a soldered seam.
 *
 * A tiny seeded PRNG keeps the noise deterministic so textures do not "boil"
 * between rebuilds (same finish -> same canvas).
 *
 * Because they ARE deterministic, every map set here is memoized at module
 * level on its full input tuple — a hit is bit-exact, not an approximation.
 * That matters: the maps cost ~720k CPU noise samples plus ~29 canvas uploads
 * per BoxScene rebuild, and a finish/dimension slider drags the rebuild at
 * frame rate. CACHED TEXTURES ARE OWNED BY THIS MODULE — callers must NOT
 * push them into their per-rebuild disposal lists (see BoxScene's rebuild
 * disposables). They survive a GPU context loss on their own: unlike a render
 * target, a CanvasTexture keeps its CPU-side source and three re-uploads it.
 */

/**
 * Insert into a bounded FIFO texture cache, disposing whatever falls out.
 *
 * Disposing an evicted texture is safe even if some live material still points
 * at it: dispose() only frees the GL upload, and three re-uploads from the
 * canvas source on the next render. The cap only exists because the heat maps
 * are seeded from plate dimensions, so a dimension sweep would otherwise grow
 * the cache without bound.
 */
function cachePut<T extends THREE.Texture>(
  cache: Map<string, T>,
  key: string,
  value: T,
  cap: number
): T {
  cache.set(key, value);
  while (cache.size > cap) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.get(oldest.value)?.dispose();
    cache.delete(oldest.value);
  }
  return value;
}

/** Mulberry32 — cheap deterministic 32-bit PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Smooth value-noise sampled from a coarse random lattice (bilinear). */
function makeValueNoise(
  cells: number,
  rnd: () => number
): (x: number, y: number) => number {
  const n = cells + 1;
  const grid = new Float32Array(n * n);
  for (let i = 0; i < grid.length; i++) grid[i] = rnd();
  return (x: number, y: number): number => {
    // x, y in [0,1)
    const fx = x * cells;
    const fy = y * cells;
    const x0 = Math.floor(fx) % cells;
    const y0 = Math.floor(fy) % cells;
    const x1 = (x0 + 1) % cells;
    const y1 = (y0 + 1) % cells;
    const tx = fx - Math.floor(fx);
    const ty = fy - Math.floor(fy);
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const a = grid[y0 * n + x0];
    const b = grid[y0 * n + x1];
    const c = grid[y1 * n + x0];
    const d = grid[y1 * n + x1];
    const top = a + (b - a) * sx;
    const bot = c + (d - c) * sx;
    return top + (bot - top) * sy;
  };
}

/** Multi-octave fbm from a value-noise field. */
function fbm(
  noise: (x: number, y: number) => number,
  x: number,
  y: number,
  octaves: number
): number {
  let sum = 0;
  let amp = 0.5;
  let freq = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise((x * freq) % 1, (y * freq) % 1);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

export type FinishTuning = {
  /** 0 = smooth rolled tape, 1 = heavily oxidised / matte. */
  oxidation: number;
  /** Base tint (linear-ish sRGB hex string). */
  tint: string;
};

function finalizeTex(
  canvas: HTMLCanvasElement,
  srgb: boolean,
  repeat?: [number, number]
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  if (repeat) tex.repeat.set(repeat[0], repeat[1]);
  tex.needsUpdate = true;
  return tex;
}

/** Parse "#rrggbb" -> [r,g,b] in 0..255. */
function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

export type FoilMaps = {
  color: THREE.CanvasTexture;
  rough: THREE.CanvasTexture;
  /**
   * `rough` with a 90deg UV rotation about its centre, for strips whose long
   * axis is the plate Y — the brushed streaks must run along the physical
   * strip. Shared by every vertical strip: rotation/centre are identical, so
   * one clone replaces the 24 per-rebuild clones this used to make.
   */
  roughRotated: THREE.CanvasTexture;
};

/** Finish -> map set. At most one entry per FOIL_COLORS finish (6). */
const FOIL_CACHE = new Map<string, FoilMaps>();
const SOLDER_CACHE = new Map<string, SolderMaps>();
/** Heat maps are seeded from plate dimensions too, so this one is bounded. */
const HEAT_CACHE = new Map<string, THREE.CanvasTexture>();
const HEAT_CACHE_CAP = 128;

/**
 * Foil (rolled copper tape) map set.
 *  - color: base tint with subtle warm/cool mottle,
 *  - roughness: brushed streaks running along the tape length. The tape long
 *    axis is the texture U axis; callers rotate the strip UVs so the streaks
 *    always run along the physical strip (see BoxScene addFoilFrame).
 *
 * Memoized on (finish, tuning) — see the module docstring on ownership.
 */
export function makeFoilMaps(finish: string, tuning: FinishTuning): FoilMaps {
  const key = `${finish}|${tuning.oxidation}|${tuning.tint}`;
  const hit = FOIL_CACHE.get(key);
  if (hit) return hit;
  const built = buildFoilMaps(finish, tuning);
  FOIL_CACHE.set(key, built);
  return built;
}

function buildFoilMaps(finish: string, tuning: FinishTuning): FoilMaps {
  const S = 256;
  const seedBase = hashStr(finish + ':foil');
  const [r, g, b] = hexRgb(tuning.tint);

  // --- colour mottle ---
  const cCanvas = document.createElement('canvas');
  cCanvas.width = cCanvas.height = S;
  const cx = cCanvas.getContext('2d')!;
  const cNoise = makeValueNoise(6, mulberry32(seedBase));
  const cImg = cx.createImageData(S, S);
  const mottle = 0.14 + tuning.oxidation * 0.22;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      const m = fbm(cNoise, u, v, 4) - 0.5; // -0.5..0.5
      const f = 1 + m * 2 * mottle;
      // Slight hue drift: shift toward warm in the darker blotches.
      const warm = 1 + m * 0.12;
      const i = (y * S + x) * 4;
      cImg.data[i] = clamp255(r * f * warm);
      cImg.data[i + 1] = clamp255(g * f);
      cImg.data[i + 2] = clamp255(b * f * (2 - warm));
      cImg.data[i + 3] = 255;
    }
  }
  cx.putImageData(cImg, 0, 0);

  // --- brushed roughness: fine streaks along U + broad blotches ---
  const rCanvas = document.createElement('canvas');
  rCanvas.width = rCanvas.height = S;
  const rx = rCanvas.getContext('2d')!;
  const rNoiseBroad = makeValueNoise(5, mulberry32(seedBase ^ 0x9e37));
  // Streak noise: many cells along V (across the tape), few along U (along it),
  // which stretches the noise into streaks parallel to U.
  const streakRnd = mulberry32(seedBase ^ 0x51ab);
  const streakRows = new Float32Array(S);
  for (let y = 0; y < S; y++) streakRows[y] = streakRnd();
  const rImg = rx.createImageData(S, S);
  const baseRough = 0.18 + tuning.oxidation * 0.42;
  for (let y = 0; y < S; y++) {
    // combine a couple of neighbouring rows so streaks have soft width
    const sy = y / S;
    const streak =
      0.6 * streakRows[y] +
      0.25 * streakRows[(y + 1) % S] +
      0.15 * streakRows[(y + 3) % S];
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const broad = fbm(rNoiseBroad, u, sy, 3) - 0.5;
      // High-freq scratch along U so the streak has grain.
      const scratch = Math.sin(u * Math.PI * 2 * 40 + streak * 30) * 0.5 + 0.5;
      let rough =
        baseRough +
        (streak - 0.5) * 0.28 +
        broad * 0.18 +
        (scratch - 0.5) * 0.06;
      rough = Math.min(0.98, Math.max(0.05, rough));
      const val = clamp255(rough * 255);
      const i = (y * S + x) * 4;
      rImg.data[i] = rImg.data[i + 1] = rImg.data[i + 2] = val;
      rImg.data[i + 3] = 255;
    }
  }
  rx.putImageData(rImg, 0, 0);

  const rough = finalizeTex(rCanvas, false);
  const roughRotated = rough.clone() as THREE.CanvasTexture;
  roughRotated.center.set(0.5, 0.5);
  roughRotated.rotation = Math.PI / 2;
  roughRotated.needsUpdate = true;

  return {
    color: finalizeTex(cCanvas, true),
    rough,
    roughRotated,
  };
}

export type SolderMaps = {
  color: THREE.CanvasTexture;
  rough: THREE.CanvasTexture;
  bump: THREE.CanvasTexture;
};

/**
 * Solder map set: blotchy "fingerprint" roughness + a matching bump so the
 * frozen bead surface has real micro-relief instead of mirror-smoothness.
 *
 * Memoized on (finish, tuning) — see the module docstring on ownership.
 */
export function makeSolderMaps(finish: string, tuning: FinishTuning): SolderMaps {
  const key = `${finish}|${tuning.oxidation}|${tuning.tint}`;
  const hit = SOLDER_CACHE.get(key);
  if (hit) return hit;
  const built = buildSolderMaps(finish, tuning);
  SOLDER_CACHE.set(key, built);
  return built;
}

function buildSolderMaps(finish: string, tuning: FinishTuning): SolderMaps {
  const S = 256;
  const seedBase = hashStr(finish + ':solder');
  const [r, g, b] = hexRgb(tuning.tint);

  const noiseA = makeValueNoise(7, mulberry32(seedBase));
  const noiseB = makeValueNoise(14, mulberry32(seedBase ^ 0x2c1a));

  const field = new Float32Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      // Blotches (low freq) + pitting (higher freq).
      const blotch = fbm(noiseA, u, v, 3);
      const pit = fbm(noiseB, u, v, 2);
      field[y * S + x] = blotch * 0.7 + pit * 0.3;
    }
  }

  // color
  const cCanvas = document.createElement('canvas');
  cCanvas.width = cCanvas.height = S;
  const cx = cCanvas.getContext('2d')!;
  const cImg = cx.createImageData(S, S);
  const cMottle = 0.12 + tuning.oxidation * 0.25;
  for (let i = 0; i < S * S; i++) {
    const m = field[i] - 0.5;
    const f = 1 + m * 2 * cMottle;
    cImg.data[i * 4] = clamp255(r * f);
    cImg.data[i * 4 + 1] = clamp255(g * f);
    cImg.data[i * 4 + 2] = clamp255(b * f);
    cImg.data[i * 4 + 3] = 255;
  }
  cx.putImageData(cImg, 0, 0);

  // roughness
  const rCanvas = document.createElement('canvas');
  rCanvas.width = rCanvas.height = S;
  const rx = rCanvas.getContext('2d')!;
  const rImg = rx.createImageData(S, S);
  const baseRough = 0.22 + tuning.oxidation * 0.4;
  for (let i = 0; i < S * S; i++) {
    let rough = baseRough + (field[i] - 0.5) * 0.5;
    rough = Math.min(0.95, Math.max(0.08, rough));
    const v = clamp255(rough * 255);
    rImg.data[i * 4] = rImg.data[i * 4 + 1] = rImg.data[i * 4 + 2] = v;
    rImg.data[i * 4 + 3] = 255;
  }
  rx.putImageData(rImg, 0, 0);

  // bump (same field, higher contrast)
  const bCanvas = document.createElement('canvas');
  bCanvas.width = bCanvas.height = S;
  const bx = bCanvas.getContext('2d')!;
  const bImg = bx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    const v = clamp255(field[i] * 255);
    bImg.data[i * 4] = bImg.data[i * 4 + 1] = bImg.data[i * 4 + 2] = v;
    bImg.data[i * 4 + 3] = 255;
  }
  bx.putImageData(bImg, 0, 0);

  // Geometry UVs already tile these along/around the bead (see solderBead.ts),
  // so leave the texture repeat at 1x1.
  return {
    color: finalizeTex(cCanvas, true),
    rough: finalizeTex(rCanvas, false),
    bump: finalizeTex(bCanvas, false),
  };
}

/**
 * Heat-patina gradient for a single foil strip. The strip's UV V axis runs
 * across the tape (0 at one long edge, 1 at the other); soldered seams sit at
 * whichever long edge touches a plate join. `seamEdges` marks which of the two
 * long edges (v=0 / v=1) and the two ends (u=0 / u=1) are next to solder, and
 * the darkened iridescent tint fades a few mm inward.
 *
 * Returned as an alpha map to layer over the base foil colour via a second
 * material would be heavy; instead we bake it directly into a per-strip colour
 * texture, tinting the base foil colour toward a heat-oxidised hue near seams.
 *
 * Memoized on (base tint, seam edges, seed) — see the module docstring on
 * ownership. This is the hottest builder here: 24 of these (128² x 3-octave
 * fbm each) per assembled-layout rebuild.
 */
export function makeStripHeatColor(
  base: [number, number, number],
  seamEdges: { v0: boolean; v1: boolean; u0: boolean; u1: boolean },
  seed: number
): THREE.CanvasTexture {
  const edgeBits =
    (seamEdges.v0 ? 1 : 0) |
    (seamEdges.v1 ? 2 : 0) |
    (seamEdges.u0 ? 4 : 0) |
    (seamEdges.u1 ? 8 : 0);
  const key = `${base[0]},${base[1]},${base[2]}|${edgeBits}|${seed}`;
  const hit = HEAT_CACHE.get(key);
  if (hit) return hit;
  return cachePut(HEAT_CACHE, key, buildStripHeatColor(base, seamEdges, seed), HEAT_CACHE_CAP);
}

function buildStripHeatColor(
  base: [number, number, number],
  seamEdges: { v0: boolean; v1: boolean; u0: boolean; u1: boolean },
  seed: number
): THREE.CanvasTexture {
  const S = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = S;
  const cx = canvas.getContext('2d')!;
  const noise = makeValueNoise(8, mulberry32(seed));
  const img = cx.createImageData(S, S);
  // Heat tint: darkened brown-violet iridescence.
  const heat: [number, number, number] = [70, 46, 58];
  const feather = 0.34; // fraction of strip that patina reaches
  for (let y = 0; y < S; y++) {
    const v = y / S;
    for (let x = 0; x < S; x++) {
      const u = x / S;
      // distance (0..1) to the nearest soldered edge
      let d = 1;
      if (seamEdges.v0) d = Math.min(d, v);
      if (seamEdges.v1) d = Math.min(d, 1 - v);
      if (seamEdges.u0) d = Math.min(d, u);
      if (seamEdges.u1) d = Math.min(d, 1 - u);
      // patina strength: 1 at seam, 0 past the feather band, with wobble
      const wob = (fbm(noise, u, v, 3) - 0.5) * 0.18;
      let t = 1 - d / feather + wob;
      t = Math.max(0, Math.min(1, t));
      // ease
      t = t * t * (3 - 2 * t);
      const i = (y * S + x) * 4;
      img.data[i] = clamp255(base[0] * (1 - t) + heat[0] * t);
      img.data[i + 1] = clamp255(base[1] * (1 - t) + heat[1] * t);
      img.data[i + 2] = clamp255(base[2] * (1 - t) + heat[2] * t);
      img.data[i + 3] = 255;
    }
  }
  cx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

function clamp255(x: number): number {
  return x < 0 ? 0 : x > 255 ? 255 : Math.round(x);
}

/** Small deterministic string hash for per-finish seeds. */
export function hashStr(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
