import { log } from './logger';
// NOTE: assembly.ts imports only *types* from this module, so this is not a
// runtime cycle — stampFaces keeps fresh BoxSpecs internally consistent.
import { stampFaces } from './assembly';

export type ParamSpec = {
  name: string;
  label: string;
  type: 'float' | 'int' | 'bool' | 'choice';
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  choices?: string[];
};

/**
 * Names map 1:1 to the `uRecipe` switch in plate.frag. Keep in sync with
 * RECIPE_NAMES in backend/app/patterns/base.py.
 */
export type RenderRecipe =
  | 'stereo_lenticular'
  | 'moire_interactive'
  | 'phase_shift_overlay';

export type PatternDescriptor = {
  slug: string;
  name: string;
  description: string;
  tags: string[];
  tier: 1 | 2 | 3;
  theme: 'Colombia' | 'Global Travel';
  render_recipe?: RenderRecipe;
  params: ParamSpec[];
};

export type PatternManifest = {
  slug: string;
  variant: string;
  name: string;
  description: string;
  tags: string[];
  params: Record<string, unknown>;
  substrate: { thickness_um: number; material: string; n: number };
  extent_um: [number, number];
  pixel_pitch_um: number;
  min_feature_um: number;
  extra: Record<string, unknown>;
  render_recipe?: RenderRecipe;
  recipe_data?: Record<string, unknown>;
  files: {
    front_png: string;
    back_png: string;
    front_svg: string;
    back_svg: string;
    thumbnail: string;
  };
};

export const RECIPE_IDS: Record<RenderRecipe, number> = {
  stereo_lenticular: 0,
  moire_interactive: 1,
  phase_shift_overlay: 2,
};

/**
 * Thin wrapper around `fetch` that pushes a `fetch_error` log event on
 * non-2xx responses and on network errors (including AbortError, which we
 * tag with `aborted: true` so test assertions can distinguish them from
 * genuine failures). Always re-throws so existing handlers behave the same.
 */
async function tracedFetch(url: string, init?: RequestInit): Promise<Response> {
  let r: Response;
  try {
    r = init === undefined ? await fetch(url) : await fetch(url, init);
  } catch (e) {
    const err = e as Error;
    log('fetch_error', {
      url,
      aborted: err.name === 'AbortError',
      message: err.message,
    });
    throw e;
  }
  if (!r.ok) {
    log('fetch_error', { url, status: r.status, statusText: r.statusText });
  }
  return r;
}

export async function listPatterns(): Promise<PatternDescriptor[]> {
  const r = await tracedFetch('/patterns');
  if (!r.ok) throw new Error(`listPatterns: ${r.status}`);
  return r.json();
}

export async function getDefault(
  slug: string,
  opts: { signal?: AbortSignal } = {}
): Promise<PatternManifest> {
  const r = await tracedFetch(`/patterns/${slug}/default`, { signal: opts.signal });
  if (!r.ok) throw new Error(`getDefault(${slug}): ${r.status}`);
  return r.json();
}

export async function generatePattern(
  slug: string,
  params: Record<string, unknown>
): Promise<PatternManifest> {
  const r = await tracedFetch('/patterns/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, params }),
  });
  if (!r.ok) throw new Error(`generatePattern: ${r.status} ${await r.text()}`);
  return r.json();
}

// -----------------------------------------------------------------------------
// Ring Box Studio — BoxSpec / BoxManifest v2 per the design contract.
// All stored/API values in micrometers (um). UI displays mm (1 decimal).
// -----------------------------------------------------------------------------

export type FaceId = 'front' | 'back' | 'top' | 'bottom' | 'left' | 'right';
export const FACE_IDS: FaceId[] = ['front', 'back', 'top', 'bottom', 'left', 'right'];

/** Default pattern slug stamped onto all six faces of a fresh box. */
export const DEFAULT_PATTERN_SLUG = 'wayuu-kanasu-moire';

export type FrameSpec = {
  algorithm: 'colonize';
  theme: 'esmeralda';
  density: number;
  bloom: number;
  foliage: number;
  seed: number;
  band_um: number | null;
};

export type GlassSpec = {
  thickness_um: number;
  material: string;
  n: number;
};

export type FoilFinish = 'bright' | 'copper' | 'patina';

export type FoilSpec = {
  /** Copper foil tape width. Presets: 4763 (3/16"), 5556 (7/32"), 6350 (1/4"). */
  tape_width_um: number;
  /** Extra pattern keep-out beyond the tape overlap. */
  safety_um: number;
  /** Solder bead DIAMETER for the preview render. */
  bead_um: number;
  finish: FoilFinish;
};

export const FOIL_TAPE_PRESETS_UM = [
  { label: '3/16″', um: 4763 },
  { label: '7/32″', um: 5556 },
  { label: '1/4″', um: 6350 },
] as const;

export type HingeSpec = {
  /** Brass tube-and-rod. */
  style: 'tube';
  tube_od_um: number;
  rod_od_um: number;
  /** Odd, >= 3; segments alternate body,lid,body,... (both ends body). */
  segments: number;
  /** Fraction of box width W spanned by the tube run, centered. */
  coverage: number;
};

/**
 * PlateSpec keeps its pre-box shape, BUT glass, width/height and weld_margin
 * are STAMPED from box level by normalization — they are not independent
 * degrees of freedom inside a box.
 */
export type PlateSpec = {
  pattern_slug: string;
  pattern_params: Record<string, unknown>;
  frame: FrameSpec;
  glass: GlassSpec;
  width_um: number;
  height_um: number;
  /** Blank rim reserved for foil overlap + safety — no gold patterned inside. */
  weld_margin_um: number;
  label: string;
};

export type PlateManifest = {
  kind: 'plate';
  id: string;
  spec: PlateSpec;
  name: string;
  description: string;
  tags: string[];
  substrate: { thickness_um: number; material: string; n: number };
  extent_um: [number, number];
  pixel_pitch_um: number;
  min_feature_um: number;
  extra: Record<string, unknown>;
  render_recipe?: RenderRecipe;
  recipe_data?: Record<string, unknown>;
  files: {
    front_png: string;
    back_png: string;
    front_svg: string;
    back_svg: string;
    thumbnail: string;
  };
};

export type BoxSpec = {
  /** Outer X. */
  width_um: number;
  /** Outer Z. */
  depth_um: number;
  /** Outer Y. */
  height_um: number;
  /** Box-level glass — applies to all six plates. */
  glass: GlassSpec;
  foil: FoilSpec;
  hinge: HingeSpec;
  faces: Partial<Record<FaceId, PlateSpec>>;
  label: string;
};

export type CutListEntry = {
  face: FaceId;
  width_um: number;
  height_um: number;
  width_mm: number;
  height_mm: number;
};

export type BoxAssemblyInfo = {
  keepout_um: number;
  overlap_um: number;
  glass_thickness_um: number;
  cut_list: CutListEntry[];
  seams: number | unknown[];
  hinge: HingeSpec & { run_length_um: number; segment_length_um: number };
};

export type BoxManifest = {
  kind: 'box';
  id: string;
  spec: BoxSpec;
  name: string;
  /** Per-face manifests with recipe_data slimmed (no "frame_scene"). */
  faces: Partial<Record<FaceId, PlateManifest>>;
  dimensions_um: { width: number; height: number; depth: number };
  assembly: BoxAssemblyInfo;
  content_hash: string;
};

// -----------------------------------------------------------------------------
// Defaults — MUST match the backend dataclass defaults exactly.
// -----------------------------------------------------------------------------

export function defaultFrameSpec(seed = 1): FrameSpec {
  return {
    algorithm: 'colonize',
    theme: 'esmeralda',
    density: 1.0,
    bloom: 0.6,
    foliage: 0.6,
    seed,
    band_um: null,
  };
}

export function defaultGlassSpec(): GlassSpec {
  return { thickness_um: 500.0, material: 'fused silica', n: 1.46 };
}

export function defaultFoilSpec(): FoilSpec {
  return { tape_width_um: 6350.0, safety_um: 500.0, bead_um: 2000.0, finish: 'bright' };
}

export function defaultHingeSpec(): HingeSpec {
  return { style: 'tube', tube_od_um: 2400.0, rod_od_um: 1600.0, segments: 5, coverage: 0.8 };
}

export function defaultPlateSpec(patternSlug: string, seed = 1): PlateSpec {
  return {
    pattern_slug: patternSlug,
    pattern_params: {},
    frame: defaultFrameSpec(seed),
    glass: defaultGlassSpec(),
    width_um: 50000,
    height_um: 50000,
    weld_margin_um: 1000,
    label: '',
  };
}

export function defaultBoxSpec(patternSlug: string = DEFAULT_PATTERN_SLUG): BoxSpec {
  const faces: Partial<Record<FaceId, PlateSpec>> = {};
  FACE_IDS.forEach((fid, i) => {
    faces[fid] = defaultPlateSpec(patternSlug, 100 + i);
  });
  const spec: BoxSpec = {
    width_um: 50000.0,
    depth_um: 50000.0,
    height_um: 40000.0,
    glass: defaultGlassSpec(),
    foil: defaultFoilSpec(),
    hinge: defaultHingeSpec(),
    faces,
    label: '',
  };
  // Stamp glass / cut dims / keep-out into the faces so the spec is
  // internally consistent before it ever reaches the backend.
  return stampFaces(spec);
}

export async function listBoxes(): Promise<BoxManifest[]> {
  const r = await tracedFetch('/boxes');
  if (!r.ok) throw new Error(`listBoxes: ${r.status}`);
  return r.json();
}

export async function getBox(boxId: string): Promise<BoxManifest> {
  const r = await tracedFetch(`/boxes/${boxId}`);
  if (!r.ok) throw new Error(`getBox: ${r.status}`);
  return r.json();
}

export async function generateBox(
  spec: BoxSpec,
  opts: { boxId?: string; force?: boolean } = {}
): Promise<BoxManifest> {
  const body = { ...spec, box_id: opts.boxId, force: opts.force ?? false };
  const r = await tracedFetch('/boxes/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`generateBox: ${r.status} ${await r.text()}`);
  return r.json();
}
