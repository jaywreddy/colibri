import { log } from './logger';

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
// Plates + boxes — Phase I.5 / J. These wrap composed PlateSpec / BoxSpec
// objects; manifests share the recipe + texture surface of PatternManifest so
// existing scene code can render them unchanged.
// -----------------------------------------------------------------------------

export type FaceId = 'front' | 'back' | 'top' | 'bottom' | 'left' | 'right';
export const FACE_IDS: FaceId[] = ['front', 'back', 'top', 'bottom', 'left', 'right'];

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

export type PlateSpec = {
  pattern_slug: string;
  pattern_params: Record<string, unknown>;
  frame: FrameSpec;
  glass: GlassSpec;
  width_um: number;
  height_um: number;
  /** Blank rim around every edge reserved for assembly welds — no gold
   * is patterned inside this border. Default 1 mm. */
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
  width_um: number;
  height_um: number;
  depth_um: number;
  weld_margin_um: number;
  faces: Partial<Record<FaceId, PlateSpec>>;
  label: string;
};

export type BoxManifest = {
  kind: 'box';
  id: string;
  spec: BoxSpec;
  name: string;
  faces: Partial<Record<FaceId, PlateManifest>>;
  dimensions_um: { width: number; height: number; depth: number };
  content_hash: string;
};

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

export function defaultPlateSpec(patternSlug: string, seed = 1): PlateSpec {
  return {
    pattern_slug: patternSlug,
    pattern_params: {},
    frame: defaultFrameSpec(seed),
    glass: defaultGlassSpec(),
    // 30 mm (3 cm) plate edge — a hand-size piece with plenty of room for
    // the central optical pattern + decorative frame + 1 mm weld border.
    width_um: 30000,
    height_um: 30000,
    weld_margin_um: 1000,
    label: '',
  };
}

export function defaultBoxSpec(patternSlug: string): BoxSpec {
  const faces: Partial<Record<FaceId, PlateSpec>> = {};
  FACE_IDS.forEach((fid, i) => {
    faces[fid] = defaultPlateSpec(patternSlug, 100 + i);
  });
  return {
    width_um: 30000,
    height_um: 30000,
    depth_um: 30000,
    weld_margin_um: 1000,
    faces,
    label: '',
  };
}

export async function listPlates(): Promise<PlateManifest[]> {
  const r = await tracedFetch('/plates');
  if (!r.ok) throw new Error(`listPlates: ${r.status}`);
  return r.json();
}

export async function generatePlate(spec: PlateSpec, force = false): Promise<PlateManifest> {
  const r = await tracedFetch('/plates/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ spec, force }),
  });
  if (!r.ok) throw new Error(`generatePlate: ${r.status} ${await r.text()}`);
  return r.json();
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

export async function deleteBox(boxId: string): Promise<void> {
  const r = await tracedFetch(`/boxes/${boxId}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`deleteBox: ${r.status}`);
}
