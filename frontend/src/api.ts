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
  | 'phase_shift_overlay'
  | 'foliage_moire';

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
  foliage_moire: 3,
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
export const DEFAULT_PATTERN_SLUG = 'globe-duo-phase';

export type FrameSpec = {
  algorithm: 'wreath' | 'colonize';
  theme: 'esmeralda';
  density: number;
  bloom: number;
  foliage: number;
  seed: number;
  band_um: number | null;
  /** Outer-edge density bias (0 flat .. 1 strong). */
  edge_gradient: number;
  /** Density of the outer-band small-leaf infill. */
  understory: number;
  /** Continuous running-ornament border line. */
  border_vine: number;
  /** Size/reach of the corner fan compositions. */
  corner_fans: number;
  /** Wreath composition preset (wreath algorithm only). 'garland2' is the lush
   * mixed-tropical default; 'laurel' austere single-species; plus 'garland' | 'clusters'. */
  wreath_style: 'garland2' | 'laurel' | 'garland' | 'clusters';
};

export type GlassSpec = {
  thickness_um: number;
  material: string;
  n: number;
};

export type FoilFinish = 'bright' | 'copper' | 'patina' | 'gold' | 'rose' | 'gunmetal';

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
  /** Front-art blank rim: foil overlap + safety — no foliage gold inside. */
  weld_margin_um: number;
  /** Back-carrier blank rim: foil overlap only (wider window). null = fall
   * back to weld_margin_um for a standalone plate. */
  back_margin_um: number | null;
  /** Fabricated grating pitch (μm) of the back carrier + leaf louvre family.
   * Stamped from the box level; the louvre is this × 1.09. Litho floor 4 µm. */
  carrier_pitch_um: number;
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
  /** Box-level fabricated grating pitch (μm) — stamped onto every face by
   * stampFaces (mirrors backend normalize_face_dims). Default 22 µm. */
  carrier_pitch_um: number;
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
  back_window_um: number;
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

export function defaultFrameSpec(seed = 1, overrides: Partial<FrameSpec> = {}): FrameSpec {
  return {
    algorithm: 'wreath',
    theme: 'esmeralda',
    density: 1.0,
    bloom: 0.6,
    foliage: 0.6,
    seed,
    band_um: null,
    edge_gradient: 0.8,
    understory: 0.85,
    border_vine: 1.15,
    corner_fans: 1.0,
    wreath_style: 'garland2',
    ...overrides,
  };
}

/** Per-face frame recipe — mirrors backend boxes._FACE_FRAME_PROFILE so the
 * frontend default box (what the live preview POSTs) matches the backend's
 * default_box_spec exactly. Each face gets a distinct seed (→ distinct moiré
 * carrier angle) + distinct band composition, so every side reads uniquely. */
export const LID_PATTERN_SLUG = 'monogram-jp';
export const BOTTOM_PATTERN_SLUG = 'inscription-line';
export const BACK_PATTERN_SLUG = 'capybara-scanimation';
export const LEFT_PATTERN_SLUG = 'jamon-tray';
export const RIGHT_PATTERN_SLUG = 'gear-quill-switch';
/** Confirmed per-face default centerpiece — mirrors backend _FACE_PATTERN_SLUG.
 * front stays the colibrí↔globe switch (or a caller-supplied override); every
 * other wall gets its own showpiece. Keeps the live-preview POST byte-identical
 * to the backend's default_box_spec. */
const FACE_PATTERN_SLUG: Record<FaceId, string> = {
  front: DEFAULT_PATTERN_SLUG,
  back: BACK_PATTERN_SLUG,
  left: LEFT_PATTERN_SLUG,
  right: RIGHT_PATTERN_SLUG,
  top: LID_PATTERN_SLUG,
  bottom: BOTTOM_PATTERN_SLUG,
};
const FACE_FRAME_PROFILE: Record<FaceId, Partial<FrameSpec> & { seed: number }> = {
  front: { seed: 100, edge_gradient: 0.75, understory: 0.9, border_vine: 1.2, corner_fans: 1.1 },
  back: { seed: 101, edge_gradient: 1.0, understory: 0.6, border_vine: 0.9, corner_fans: 0.8 },
  top: { seed: 102, edge_gradient: 0.55, understory: 1.05, border_vine: 1.35, corner_fans: 1.25 },
  bottom: { seed: 103, edge_gradient: 0.9, understory: 0.75, border_vine: 1.0, corner_fans: 0.9 },
  left: { seed: 104, edge_gradient: 0.65, understory: 1.0, border_vine: 1.25, corner_fans: 1.15 },
  right: { seed: 105, edge_gradient: 0.85, understory: 0.8, border_vine: 1.05, corner_fans: 0.95 },
};

export function defaultGlassSpec(): GlassSpec {
  return { thickness_um: 500.0, material: 'fused silica', n: 1.46 };
}

export function defaultFoilSpec(): FoilSpec {
  return { tape_width_um: 6350.0, safety_um: 500.0, bead_um: 2000.0, finish: 'bright' };
}

export function defaultHingeSpec(): HingeSpec {
  return { style: 'tube', tube_od_um: 2400.0, rod_od_um: 1600.0, segments: 5, coverage: 0.8 };
}

export function defaultPlateSpec(
  patternSlug: string,
  seed = 1,
  frameOverrides: Partial<FrameSpec> = {}
): PlateSpec {
  return {
    pattern_slug: patternSlug,
    pattern_params: {},
    frame: defaultFrameSpec(seed, frameOverrides),
    glass: defaultGlassSpec(),
    width_um: 50000,
    height_um: 50000,
    weld_margin_um: 1000,
    back_margin_um: null,
    carrier_pitch_um: 22.0,
    label: '',
  };
}

export function defaultBoxSpec(patternSlug: string = DEFAULT_PATTERN_SLUG): BoxSpec {
  const faces: Partial<Record<FaceId, PlateSpec>> = {};
  FACE_IDS.forEach((fid) => {
    const { seed, ...frameOverrides } = FACE_FRAME_PROFILE[fid];
    // Confirmed six-face plan (see FACE_PATTERN_SLUG): top = J+P monogram,
    // bottom = hidden inscription, back = capybara scanimation, left = coffee +
    // arepa, right = gear↔quill, front = colibrí↔globe. A caller-supplied
    // `patternSlug` overrides only the FRONT face (themed override boxes).
    const slug = fid === 'front' ? patternSlug : FACE_PATTERN_SLUG[fid];
    faces[fid] = defaultPlateSpec(slug, seed, frameOverrides);
  });
  const spec: BoxSpec = {
    width_um: 50000.0,
    depth_um: 50000.0,
    height_um: 40000.0,
    glass: defaultGlassSpec(),
    foil: defaultFoilSpec(),
    hinge: defaultHingeSpec(),
    faces,
    carrier_pitch_um: 22.0,
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
