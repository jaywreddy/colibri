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
 * The render recipe a manifest may declare. ONE is left: every composed box
 * plate binds `foliage_moire` (plates.py forces it) and BoxScene REFUSES
 * anything else rather than falling back on a single-plane approximation.
 *
 * The retired names and their ids stay documented because the backend's
 * RECIPE_NAMES (app/patterns/base.py) is a positional list and the numbers
 * must never be reused: 0 stereo_lenticular and 1 moire_interactive were the
 * single-plane previews of standalone patterns, 2 phase_shift_overlay was the
 * banned two-image phase split (a front-layer image cannot vanish under
 * parallax), and foliage_moire has been 3 throughout.
 */
export type RenderRecipe = 'foliage_moire';

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
    /** Per-litho-metal chips (gold | chrome | chrome-ar). Absent on manifests
     * written before per-metal thumbnails existed — fall back to `thumbnail`. */
    thumbnails?: Record<string, string>;
  };
};

/**
 * Recipe name → the backend's numeric id. plate.frag no longer switches on it
 * (it implements foliage_moire and nothing else), but BoxScene still records
 * the accepted recipe per face as `uRecipe` so a dump says which geometry the
 * bind path agreed to draw. 3, not 0, because ids 0-2 are retired names the
 * backend's positional RECIPE_NAMES must never reuse.
 */
export const RECIPE_IDS: Record<RenderRecipe, number> = {
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

/**
 * Human-readable text for a non-2xx response, safe to show in the header
 * error banner or a picker retry tile.
 *
 * FastAPI reports every failure as `{"detail": "..."}` and our handlers put
 * the actionable sentence there ("regenerate the box and export again"), so
 * that field is what the user needs — not the raw body, which for a proxy
 * error or a truncated stream is an HTML page. Truncated: the banner is one
 * line, and the full text still reaches the `title` tooltip.
 *
 * Every thrower whose message can reach a HUMAN routes through here — which is
 * now box load and box generate, the only two calls this visualizer makes that
 * can be refused with something worth reading (the 400k lattice budget, the
 * ParamSpec range check, the litho floor). Keep it that way when adding an
 * endpoint: a status code plus the RAW body is how those carefully worded
 * refusals used to surface as JSON repr noise.
 *
 * `listPatterns` / `listBoxes` deliberately still throw a bare status line:
 * neither message is ever shown (the catalog fetch retries forever and only
 * logs; the box listing is a debug affordance), and the non-ok unit test pins
 * `listPatterns` to a message containing the status code.
 */
async function errorDetail(r: Response, what: string): Promise<string> {
  let body = '';
  try {
    body = await r.text();
  } catch {
    /* connection dropped mid-body — fall back to the status line */
  }
  let detail = body.trim();
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed?.detail === 'string') {
      detail = parsed.detail;
    } else if (Array.isArray(parsed?.detail)) {
      // pydantic 422: [{loc, msg, ...}] — the `msg` fields are the readable part.
      const msgs = parsed.detail
        .map((d) => (d as { msg?: unknown } | null)?.msg)
        .filter((m): m is string => typeof m === 'string');
      if (msgs.length > 0) detail = msgs.join('; ');
    }
  } catch {
    /* not JSON — keep the raw text */
  }
  if (detail.length > 300) detail = `${detail.slice(0, 300)}…`;
  return detail
    ? `${what}: ${detail}`
    : `${what}: HTTP ${r.status}${r.statusText ? ` ${r.statusText}` : ''}`;
}

export async function listPatterns(): Promise<PatternDescriptor[]> {
  const r = await tracedFetch('/patterns');
  if (!r.ok) throw new Error(`listPatterns: ${r.status}`);
  return r.json();
}

// GET /patterns/{slug}/default and POST /patterns/generate had exactly one
// caller each — the Pattern Lab, which asked the backend for a STANDALONE
// pattern. The visualizer shows one box, composed in code, so the only thing it
// asks the backend to build is a box (generateBox below); the picker's
// thumbnails come straight off the cached PNG URLs (store.ts::loadThumbnails).
// Both routes stay live for the CLI and for curl; the clients are gone.

// -----------------------------------------------------------------------------
// Ring Box Studio — BoxSpec / BoxManifest v2 per the design contract.
// All stored/API values in micrometers (um). UI displays mm (1 decimal).
// -----------------------------------------------------------------------------

export type FaceId = 'front' | 'back' | 'top' | 'bottom' | 'left' | 'right';
export const FACE_IDS: FaceId[] = ['front', 'back', 'top', 'bottom', 'left', 'right'];

/** Default pattern slug stamped onto all six faces of a fresh box. */
export const DEFAULT_PATTERN_SLUG = 'globe-atlantic';

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
  /** Motif-only size dial (wreath only): scales leaf/bloom/understory sizes and
   * their spacing along the vine, leaving the band width and vine gauge alone.
   * 1.0 is the tuned production look. */
  motif_scale: number;
};

export type GlassSpec = {
  thickness_um: number;
  material: string;
  n: number;
};

export type FoilFinish = 'bright' | 'copper' | 'patina' | 'gold' | 'rose' | 'gunmetal';

export type FoilSpec = {
  /** Copper foil tape width. Presets: 4763 (3/16"), 5556 (7/32"), 6350 (1/4"), 7938 (5/16"), 9525 (3/8"). */
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
  { label: '5/16″', um: 7938 },
  { label: '3/8″', um: 9525 },
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
  /**
   * Single-ply face: chrome on the OUTER ply only, the inner ply left as bare
   * glass. The composed plate then publishes an EMPTY back raster, the renderer
   * skips the inner pattern plane entirely (the glass slabs stay), and nothing
   * on this face can beat against a second layer — which is the point for a
   * continuous-tone photo halftone, where a back carrier would only add a moiré
   * the picture does not want. Default false (both plies carry chrome).
   */
  single_ply: boolean;
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
  /** Per-face carrier scaling policy vs the plate's real glass: 'gap'
   * (default) scales the fabricated carrier family with the paraxial gap t/n,
   * preserving the designed reveal tilt on any stock (no-op at the 500 µm
   * baseline); 'fixed' keeps the literal pitch — on thick stock the reveals
   * compress into sub-degree refraction shimmer. Barrier switch periods scale
   * with the gap regardless. */
  carrier_scale_mode?: 'gap' | 'fixed';
  label: string;
};

/**
 * A composed plate's `recipe_data`. Still an open bag — the renderer reads
 * ~30 scalar knobs out of it by name — but the three keys that decide WHICH
 * material path a face takes are typed, because getting one of them wrong is
 * not a shading difference, it is the wrong physics on the wall.
 */
export type PlateRecipeData = Record<string, unknown> & {
  /**
   * This face publishes LITERAL rasters of the fabricated chrome geometry
   * (`files.literal_front` / `literal_back`), so the renderer samples the real
   * mask instead of drawing procedural gratings inside level-coded regions.
   * Moiré, switches and shimmer then emerge from perspective across the real
   * T/n plane gap with no analytic grating anywhere in the shader.
   */
  literal?: boolean;
  /** Informational mirror of `spec.single_ply` — the back raster is empty. */
  single_ply?: boolean;
  /** Informational: BOTH rasters are empty (a bare-glass face). */
  blank?: boolean;
  /** The per-leaf diffractive grating pitch (µm) a single-ply face writes its
   * garland at (0, or absent, on a two-ply face, whose leaves are moiré
   * louvres instead). Mirrors backend `plates.single_ply_leaf_period_um()` —
   * ONE helper answers here and in `files.period_front`, so the advertised
   * pitch is the pitch that map carries over the leaf texels outside the art
   * box. Under the shipping "hue" fill the leaves are written at one PERIOD
   * PER MOTIF FAMILY (4.15–6.02 µm), which a single scalar can only summarise:
   * this is that ladder's mean (≈5.04 µm), and the per-family split is in the
   * period map itself. */
  single_ply_leaf_period_um?: number;
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
  recipe_data?: PlateRecipeData;
  files: {
    front_png: string;
    back_png: string;
    front_svg: string;
    back_svg: string;
    thumbnail: string;
    /** Per-litho-metal chips (gold | chrome | chrome-ar). Absent on manifests
     * written before per-metal thumbnails existed — fall back to `thumbnail`. */
    thumbnails?: Record<string, string>;
    /**
     * LITERAL rasters of the fabricated chrome, one per layer. PNG mode L,
     * 2048 px on the long side, 255 = metal present / 0 = bare glass, same
     * orientation and extent as `front_png`/`back_png` (the whole plate incl.
     * weld margin and design frame, unmirrored). Present iff
     * `recipe_data.literal`; an all-zero raster means that layer carries no
     * chrome at all (blank face, or the back of a single-ply face) and the
     * renderer drops the plane rather than uploading an empty texture.
     */
    literal_front?: string;
    literal_back?: string;
    /**
     * Optional sub-grating period map for the FRONT layer, same size/mode:
     * R = period µm × 25 (0..255 → 0..10.2 µm; 0 = no sub-grating). Bands that
     * carry a diffraction colour grating light up through diffractionSheen()
     * at the sampled period.
     */
    period_front?: string;
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
  /** Bonded (two-ply) construction: each face is TWO single-side plates glued
   * face-to-face. glass.thickness_um is then the PLY (also the optical
   * parallax gap); the wall is 2x; cut dims / foil margins follow the
   * nested-shell math. Default false — the production box is six single plies
   * since 2026-09-16. */
  bonded?: boolean;
  /** PINNED art rim (um), overriding the rim stampFaces would derive from the
   * foil, on BOTH layers of every face. null/absent = derive it. The production
   * box pins it because its mask is already written — mirrors backend
   * BoxSpec.art_rim_um / boxes.PRODUCTION_ART_RIM_UM. */
  art_rim_um?: number | null;
  /** Litho metal for the PREVIEW's conductor response (masks are identical):
   * 'gold' | 'chrome' (bright, platinum-line read) | 'chrome-ar' (AR-coated
   * mask grade, ink-black linework). */
  metal?: 'gold' | 'chrome' | 'chrome-ar';
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
    motif_scale: 1.0,
    ...overrides,
  };
}

/** Per-face frame recipe — mirrors backend boxes._FACE_FRAME_PROFILE so the
 * frontend default box (what the live preview POSTs) matches the backend's
 * default_box_spec exactly. Each face gets a distinct seed (→ distinct moiré
 * carrier angle) + distinct band composition, so every side reads uniquely. */
export const LID_PATTERN_SLUG = 'monogram-jp';
/** Bare glass — no centerpiece, no frame gold. Both literal rasters empty. */
export const BLANK_PATTERN_SLUG = 'blank';
/** Continuous-tone photo, rasterised as halftone bands in the front layer. */
export const PHOTO_PATTERN_SLUG = 'photo-halftone';
export const SOLID_PATTERN_SLUG = 'solid-gold';
export const BOTTOM_PATTERN_SLUG = SOLID_PATTERN_SLUG; // a solid gold base plate (2026-09-15)
export const BACK_PATTERN_SLUG = 'photo-halftone'; // a third photograph (2026-09-15)
export const LEFT_PATTERN_SLUG = PHOTO_PATTERN_SLUG;
export const RIGHT_PATTERN_SLUG = PHOTO_PATTERN_SLUG;

/**
 * The slugs the face picker offers, in picker order.
 *
 * The visualizer shows ONE box — the six faces backend `boxes.default_box_spec`
 * composes — so this is a whitelist, not a catalogue listing: whatever
 * GET /patterns happens to return (a dev exemplar, a slug a stale backend
 * still registers) is filtered against it, and a pattern that is not one of
 * the box's own constructions can never be assigned to a wall from the UI.
 */
export const PICKER_SLUGS: string[] = [
  LID_PATTERN_SLUG,
  DEFAULT_PATTERN_SLUG,
  PHOTO_PATTERN_SLUG,
  SOLID_PATTERN_SLUG,
  BLANK_PATTERN_SLUG,
];

/**
 * Registered but kept OUT of the picker: the two-ply exemplars that exist so
 * the renderer's barrier-interlace and shading-moiré paths (and the @effects
 * suite that pins them) still have a subject. They are not faces of this box —
 * every production wall is single-ply — so they are reachable only by setting
 * the slug directly (store.patchFace, which is what the e2e specs do).
 */
export const DEV_EXEMPLAR_SLUGS: string[] = ['globe-duo-phase'];

/** Pattern param that names which prepared photograph a photo face carries. */
export const PHOTO_PARAM = 'image';
/**
 * Every prepared photograph ships an `<image>.colour.json` authored plan
 * (backend app/assets/photos), and the box is designed around those plans, so
 * the one photo knob the UI exposes never has to offer a colour mode.
 */
export const PHOTO_COLOUR_MODE = 'authored';
/**
 * The PRODUCTION six-face plan — mirrors backend boxes._FACE_PATTERN_SLUG /
 * _FACE_PATTERN_PARAMS / _FACE_SINGLE_PLY, so the live-preview POST is the
 * same box the fab bake ships.
 *
 *   TOP    — the interlocked cursive J+P monogram, a single-layer diffraction
 *            mapping (colour by region).
 *   FRONT  — the Atlantic globe (US with California, Colombia, Europe in one
 *            view), colour by region (or a caller-supplied override).
 *   LEFT   — the beach photo, halftoned, its AUTHORED colour plan.
 *   RIGHT  — the sunset photo, halftoned, its AUTHORED colour plan.
 *   BACK   — the Paris photo, halftoned, its AUTHORED colour plan.
 *   BOTTOM — solid gold: the base plate.
 *
 * EVERY face is SINGLE-PLY (2026-09-15): the bonded moiré effects of the first
 * plate read badly on glass, so each face is one written ply over a bare
 * inner ply. Mirrors backend boxes._SINGLE_PLY_FACES = all six.
 */
type FacePlan = {
  slug: string;
  params: Record<string, unknown>;
  singlePly: boolean;
};
const FACE_PLAN: Record<FaceId, FacePlan> = {
  front: { slug: DEFAULT_PATTERN_SLUG, params: {}, singlePly: true },
  back: { slug: BACK_PATTERN_SLUG, params: { image: 'paris', colour_mode: 'authored' }, singlePly: true },
  left: {
    slug: LEFT_PATTERN_SLUG,
    params: { image: 'beach', colour_mode: 'authored' },
    singlePly: true,
  },
  right: {
    slug: RIGHT_PATTERN_SLUG,
    params: { image: 'sunset', colour_mode: 'authored' },
    singlePly: true,
  },
  top: { slug: LID_PATTERN_SLUG, params: {}, singlePly: true },
  bottom: { slug: BOTTOM_PATTERN_SLUG, params: {}, singlePly: true },
};
const FACE_FRAME_PROFILE: Record<FaceId, Partial<FrameSpec> & { seed: number }> = {
  front: { seed: 100, edge_gradient: 0.75, understory: 0.9, border_vine: 1.2, corner_fans: 1.1 },
  back: { seed: 101, edge_gradient: 1.0, understory: 0.6, border_vine: 0.9, corner_fans: 0.8 },
  top: { seed: 102, edge_gradient: 0.55, understory: 1.05, border_vine: 1.35, corner_fans: 1.25 },
  bottom: { seed: 103, edge_gradient: 0.9, understory: 0.75, border_vine: 1.0, corner_fans: 0.9 },
  left: { seed: 104, edge_gradient: 0.65, understory: 1.0, border_vine: 1.25, corner_fans: 1.15 },
  right: { seed: 105, edge_gradient: 0.85, understory: 0.8, border_vine: 1.05, corner_fans: 0.95 },
};

/**
 * PRODUCTION glass: 2.25 mm fused quartz per ply (the box is bonded, so the
 * wall is 4.5 mm and the optical parallax gap is one ply, 2250/1.4585 ≈
 * 1543 µm of paraxial air). n is the real fused-quartz index at d-line, not
 * the 1.46 round number the pre-production default carried.
 */
export function defaultGlassSpec(): GlassSpec {
  return { thickness_um: 2250.0, material: 'fused quartz', n: 1.4585 };
}

/**
 * PRODUCTION foil: 1/4" copper. A single 2.25 mm ply is a 2.25 mm edge, so the
 * tape leaves a 2.05 mm fold on each face — clear of the 3.6375 mm art rim.
 * (The bonded build needed 3/8" to wrap its 6.75 mm stepped edge.) Mirrors
 * backend boxes.PRODUCTION_TAPE_UM.
 */
export function defaultFoilSpec(): FoilSpec {
  return { tape_width_um: 6350.0, safety_um: 500.0, bead_um: 2000.0, finish: 'bright' };
}

export function defaultHingeSpec(): HingeSpec {
  return { style: 'tube', tube_od_um: 2400.0, rod_od_um: 1600.0, segments: 5, coverage: 0.8 };
}

export function defaultPlateSpec(
  patternSlug: string,
  seed = 1,
  frameOverrides: Partial<FrameSpec> = {},
  opts: { params?: Record<string, unknown>; singlePly?: boolean; carrierScaleMode?: 'gap' | 'fixed' } = {}
): PlateSpec {
  return {
    pattern_slug: patternSlug,
    pattern_params: { ...(opts.params ?? {}) },
    frame: defaultFrameSpec(seed, frameOverrides),
    glass: defaultGlassSpec(),
    single_ply: opts.singlePly ?? false,
    // Backend `plates.PlateSpec` dataclass defaults. Every one of these is
    // stamped over by `stampFaces` (cut dims, keep-out) or by defaultBoxSpec
    // (carrier pitch, carrier scale mode) before the spec is POSTed — but they
    // are the documented mirror, so they must be the backend's numbers and not
    // a second set that merely happens never to be read.
    width_um: 30000,
    height_um: 30000,
    weld_margin_um: 1000,
    back_margin_um: null,
    carrier_pitch_um: 22.0,
    carrier_scale_mode: opts.carrierScaleMode ?? 'gap',
    label: '',
  };
}

/**
 * PRODUCTION moire carrier, micrometres as fabricated: the period subtends
 * 0.75 arcmin at 300 mm so the lines are invisible in hand and only the beat
 * shows. Mirrors backend witness_geom.BOX_CARRIER_UM / boxes.PRODUCTION_CARRIER_UM;
 * the production faces run carrier_scale_mode 'fixed' so this is the literal pitch.
 */
export const PRODUCTION_CARRIER_UM = 65.5;

/**
 * The PRODUCTION art rim, micrometres, PINNED. Every face's gold starts
 * 3.6375 mm in from its edge — the number the 2026-09-15 plate was WRITTEN
 * with, chosen when the box was bonded (one ply + the interior foil fold) and
 * kept so the plate does not change now that the box is six single plies.
 * Mirrors backend boxes.PRODUCTION_ART_RIM_UM.
 */
export const PRODUCTION_ART_RIM_UM = 3637.5;

/**
 * The PRODUCTION box — 32 × 32 × 35 mm outer, SIX SINGLE 2.25 mm fused-quartz
 * plies butt-jointed with 1/4" foil (2026-09-16: no inner plies, no bonding),
 * which opens the interior to 27.5 × 27.5 × 30.5 mm for a 21 mm ring standing
 * in a 1 mm liner (backend boxes.RING_*). The art rim is PINNED, not derived
 * from the foil, because the plate is already written. MUST stay identical to
 * backend `boxes.default_box_spec()`: this is what the live preview POSTs and
 * what the fab bake ships.
 */
export function defaultBoxSpec(patternSlug: string = DEFAULT_PATTERN_SLUG): BoxSpec {
  const faces: Partial<Record<FaceId, PlateSpec>> = {};
  FACE_IDS.forEach((fid) => {
    const { seed, ...frameOverrides } = FACE_FRAME_PROFILE[fid];
    const plan = FACE_PLAN[fid];
    // The production plan (see FACE_PLAN): top = J+P monogram, front = the
    // Atlantic globe, left/right = the two halftone photos, back + bottom =
    // bare glass — every face one ply. A caller-supplied `patternSlug`
    // overrides only the FRONT face (themed override boxes).
    const slug = fid === 'front' ? patternSlug : plan.slug;
    // motif_scale / band_um mirror backend boxes.PRODUCTION_MOTIF_SCALE /
    // PRODUCTION_BAND_UM: one 2.4 mm band on every face, foliage at 0.68.
    faces[fid] = defaultPlateSpec(slug, seed, { ...frameOverrides, motif_scale: 0.68, band_um: 2400 }, {
      params: plan.params,
      singlePly: plan.singlePly,
      carrierScaleMode: 'fixed',
    });
  });
  const spec: BoxSpec = {
    width_um: 32000.0,
    depth_um: 32000.0,
    height_um: 35000.0,
    glass: defaultGlassSpec(),
    foil: defaultFoilSpec(),
    hinge: defaultHingeSpec(),
    faces,
    carrier_pitch_um: PRODUCTION_CARRIER_UM,
    bonded: false,
    art_rim_um: PRODUCTION_ART_RIM_UM,
    metal: 'gold',
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
  if (!r.ok) throw new Error(await errorDetail(r, 'Load box'));
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
  if (!r.ok) throw new Error(await errorDetail(r, 'Generate box'));
  return r.json();
}
