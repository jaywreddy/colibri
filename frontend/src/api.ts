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
 *
 * Id 2 (phase_shift_overlay) is RETIRED with zero catalog users — its
 * two-image front/back phase split could never switch under honest parallax
 * (the front mask does not move with tilt). The numeric hole at 2 is
 * intentional: ids 0/1/3 are stable and must never be renumbered.
 */
export type RenderRecipe =
  | 'stereo_lenticular'
  | 'moire_interactive'
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
    /** Per-litho-metal chips (gold | chrome | chrome-ar). Absent on manifests
     * written before per-metal thumbnails existed — fall back to `thumbnail`. */
    thumbnails?: Record<string, string>;
  };
};

export const RECIPE_IDS: Record<RenderRecipe, number> = {
  stereo_lenticular: 0,
  moire_interactive: 1,
  // 2 = retired phase_shift_overlay — hole kept so 3 never renumbers.
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
 * error banner, the Pattern Lab error strip, or a picker retry tile.
 *
 * FastAPI reports every failure as `{"detail": "..."}` and our handlers put
 * the actionable sentence there ("regenerate the box and export again"), so
 * that field is what the user needs — not the raw body, which for a proxy
 * error or a truncated stream is an HTML page. Truncated: the banner is one
 * line, and the full text still reaches the `title` tooltip.
 *
 * Every thrower whose message can reach a HUMAN routes through here: box
 * load/generate, fab export, and (as of this change) the two pattern endpoints
 * behind the Pattern Lab and the picker thumbnails. Those two used to build
 * their own strings — `getDefault(slug): 404` and `generatePattern: 400
 * {"detail":"…"}`, i.e. a status code plus the RAW body — which is the whole
 * reason PatternLab carries a `readableError` brace-scanner: the carefully
 * worded backend refusals (the 400k lattice budget, the ParamSpec range check,
 * the litho floor) surfaced as JSON repr noise. With the detail extracted here
 * `readableError` finds no brace and degrades to a passthrough; keep it that
 * way when adding an endpoint.
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

export async function getDefault(
  slug: string,
  opts: { signal?: AbortSignal } = {}
): Promise<PatternManifest> {
  const r = await tracedFetch(`/patterns/${slug}/default`, { signal: opts.signal });
  // The slug is in the prefix because this is also the pattern-picker
  // thumbnail fetch: a failed tile's tooltip has to name which pattern failed.
  if (!r.ok) throw new Error(await errorDetail(r, `Load pattern ${slug}`));
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
  // 400s here are the ones the user most needs to READ verbatim: the lattice
  // budget refusal, the ParamSpec out-of-range rejection and the litho-floor
  // refusal all put an actionable sentence in `detail`.
  if (!r.ok) throw new Error(await errorDetail(r, `Regenerate ${slug}`));
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
  /** Per-face carrier scaling policy vs the plate's real glass: 'gap'
   * (default) scales the fabricated carrier family with the paraxial gap t/n,
   * preserving the designed reveal tilt on any stock (no-op at the 500 µm
   * baseline); 'fixed' keeps the literal pitch — on thick stock the reveals
   * compress into sub-degree refraction shimmer. Barrier switch periods scale
   * with the gap regardless. */
  carrier_scale_mode?: 'gap' | 'fixed';
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
    /** Per-litho-metal chips (gold | chrome | chrome-ar). Absent on manifests
     * written before per-metal thumbnails existed — fall back to `thumbnail`. */
    thumbnails?: Record<string, string>;
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
   * nested-shell math. Default false (classic single double-side plate). */
  bonded?: boolean;
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
    carrier_scale_mode: 'gap',
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
    bonded: false,
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

// -----------------------------------------------------------------------------
// Fab export — the JOB api (api/export.py). A cold box export is six plate
// composes, six SVG bakes and six merged-region klayout DRC heals, minutes end
// to end, so the build runs in a SUBPROCESS the client polls:
//
//   POST /export/box/{box_id}/fab/start  -> 202 {job_id, poll_url, download_url}
//   GET  /export/jobs/{job_id}           -> {status, progress, error, ...}
//   GET  /export/jobs/{job_id}/fab.zip   -> the finished archive
//
// The legacy synchronous `GET /export/box/{box_id}/fab.zip` is deliberately NOT
// wrapped here: it holds the connection silently for the whole build, which is
// exactly the "looks hung" UX the job routes exist to replace. It stays on the
// backend for bench tools only.
// -----------------------------------------------------------------------------

/**
 * One progress snapshot published by the export worker (export_job.py::Progress).
 *
 * `phase` is one of export_job.PHASES ('resolve faces', 'rebuild plate',
 * 'svg bake', 'fine mask (cached)', 'fine mask (compose)', 'fine mask (DRC
 * heal)', 'zip', plus the 'starting'/'done'/'failed' bookends) — treat it as an
 * open string, the worker owns the list. `updated_at` is a monotonic counter,
 * NOT a clock: its only job is "is this newer than what I last saw".
 */
export type ExportJobProgress = {
  phase: string;
  /** Face the phase is scoped to, or null for a whole-archive phase. */
  face: string | null;
  faces_done: number;
  faces_total: number;
  detail: string;
  updated_at: number;
  elapsed_s?: number;
  error?: string;
};

export type ExportJobState = 'running' | 'done' | 'failed';

/** GET /export/jobs/{job_id} — merges worker progress with process liveness. */
export type ExportJobStatus = {
  job_id: string;
  box_id: string;
  status: ExportJobState;
  progress: ExportJobProgress | null;
  /** Set only on `failed`; carries the worker's own diagnosis when it has one. */
  error: string | null;
  returncode: number | null;
  /** Present only once `status === 'done'`. */
  download_url: string | null;
};

/** POST /export/box/{box_id}/fab/start — 202. */
export type ExportJobStart = {
  job_id: string;
  box_id: string;
  status: string;
  progress: ExportJobProgress | null;
  poll_url: string;
  download_url: string;
};

/**
 * Exactly one export may build at a time across the host (CLAUDE.md), so a
 * start that collides answers 429 instead of queueing behind minutes of
 * compute. The backend's `detail` names the job that holds the slot, which is
 * the only handle the API gives us on it — there is no "list jobs" route — so
 * `runningJobId`/`runningBoxId` are parsed out of that sentence and are null
 * when the wording doesn't carry them (the in-process-slot variant doesn't).
 *
 * They are a handle for WAITING OUT that job (poll it, show its phase, then
 * start your own build — App.tsx::waitOutRunningExport), never for downloading
 * its archive: a job that started before the last regen can carry a design the
 * screen no longer shows, even for the same box id, and that is the
 * unrecoverable fab error the staleness gate exists to prevent.
 */
export class ExportBusyError extends Error {
  readonly status = 429;
  readonly runningJobId: string | null;
  readonly runningBoxId: string | null;

  constructor(message: string, jobId: string | null, boxId: string | null) {
    super(message);
    this.name = 'ExportBusyError';
    this.runningJobId = jobId;
    this.runningBoxId = boxId;
  }
}

/**
 * Both 429 wordings that name a job (export_job.py::start_job and
 * ::claim_slot): "export job <id> (box <box>) is still running|building — …".
 * A miss is not an error, just an unresumable busy state.
 */
const BUSY_JOB_RE = /export job ([A-Za-z0-9_-]{1,64}) \(box ([^)]+)\) is still (?:running|building)/i;

/**
 * Start the out-of-process fab build for one box. Resolves as soon as the
 * worker is spawned — poll `getExportJob(job_id)` for the phase and download
 * the archive when the status flips to `done`.
 *
 * Throws `ExportBusyError` on 429 (an export is already building).
 */
export async function startBoxExport(
  boxId: string,
  opts: { signal?: AbortSignal } = {}
): Promise<ExportJobStart> {
  const r = await tracedFetch(`/export/box/${encodeURIComponent(boxId)}/fab/start`, {
    method: 'POST',
    signal: opts.signal,
  });
  if (!r.ok) {
    // One body read for both the message and the job handle inside it.
    const message = await errorDetail(r, 'Fab export');
    if (r.status === 429) {
      const m = BUSY_JOB_RE.exec(message);
      throw new ExportBusyError(message, m?.[1] ?? null, m?.[2] ?? null);
    }
    throw new Error(message);
  }
  return r.json();
}

/**
 * Poll one export job. Cheap and side-effect free on the server (it reads a
 * progress file and one `proc.poll()`), so a ~1 s interval is fine and takes no
 * heavy-compute slot.
 */
export async function getExportJob(
  jobId: string,
  opts: { signal?: AbortSignal } = {}
): Promise<ExportJobStatus> {
  const r = await tracedFetch(`/export/jobs/${encodeURIComponent(jobId)}`, {
    signal: opts.signal,
  });
  // Job ids live in the server process only, so a 404 here means the backend
  // restarted mid-export — the detail says to start a new one.
  if (!r.ok) throw new Error(await errorDetail(r, 'Fab export'));
  return r.json();
}

/** URL of a finished job's archive. 409s until its status is `done`. */
export function exportJobZipUrl(jobId: string): string {
  return `/export/jobs/${encodeURIComponent(jobId)}/fab.zip`;
}

/**
 * The finished archive as a Blob, so the caller can name the download and
 * report a failure in the UI instead of handing the URL to the browser's
 * download shelf (where a 409/404 becomes an unexplained failed download).
 */
export async function exportJobZip(
  jobId: string,
  opts: { signal?: AbortSignal } = {}
): Promise<Blob> {
  const r = await tracedFetch(exportJobZipUrl(jobId), { signal: opts.signal });
  if (!r.ok) throw new Error(await errorDetail(r, 'Fab export'));
  return r.blob();
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
