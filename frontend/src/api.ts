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
 * RECIPE_NAMES in backend/app/patterns/base.py. `stylized_amplitude` is the
 * back-compat default for patterns that haven't been upgraded yet.
 */
export type RenderRecipe =
  | 'iridescent_grating'
  | 'stereo_lenticular'
  | 'moire_interactive'
  | 'near_field_carpet'
  | 'far_field_hologram'
  | 'stylized_amplitude';

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
  // Present in new manifests; older on-disk manifests (pre-Phase A) omit it
  // and must fall back to 'stylized_amplitude' at read time.
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
  iridescent_grating: 0,
  stereo_lenticular: 1,
  moire_interactive: 2,
  near_field_carpet: 3,
  far_field_hologram: 4,
  stylized_amplitude: 5,
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
    // Avoid passing an explicit `undefined` so callers with no init match
    // existing unit-test expectations (`fetch('/patterns')`).
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

/**
 * Near-field propagation carpet — the artifact for the `near_field_carpet`
 * recipe (tairona-talbot, muzo-emerald-zone). Returns a vertical atlas of
 * `n_slices` 2D intensity tiles from z_min to z_max.
 */
export type CarpetLayout = 'tiles' | 'stripe';

export async function fetchCarpet(
  slug: string,
  variant: string,
  opts: {
    wavelength_um?: number;
    z_min_um?: number;
    z_max_um?: number;
    n_slices?: number;
    downsample?: number;
    tile_size?: number;
    layout?: CarpetLayout;
    signal?: AbortSignal;
  } = {}
): Promise<{
  atlas_png: string;
  rows: number;
  cols: number;
  tile: [number, number];
  layout: CarpetLayout;
  z_min_um: number;
  z_max_um: number;
  wavelength_um: number;
  cached: boolean;
}> {
  const {
    wavelength_um = 0.55,
    z_min_um = 0.0,
    z_max_um = 4000.0,
    n_slices = 48,
    downsample = 8,
    tile_size = 128,
    layout = 'tiles',
    signal,
  } = opts;
  const r = await tracedFetch('/sim/carpet', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      slug,
      variant,
      wavelength_um,
      z_min_um,
      z_max_um,
      n_slices,
      downsample,
      tile_size,
      layout,
    }),
    signal,
  });
  if (!r.ok) throw new Error(`fetchCarpet: ${r.status} ${await r.text()}`);
  return r.json();
}

/**
 * Merged-RGB Fraunhofer reconstruction for the `far_field_hologram` recipe
 * (colibri-hologram, meridian-speckle). Returns a single PNG whose RGB
 * channels are the three-wavelength reconstructions — what you would see
 * projected on a screen under white coherent illumination.
 */
export async function fetchFarfield(
  slug: string,
  variant: string,
  opts: {
    wavelengths_um?: [number, number, number];
    n_angles?: number;
    max_angle_deg?: number;
    carrier_cells?: number;
    signal?: AbortSignal;
  } = {}
): Promise<{
  farfield_png: string;
  shape: [number, number];
  wavelengths_um: [number, number, number];
  cached: boolean;
}> {
  const {
    wavelengths_um = [0.65, 0.55, 0.45],
    n_angles = 256,
    max_angle_deg = 30.0,
    carrier_cells = 0,
    signal,
  } = opts;
  const r = await tracedFetch('/sim/farfield', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      slug,
      variant,
      wavelengths_um,
      n_angles,
      max_angle_deg,
      carrier_cells,
    }),
    signal,
  });
  if (!r.ok) throw new Error(`fetchFarfield: ${r.status} ${await r.text()}`);
  return r.json();
}
