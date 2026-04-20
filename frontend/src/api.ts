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

export async function fftSim(
  slug: string,
  variant: string,
  wavelengths_um: number[] = [0.65, 0.55, 0.45]
): Promise<{ atlas_png: string }> {
  const r = await tracedFetch('/sim/fft', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, variant, wavelengths_um }),
  });
  if (!r.ok) throw new Error(`fftSim: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function propagateSim(
  slug: string,
  variant: string,
  wavelengths_um: number[] = [0.65, 0.55, 0.45],
  view_angles_deg: number[] = [-15, 0, 15]
): Promise<{
  atlas_png: string;
  rows: number;
  cols: number;
  tile: [number, number];
  view_angles_deg: number[];
  wavelengths_um: number[];
  cached: boolean;
}> {
  const r = await tracedFetch('/sim/propagate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, variant, wavelengths_um, view_angles_deg }),
  });
  if (!r.ok) throw new Error(`propagateSim: ${r.status} ${await r.text()}`);
  return r.json();
}
