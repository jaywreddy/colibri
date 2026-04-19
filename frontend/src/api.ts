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

export type PatternDescriptor = {
  slug: string;
  name: string;
  description: string;
  tags: string[];
  tier: 1 | 2 | 3;
  theme: 'Colombia' | 'Global Travel';
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
  files: {
    front_png: string;
    back_png: string;
    front_svg: string;
    back_svg: string;
    thumbnail: string;
  };
};

export async function listPatterns(): Promise<PatternDescriptor[]> {
  const r = await fetch('/patterns');
  if (!r.ok) throw new Error(`listPatterns: ${r.status}`);
  return r.json();
}

export async function getDefault(slug: string): Promise<PatternManifest> {
  const r = await fetch(`/patterns/${slug}/default`);
  if (!r.ok) throw new Error(`getDefault(${slug}): ${r.status}`);
  return r.json();
}

export async function generatePattern(
  slug: string,
  params: Record<string, unknown>
): Promise<PatternManifest> {
  const r = await fetch('/patterns/generate', {
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
  const r = await fetch('/sim/fft', {
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
  const r = await fetch('/sim/propagate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, variant, wavelengths_um, view_angles_deg }),
  });
  if (!r.ok) throw new Error(`propagateSim: ${r.status} ${await r.text()}`);
  return r.json();
}
