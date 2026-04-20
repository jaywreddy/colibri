import { create } from 'zustand';
import type { PatternDescriptor, PatternManifest } from './api';

export type Illumination = 'ambient' | 'laser' | 'backlight';
export type EngineTier = 'stylized' | 'fraunhofer' | 'waveprop';

type State = {
  catalog: PatternDescriptor[];
  activeSlug: string | null;
  /** Slug the user most recently clicked — selection races use this to ignore
   * late-arriving responses for a slug that's no longer wanted. */
  pendingSlug: string | null;
  manifest: PatternManifest | null;
  params: Record<string, unknown>;
  illumination: Illumination;
  laserColor: 'red' | 'green' | 'blue';
  lightAzimuthDeg: number;
  lightElevationDeg: number;
  tilt: [number, number]; // deg X, deg Y — driven by orbit controls
  engine: EngineTier;
  fftAtlasUrl: string | null;

  setCatalog: (c: PatternDescriptor[]) => void;
  /** Mark `slug` as the user's latest intent. Call this BEFORE kicking off a
   * fetch so stale responses can be discarded on arrival. */
  beginSelect: (slug: string) => void;
  /** Commit a selection only if `slug` still matches `pendingSlug`. Returns
   * `true` if the write happened, `false` if the response was stale. */
  selectPatternIfCurrent: (slug: string, manifest: PatternManifest) => boolean;
  /** Unconditional select — useful for tests and for the initial auto-load. */
  selectPattern: (slug: string, manifest: PatternManifest) => void;
  setManifest: (m: PatternManifest) => void;
  setParams: (p: Record<string, unknown>) => void;
  patchParams: (p: Record<string, unknown>) => void;
  setIllumination: (i: Illumination) => void;
  setLaserColor: (c: 'red' | 'green' | 'blue') => void;
  setLight: (az: number, el: number) => void;
  setTilt: (t: [number, number]) => void;
  setEngine: (e: EngineTier) => void;
  setFftAtlas: (url: string | null) => void;
};

export const useStore = create<State>((set, get) => ({
  catalog: [],
  activeSlug: null,
  pendingSlug: null,
  manifest: null,
  params: {},
  illumination: 'ambient',
  laserColor: 'green',
  lightAzimuthDeg: 35,
  lightElevationDeg: 55,
  tilt: [0, 0],
  engine: 'stylized',
  fftAtlasUrl: null,

  setCatalog: (catalog) => set({ catalog }),
  beginSelect: (slug) => set({ pendingSlug: slug }),
  selectPatternIfCurrent: (slug, manifest) => {
    if (get().pendingSlug !== slug) return false;
    set({
      activeSlug: slug,
      manifest,
      params: { ...(manifest.params as Record<string, unknown>) },
      fftAtlasUrl: null,
    });
    return true;
  },
  selectPattern: (slug, manifest) =>
    set({
      activeSlug: slug,
      pendingSlug: slug,
      manifest,
      params: { ...(manifest.params as Record<string, unknown>) },
      fftAtlasUrl: null,
    }),
  setManifest: (manifest) => set({ manifest, fftAtlasUrl: null }),
  setParams: (params) => set({ params }),
  patchParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),
  setIllumination: (illumination) => set({ illumination }),
  setLaserColor: (laserColor) => set({ laserColor }),
  setLight: (lightAzimuthDeg, lightElevationDeg) =>
    set({ lightAzimuthDeg, lightElevationDeg }),
  setTilt: (tilt) => set({ tilt }),
  setEngine: (engine) => set({ engine }),
  setFftAtlas: (fftAtlasUrl) => set({ fftAtlasUrl }),
}));
