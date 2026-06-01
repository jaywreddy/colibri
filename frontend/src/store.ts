import { create } from 'zustand';
import type {
  BoxManifest,
  BoxSpec,
  FaceId,
  PatternDescriptor,
  PatternManifest,
  PlateSpec,
} from './api';
import { FACE_IDS, defaultBoxSpec } from './api';

export type Illumination = 'ambient' | 'laser' | 'backlight';
export type StudioMode = 'plate' | 'box';

type State = {
  // --- shared catalogue ---
  catalog: PatternDescriptor[];
  setCatalog: (c: PatternDescriptor[]) => void;

  // --- studio mode ---
  mode: StudioMode;
  setMode: (m: StudioMode) => void;

  // --- single-pattern (legacy / Plate mode) ---
  activeSlug: string | null;
  pendingSlug: string | null;
  manifest: PatternManifest | null;
  params: Record<string, unknown>;
  beginSelect: (slug: string) => void;
  selectPatternIfCurrent: (slug: string, manifest: PatternManifest) => boolean;
  selectPattern: (slug: string, manifest: PatternManifest) => void;
  setManifest: (m: PatternManifest) => void;
  setParams: (p: Record<string, unknown>) => void;
  patchParams: (p: Record<string, unknown>) => void;

  // --- Box mode ---
  boxSpec: BoxSpec;
  boxManifest: BoxManifest | null;
  selectedFaceId: FaceId;
  setBoxSpec: (b: BoxSpec) => void;
  patchBoxSpec: (patch: Partial<BoxSpec>) => void;
  setBoxManifest: (m: BoxManifest | null) => void;
  setSelectedFace: (id: FaceId) => void;
  patchFace: (faceId: FaceId, patch: Partial<PlateSpec>) => void;
  patchFaceFrame: (
    faceId: FaceId,
    patch: Partial<PlateSpec['frame']>
  ) => void;

  // --- viewer state (shared across modes) ---
  illumination: Illumination;
  laserColor: 'red' | 'green' | 'blue';
  lightAzimuthDeg: number;
  lightElevationDeg: number;
  tilt: [number, number];
  setIllumination: (i: Illumination) => void;
  setLaserColor: (c: 'red' | 'green' | 'blue') => void;
  setLight: (az: number, el: number) => void;
  setTilt: (t: [number, number]) => void;
};

const INITIAL_BOX_SLUG = 'wayuu-kanasu-moire';

export const useStore = create<State>((set, get) => ({
  catalog: [],
  mode: 'box',  // Box is the new default studio surface.
  activeSlug: null,
  pendingSlug: null,
  manifest: null,
  params: {},
  boxSpec: defaultBoxSpec(INITIAL_BOX_SLUG),
  boxManifest: null,
  selectedFaceId: 'front',
  illumination: 'ambient',
  laserColor: 'green',
  lightAzimuthDeg: 35,
  lightElevationDeg: 55,
  tilt: [0, 0],

  setCatalog: (catalog) => set({ catalog }),
  setMode: (mode) => set({ mode }),

  beginSelect: (slug) => set({ pendingSlug: slug }),
  selectPatternIfCurrent: (slug, manifest) => {
    if (get().pendingSlug !== slug) return false;
    set({
      activeSlug: slug,
      manifest,
      params: { ...(manifest.params as Record<string, unknown>) },
    });
    return true;
  },
  selectPattern: (slug, manifest) =>
    set({
      activeSlug: slug,
      pendingSlug: slug,
      manifest,
      params: { ...(manifest.params as Record<string, unknown>) },
    }),
  setManifest: (manifest) => set({ manifest }),
  setParams: (params) => set({ params }),
  patchParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),

  setBoxSpec: (boxSpec) => set({ boxSpec }),
  patchBoxSpec: (patch) =>
    set((s) => ({ boxSpec: { ...s.boxSpec, ...patch } })),
  setBoxManifest: (boxManifest) => set({ boxManifest }),
  setSelectedFace: (selectedFaceId) => set({ selectedFaceId }),
  patchFace: (faceId, patch) =>
    set((s) => {
      const cur = s.boxSpec.faces[faceId];
      if (!cur) return s;
      return {
        boxSpec: {
          ...s.boxSpec,
          faces: { ...s.boxSpec.faces, [faceId]: { ...cur, ...patch } },
        },
      };
    }),
  patchFaceFrame: (faceId, patch) =>
    set((s) => {
      const cur = s.boxSpec.faces[faceId];
      if (!cur) return s;
      return {
        boxSpec: {
          ...s.boxSpec,
          faces: {
            ...s.boxSpec.faces,
            [faceId]: { ...cur, frame: { ...cur.frame, ...patch } },
          },
        },
      };
    }),

  setIllumination: (illumination) => set({ illumination }),
  setLaserColor: (laserColor) => set({ laserColor }),
  setLight: (lightAzimuthDeg, lightElevationDeg) =>
    set({ lightAzimuthDeg, lightElevationDeg }),
  setTilt: (tilt) => set({ tilt }),
}));

// Re-export FACE_IDS for convenience.
export { FACE_IDS };
