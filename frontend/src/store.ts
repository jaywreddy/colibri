import { create } from 'zustand';
import type {
  BoxManifest,
  BoxSpec,
  FaceId,
  FoilSpec,
  GlassSpec,
  HingeSpec,
  PatternDescriptor,
  PlateSpec,
} from './api';
import { FACE_IDS, defaultBoxSpec, getDefault } from './api';
import { stampFaces } from './assembly';
import { log } from './logger';

export type Illumination = 'ambient' | 'laser' | 'backlight';
export type SceneLayout = 'assembled' | 'flat';

export const LID_MAX_DEG = 120;

const clampLid = (deg: number): number => Math.max(0, Math.min(LID_MAX_DEG, deg));

type State = {
  // --- pattern catalogue (shared by all face editors) ---
  catalog: PatternDescriptor[];
  setCatalog: (c: PatternDescriptor[]) => void;

  // --- pattern picker thumbnails (slug -> URL; null = loading) ---
  thumbnails: Record<string, string | null>;
  /**
   * Lazily fetch GET /patterns/{slug}/default for every catalog slug that
   * has no thumbnail yet (sequentially — cold patterns materialize ~2 s
   * server-side, then cache). Safe to call on every picker open: loaded and
   * in-flight (null-marked) slugs are skipped, and failed slugs are cleared
   * so the next call retries them.
   */
  loadThumbnails: () => Promise<void>;

  // --- the box ---
  boxSpec: BoxSpec;
  boxManifest: BoxManifest | null;
  setBoxSpec: (b: BoxSpec) => void;
  patchBoxSpec: (patch: Partial<BoxSpec>) => void;
  patchGlass: (patch: Partial<GlassSpec>) => void;
  patchFoil: (patch: Partial<FoilSpec>) => void;
  patchHinge: (patch: Partial<HingeSpec>) => void;
  setBoxManifest: (m: BoxManifest | null) => void;
  patchFace: (faceId: FaceId, patch: Partial<PlateSpec>) => void;
  patchFaceFrame: (faceId: FaceId, patch: Partial<PlateSpec['frame']>) => void;
  /**
   * Copy one face's design (pattern slug + params + frame dials) onto all
   * six faces, PRESERVING each face's own frame seed so the faces stay
   * individual variations of the same design.
   */
  applyFaceToAll: (sourceFaceId: FaceId) => void;
  /** Randomize one face's frame seed. */
  shuffleFaceSeed: (faceId: FaceId) => void;

  // --- studio UI state ---
  selectedFaceId: FaceId;
  setSelectedFace: (id: FaceId) => void;
  /** Target lid opening angle in degrees, 0..120. The scene damps toward it. */
  lidTargetDeg: number;
  setLidTargetDeg: (deg: number) => void;
  layout: SceneLayout;
  setLayout: (l: SceneLayout) => void;
  /** Slow OrbitControls turntable. Default OFF. */
  autoRotate: boolean;
  setAutoRotate: (on: boolean) => void;
  /** Pattern Lab — 2D dual-layer preview overlay. */
  labOpen: boolean;
  setLabOpen: (open: boolean) => void;
  /** Pattern under study in the lab; null = follow the selected face. */
  labSlug: string | null;
  setLabSlug: (slug: string | null) => void;

  // --- illumination / viewer state ---
  illumination: Illumination;
  laserColor: 'red' | 'green' | 'blue';
  lightAzimuthDeg: number;
  lightElevationDeg: number;
  setIllumination: (i: Illumination) => void;
  setLaserColor: (c: 'red' | 'green' | 'blue') => void;
  setLight: (az: number, el: number) => void;
};

export const useStore = create<State>((set, get) => ({
  catalog: [],
  thumbnails: {},
  boxSpec: defaultBoxSpec(),
  boxManifest: null,
  selectedFaceId: 'front',
  lidTargetDeg: 0,
  layout: 'assembled',
  autoRotate: false,
  labOpen: false,
  labSlug: null,
  illumination: 'ambient',
  laserColor: 'green',
  lightAzimuthDeg: 35,
  lightElevationDeg: 55,

  setCatalog: (catalog) => set({ catalog }),

  loadThumbnails: async () => {
    const { catalog, thumbnails } = get();
    const missing = catalog.filter((c) => thumbnails[c.slug] === undefined);
    if (missing.length === 0) return;
    // Mark as loading so the picker shows shimmers (and a re-open while
    // fetches are in flight doesn't start a duplicate run).
    set((s) => ({
      thumbnails: {
        ...s.thumbnails,
        ...Object.fromEntries(missing.map((c) => [c.slug, null])),
      },
    }));
    for (const c of missing) {
      try {
        const m = await getDefault(c.slug);
        set((s) => ({ thumbnails: { ...s.thumbnails, [c.slug]: m.files.thumbnail } }));
        log('thumbnail_loaded', { slug: c.slug });
      } catch (e) {
        // Drop the loading marker so a later open retries this slug.
        set((s) => {
          const next = { ...s.thumbnails };
          delete next[c.slug];
          return { thumbnails: next };
        });
        log('thumbnail_load_failed', { slug: c.slug, error: (e as Error).message });
      }
    }
  },

  // Geometry-affecting patches re-stamp the per-face plate specs (glass,
  // cut dims, keep-out) so the local spec always matches what backend
  // normalization would produce.
  setBoxSpec: (boxSpec) => set({ boxSpec: stampFaces(boxSpec) }),
  patchBoxSpec: (patch) =>
    set((s) => ({ boxSpec: stampFaces({ ...s.boxSpec, ...patch }) })),
  patchGlass: (patch) =>
    set((s) => ({
      boxSpec: stampFaces({ ...s.boxSpec, glass: { ...s.boxSpec.glass, ...patch } }),
    })),
  patchFoil: (patch) =>
    set((s) => ({
      boxSpec: stampFaces({ ...s.boxSpec, foil: { ...s.boxSpec.foil, ...patch } }),
    })),
  // Hinge never affects masks or cut dims — no re-stamp needed.
  patchHinge: (patch) =>
    set((s) => ({ boxSpec: { ...s.boxSpec, hinge: { ...s.boxSpec.hinge, ...patch } } })),
  setBoxManifest: (boxManifest) => set({ boxManifest }),
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
  applyFaceToAll: (sourceFaceId) =>
    set((s) => {
      const src = s.boxSpec.faces[sourceFaceId];
      if (!src) return s;
      const faces: Partial<Record<FaceId, PlateSpec>> = {};
      for (const fid of FACE_IDS) {
        const cur = s.boxSpec.faces[fid];
        if (!cur) continue;
        faces[fid] = {
          ...cur,
          pattern_slug: src.pattern_slug,
          pattern_params: { ...src.pattern_params },
          // Copy the frame dials but KEEP this face's own seed.
          frame: { ...src.frame, seed: cur.frame.seed },
        };
      }
      return { boxSpec: { ...s.boxSpec, faces } };
    }),
  shuffleFaceSeed: (faceId) =>
    set((s) => {
      const cur = s.boxSpec.faces[faceId];
      if (!cur) return s;
      const seed = Math.floor(Math.random() * 0x7fffffff);
      return {
        boxSpec: {
          ...s.boxSpec,
          faces: {
            ...s.boxSpec.faces,
            [faceId]: { ...cur, frame: { ...cur.frame, seed } },
          },
        },
      };
    }),

  setSelectedFace: (selectedFaceId) => set({ selectedFaceId }),
  setLidTargetDeg: (deg) => set({ lidTargetDeg: clampLid(deg) }),
  setLayout: (layout) => set({ layout }),
  setAutoRotate: (autoRotate) => set({ autoRotate }),
  setLabOpen: (labOpen) => set({ labOpen }),
  setLabSlug: (labSlug) => set({ labSlug }),

  setIllumination: (illumination) => set({ illumination }),
  setLaserColor: (laserColor) => set({ laserColor }),
  setLight: (lightAzimuthDeg, lightElevationDeg) =>
    set({ lightAzimuthDeg, lightElevationDeg }),
}));

// Re-export FACE_IDS for convenience.
export { FACE_IDS };
