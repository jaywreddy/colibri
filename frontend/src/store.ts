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
import { FACE_IDS, defaultBoxSpec } from './api';
import { stampFaces } from './assembly';
import { log } from './logger';

/**
 * Read-only picker thumbnail route (see loadThumbnails). Doubles as the <img>
 * src once the probe says 200 — the second request revalidates against the
 * FileResponse's ETag instead of transferring the PNG again, which keeps this
 * free of object-URL lifetime management.
 */
const thumbnailUrl = (slug: string): string =>
  `/patterns/${encodeURIComponent(slug)}/thumbnail`;

/** Controller of the in-flight sweep, so cancelThumbnails can drop it. */
let thumbnailSweep: AbortController | null = null;

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
   * Slugs whose last thumbnail fetch failed (slug -> error message). Kept
   * OUTSIDE `thumbnails` because a failed slug's marker there is deleted so
   * the next loadThumbnails retries it; this map is what lets the picker draw
   * a retry tile instead of a shimmer that would spin forever.
   */
  thumbnailErrors: Record<string, string>;
  /**
   * Lazily probe GET /patterns/{slug}/thumbnail for every catalog slug that
   * has no thumbnail yet (sequentially). That route is READ-ONLY: it serves
   * the default variant's cached PNG or 404s, so this sweep never triggers a
   * generate — the picker used to hit /{slug}/default, which made opening it
   * on a cold cache sixteen full pattern generates holding the very per-slot
   * locks a box regen wants (CLAUDE.md's single-heavy-compute rule). A 404
   * lands in `thumbnailErrors`, i.e. the existing retry tile, and the tile
   * fills in as soon as the variant is materialized for a real reason.
   *
   * Safe to call on every picker open: loaded and in-flight (null-marked)
   * slugs are skipped, and failed slugs are cleared so the next call retries
   * them. Pass a signal (or call `cancelThumbnails`) to drop a sweep whose
   * consumer went away — the remaining slugs stay unmarked so a later mount
   * re-probes them.
   */
  loadThumbnails: (opts?: { signal?: AbortSignal }) => Promise<void>;
  /** Abort the in-flight thumbnail sweep (picker/panel unmount). */
  cancelThumbnails: () => void;

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
  /**
   * Pattern Scale: uniform multiplier on every procedural preview period on
   * both plate planes (frame carrier + louvre, centerpiece switch, water comb +
   * ripple lanes, barrier-interlace lanes). 1 = exact fab dimensions (sub-pixel
   * at default zoom — the fine structure resolves when you raise this or zoom
   * in). Frontend-only; no backend regen. Default 1×.
   */
  patternScale: number;
  setPatternScale: (s: number) => void;

  // --- illumination / viewer state ---
  illumination: Illumination;
  laserColor: 'red' | 'green' | 'blue';
  lightAzimuthDeg: number;
  lightElevationDeg: number;
  setIllumination: (i: Illumination) => void;
  setLaserColor: (c: 'red' | 'green' | 'blue') => void;
  setLight: (az: number, el: number) => void;

  /**
   * Presentation props: velvet cushion + a real-dimensioned ring (size-7 band)
   * inside the assembled box. Pure display props — no fab meaning, and the
   * ring's FIXED real size doubles as an honest fit check against the box
   * interior. Default OFF so the pixel-metric harnesses see the same scene
   * they always did.
   */
  showRing: boolean;
  setShowRing: (on: boolean) => void;
  /** Scene backdrop preset. Backlight illumination overrides it with the
   * light-table field while active. */
  backdrop: 'studio' | 'velvet' | 'daylight';
  setBackdrop: (b: 'studio' | 'velvet' | 'daylight') => void;
  /**
   * Head-on tilt inspection of the selected face: the camera locks onto the
   * face normal, dragging rocks the view ±INSPECT_MAX_DEG, and a HUD reads out
   * tilt + back-layer parallax shift with the face's effect-peak angles.
   */
  inspectMode: boolean;
  setInspectMode: (on: boolean) => void;
  /** Tilt-progression view: per-face supersampled tilt sweeps + eye verdict. */
  progressionOpen: boolean;
  /** Collage view: every catalogue pattern composited across a fan of angles. */
  collageOpen: boolean;
  setProgressionOpen: (open: boolean) => void;
  setCollageOpen: (open: boolean) => void;
};

export const useStore = create<State>((set, get) => ({
  catalog: [],
  thumbnails: {},
  thumbnailErrors: {},
  boxSpec: defaultBoxSpec(),
  boxManifest: null,
  selectedFaceId: 'front',
  lidTargetDeg: 0,
  layout: 'assembled',
  autoRotate: false,
  labOpen: false,
  labSlug: null,
  patternScale: 1,
  illumination: 'ambient',
  laserColor: 'green',
  lightAzimuthDeg: 35,
  lightElevationDeg: 55,
  showRing: false,
  backdrop: 'studio',
  inspectMode: false,
  progressionOpen: false,
  collageOpen: false,

  setCatalog: (catalog) => set({ catalog }),

  loadThumbnails: async (opts = {}) => {
    const { catalog, thumbnails } = get();
    const missing = catalog.filter((c) => thumbnails[c.slug] === undefined);
    if (missing.length === 0) return;
    if (opts.signal?.aborted) return;
    const controller = new AbortController();
    thumbnailSweep = controller;
    opts.signal?.addEventListener('abort', () => controller.abort(), { once: true });
    // Mark as loading so the picker shows shimmers (and a re-open while
    // probes are in flight doesn't start a duplicate run). Any earlier
    // failure marker goes away here — this run IS the retry.
    set((s) => {
      const thumbnailErrors = { ...s.thumbnailErrors };
      for (const c of missing) delete thumbnailErrors[c.slug];
      return {
        thumbnails: {
          ...s.thumbnails,
          ...Object.fromEntries(missing.map((c) => [c.slug, null])),
        },
        thumbnailErrors,
      };
    });
    /** Drop the loading marker (optionally with a retry-tile message). */
    const forget = (slug: string, message?: string) =>
      set((s) => {
        const next = { ...s.thumbnails };
        delete next[slug];
        const thumbnailErrors = { ...s.thumbnailErrors };
        if (message === undefined) delete thumbnailErrors[slug];
        else thumbnailErrors[slug] = message;
        return { thumbnails: next, thumbnailErrors };
      });
    for (const c of missing) {
      if (controller.signal.aborted) {
        // Unmarked, not failed: the next mount re-probes instead of showing a
        // retry tile for work nobody was waiting on.
        forget(c.slug);
        continue;
      }
      try {
        const url = thumbnailUrl(c.slug);
        const r = await fetch(url, { signal: controller.signal });
        if (!r.ok) {
          throw new Error(
            r.status === 404
              ? 'no cached preview yet — it appears once this pattern is generated'
              : `HTTP ${r.status}${r.statusText ? ` ${r.statusText}` : ''}`
          );
        }
        set((s) => ({ thumbnails: { ...s.thumbnails, [c.slug]: url } }));
        log('thumbnail_loaded', { slug: c.slug });
      } catch (e) {
        const err = e as Error;
        if (err.name === 'AbortError') {
          forget(c.slug);
          continue;
        }
        // Drop the loading marker so a later open retries this slug, and
        // record the failure so the picker can show a retry tile meanwhile.
        forget(c.slug, err.message);
        log('thumbnail_load_failed', { slug: c.slug, error: err.message });
      }
    }
    if (thumbnailSweep === controller) thumbnailSweep = null;
  },

  cancelThumbnails: () => {
    thumbnailSweep?.abort();
    thumbnailSweep = null;
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
  setPatternScale: (patternScale) => set({ patternScale }),

  setIllumination: (illumination) => set({ illumination }),
  setShowRing: (showRing) => set({ showRing }),
  setBackdrop: (backdrop) => set({ backdrop }),
  setInspectMode: (inspectMode) => set({ inspectMode }),
  setProgressionOpen: (progressionOpen) => set({ progressionOpen }),
  setCollageOpen: (collageOpen) => set({ collageOpen }),
  setLaserColor: (laserColor) => set({ laserColor }),
  setLight: (lightAzimuthDeg, lightElevationDeg) =>
    set({ lightAzimuthDeg, lightElevationDeg }),
}));

// Re-export FACE_IDS for convenience.
export { FACE_IDS };
