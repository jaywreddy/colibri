/**
 * Scene catalog for the image-based visual-verification harness.
 *
 * Each pattern (keyed by slug) owns a small list of scenes that elicit its
 * physics signature. The Playwright driver (visualSignatures.spec.ts) reads
 * this catalog and captures one PNG + metadata sidecar per scene. The vision
 * verifier (tools/visual_verifier.py) reads the same catalog to build per-
 * scene prompts, so there is one source of truth.
 *
 * Keep the English descriptions concise — the vision model is prompted with
 * them verbatim and longer prose drifts into ambiguity.
 */

export type Illumination = 'ambient' | 'laser' | 'backlight';
export type LaserColor = 'red' | 'green' | 'blue';
export type Quadrant = 'ul' | 'ur' | 'll' | 'lr';

export type SceneSetup = {
  illumination?: Illumination;
  laserColor?: LaserColor;
  cameraAzEl?: [number, number]; // degrees [azimuth, elevation]
  lightAz?: number;
  lightEl?: number;
  settleMs?: number; // RAF settling after setup; default 500
};

export type SceneNativeChecks = {
  canvasNotBlank?: boolean;
  plateQuadrantDominance?: { q: Quadrant; minRatio: number };
  logEvents?: string[];
  uniformEquals?: Record<string, number | string | boolean>;
  /** plate-pixel luminance must differ by >= minDelta between this scene and the named ref scene */
  pixelDeltaVsRef?: { refScene: string; minDelta: number; samplePoints?: [number, number][] };
};

export type VisualScene = {
  name: string;
  setup: SceneSetup;
  /** If true, a FAIL on this scene fails the CI gate. */
  required?: boolean;
  expect: {
    claim: string;
    plateSignature: string;
    failModes: string[];
    native?: SceneNativeChecks;
  };
};

const defaultSettle: Partial<SceneSetup> = { settleMs: 500 };

export const CATALOG: Record<string, VisualScene[]> = {
  // ---------------------------------------------------------------------
  // wayuu-kanasu-moire — rotated diamond lattices
  // ---------------------------------------------------------------------
  'wayuu-kanasu-moire': [
    {
      name: 'ambient-front',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [0, 60] },
      required: true,
      expect: {
        claim:
          'Two diamond lattices with a small rotational mismatch beat into coarse moiré fringes whose spacing depends on the mismatch angle.',
        plateSignature:
          'A dense diamond weave overlaid with visible large-scale moiré beat fringes — coarse bright/dark bands across the plate.',
        failModes: [
          'Uniform flat weave with no large-scale fringes',
          'High-frequency hash noise with no organized pattern',
        ],
        native: {
          canvasNotBlank: true,
          logEvents: ['recipe_bound'],
          uniformEquals: { uRecipe: 1 },
        },
      },
    },
    {
      name: 'ambient-tilted',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [35, 45] },
      expect: {
        claim:
          'Parallax-shifted sampling of the back lattice makes the fringes walk as the camera orbits.',
        plateSignature:
          'Moiré fringes in a visibly different spatial arrangement from the head-on view.',
        failModes: ['Identical fringe pattern to the head-on view (no parallax response)'],
        native: {
          canvasNotBlank: true,
          pixelDeltaVsRef: {
            refScene: 'ambient-front',
            minDelta: 8,
            samplePoints: [
              [200, 200],
              [400, 200],
              [300, 300],
              [200, 400],
              [400, 400],
            ],
          },
        },
      },
    },
  ],

  // ---------------------------------------------------------------------
  // emerald-facet-moire — mismatched hex lattices
  // ---------------------------------------------------------------------
  'emerald-facet-moire': [
    {
      name: 'ambient-front',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [0, 60] },
      required: true,
      expect: {
        claim:
          'Hex lattices with slightly different pitches produce a magnified hex "ghost gem" via moiré beating.',
        plateSignature:
          'A fine hex lattice with a visibly larger-scale hex "phantom" pattern superimposed.',
        failModes: [
          'Flat uniform hex lattice with no magnified ghost',
          'Chaotic noise without hex organization',
        ],
        native: {
          canvasNotBlank: true,
          logEvents: ['recipe_bound'],
          uniformEquals: { uRecipe: 1 },
        },
      },
    },
  ],

  // ---------------------------------------------------------------------
  // colibri-globe-lenticular — parallax-barrier slit grating
  // ---------------------------------------------------------------------
  'colibri-globe-lenticular': [
    {
      name: 'tilt-left',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [-18, 55] },
      required: true,
      expect: {
        claim:
          'Vertical slits gate two interlaced silhouettes (hummingbird, globe) via Snell-shifted parallax. Tilting the camera swaps which one is visible.',
        plateSignature:
          'A hummingbird silhouette emerging through the slit barrier at this tilt — coarse vertical louvres clearly visible over the bird body.',
        failModes: [
          'Static superposition of both silhouettes regardless of tilt',
          'Slit mask only, no scene content',
        ],
        native: {
          canvasNotBlank: true,
          logEvents: ['recipe_bound', 'stereo_views_bound'],
          uniformEquals: { uRecipe: 0 },
        },
      },
    },
    {
      name: 'tilt-right',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [18, 55] },
      required: true,
      expect: {
        claim:
          'Tilting the camera to the opposite side swaps the interlaced scene from hummingbird to globe.',
        plateSignature:
          'A globe silhouette (disk with latitude/longitude wireframe) emerging through the slit barrier — visibly different from the tilt-left capture.',
        failModes: ['Same content as tilt-left (tilt produces no swap)'],
        native: {
          canvasNotBlank: true,
          pixelDeltaVsRef: {
            refScene: 'tilt-left',
            minDelta: 8,
            samplePoints: [
              [250, 250],
              [350, 250],
              [300, 200],
              [300, 300],
              [300, 350],
            ],
          },
        },
      },
    },
  ],

  // ---------------------------------------------------------------------
  // colibri-globe-moire — dual-grating image moiré
  // ---------------------------------------------------------------------
  'colibri-globe-moire': [
    {
      name: 'ambient-front',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [0, 60] },
      required: true,
      expect: {
        claim:
          'Hummingbird and globe carved into line gratings at a small angle + period mismatch beat against each other; the beat phase walks with tilt.',
        plateSignature:
          'Fine line gratings with visible large-scale beat fringes that softly outline one or both silhouettes; neither image dominates head-on.',
        failModes: [
          'Flat line grating with no beat structure',
          'Both silhouettes statically visible regardless of tilt',
        ],
        native: {
          canvasNotBlank: true,
          logEvents: ['recipe_bound'],
          uniformEquals: { uRecipe: 1 },
        },
      },
    },
    {
      name: 'ambient-tilted',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [35, 45] },
      expect: {
        claim:
          'Tilt-driven parallax slides the back grating; the beat fringes shift and one silhouette becomes dominant.',
        plateSignature:
          'Beat fringes in a different spatial arrangement from the head-on view — one image (hummingbird or globe) stands out more.',
        failModes: ['Identical pattern to the head-on view'],
        native: {
          canvasNotBlank: true,
          pixelDeltaVsRef: {
            refScene: 'ambient-front',
            minDelta: 6,
            samplePoints: [
              [200, 200],
              [400, 200],
              [300, 300],
              [200, 400],
              [400, 400],
            ],
          },
        },
      },
    },
  ],

  // ---------------------------------------------------------------------
  // colibri-globe-phase — half-period phase-shift overlay
  // ---------------------------------------------------------------------
  'colibri-globe-phase': [
    {
      name: 'tilt-neutral',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [0, 55] },
      required: true,
      expect: {
        claim:
          'Front holds hummingbird stripes (carrier phase 0), back holds globe stripes (carrier phase π). Head-on the two interlace at half intensity.',
        plateSignature:
          'A fine vertical line texture with both silhouettes faintly co-visible (interlaced stripes) — neither image dominates.',
        failModes: ['One image already dominant at head-on (phase offset not respected)'],
        native: {
          canvasNotBlank: true,
          logEvents: ['recipe_bound'],
          uniformEquals: { uRecipe: 2 },
        },
      },
    },
    {
      name: 'tilt-right',
      setup: { ...defaultSettle, illumination: 'ambient', cameraAzEl: [25, 55] },
      required: true,
      expect: {
        claim:
          'Parallax through the substrate biases which carrier phase the eye samples; one silhouette becomes dominant under tilt.',
        plateSignature:
          'A clearly different reveal from tilt-neutral: one of the two silhouettes (hummingbird or globe) is visibly more pronounced.',
        failModes: ['Identical to tilt-neutral (parallax not biasing the phase)'],
        native: {
          canvasNotBlank: true,
          pixelDeltaVsRef: {
            refScene: 'tilt-neutral',
            minDelta: 5,
            samplePoints: [
              [200, 200],
              [300, 200],
              [400, 200],
              [300, 300],
              [300, 400],
            ],
          },
        },
      },
    },
  ],
};

export function allSlugs(): string[] {
  return Object.keys(CATALOG);
}

export function scenesFor(slug: string): VisualScene[] {
  return CATALOG[slug] ?? [];
}
