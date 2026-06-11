/**
 * Scene catalog for the image-based visual-verification harness — Ring Box
 * Studio edition. Three canonical box scenes: closed, open at 100 deg, and
 * the flat fab-inspection layout.
 *
 * The Playwright driver (visualSignatures.spec.ts) reads this catalog and
 * captures one PNG + metadata sidecar per scene. The vision verifier
 * (tools/visual_verifier.py) reads the same catalog to build per-scene
 * prompts, so there is one source of truth.
 *
 * Keep the English descriptions concise — the vision model is prompted with
 * them verbatim and longer prose drifts into ambiguity.
 */

export type Illumination = 'ambient' | 'laser' | 'backlight';

export type BoxSceneSetup = {
  /** Target lid opening angle (degrees, 0..120). */
  lidDeg?: number;
  layout?: 'assembled' | 'flat';
  illumination?: Illumination;
  cameraAzEl?: [number, number]; // degrees [azimuth, elevation from +Y]
  settleMs?: number; // RAF settling after setup; default 800
};

export type BoxSceneNativeChecks = {
  canvasNotBlank?: boolean;
  /** |pivot rotation| must be at least this many degrees after settling. */
  lidRotationAtLeastDeg?: number;
  /** |pivot rotation| must be at most this many degrees after settling. */
  lidRotationAtMostDeg?: number;
  /** Store layout must equal this value. */
  layoutEquals?: 'assembled' | 'flat';
};

export type BoxVisualScene = {
  name: string;
  setup: BoxSceneSetup;
  /** If true, a FAIL on this scene fails the CI gate. */
  required?: boolean;
  expect: {
    claim: string;
    signature: string;
    failModes: string[];
    native?: BoxSceneNativeChecks;
  };
};

export const BOX_SCENES: BoxVisualScene[] = [
  {
    name: 'assembled-closed',
    setup: { lidDeg: 0, layout: 'assembled', illumination: 'ambient', cameraAzEl: [30, 55], settleMs: 1000 },
    required: true,
    expect: {
      claim:
        'A closed glass ring box: 6 fused-silica plates with gold patterns, copper-foil strips along every plate border, solder beads on the bottom and corner seams, and a brass tube-and-rod hinge along the back top edge.',
      signature:
        'A closed rectangular glass box with visible gold patterning on its faces, metallic strips framing each plate, and a thin brass cylinder run along one top edge.',
      failModes: [
        'Six flat untextured planes with no metalwork (foil/seams/hinge missing)',
        'Plates floating apart or intersecting (assembly positions wrong)',
        'Entirely dark canvas',
      ],
      native: {
        canvasNotBlank: true,
        lidRotationAtMostDeg: 2,
        layoutEquals: 'assembled',
      },
    },
  },
  {
    name: 'assembled-open-100',
    setup: { lidDeg: 100, layout: 'assembled', illumination: 'ambient', cameraAzEl: [30, 55], settleMs: 2000 },
    required: true,
    expect: {
      claim:
        'The same box with the lid rotated ~100 degrees about the back-edge hinge: the lid front edge swings UP and BACK over the hinge, exposing the box interior.',
      signature:
        'An open glass box — the flat lid plate tilted far back past vertical behind the box, interior visible, hinge tubes at the pivot line.',
      failModes: [
        'Lid rotated INTO the box volume (wrong rotation sign)',
        'Lid translated instead of rotated about the back edge',
        'Lid still closed (animation not driven)',
      ],
      native: {
        canvasNotBlank: true,
        lidRotationAtLeastDeg: 95,
        layoutEquals: 'assembled',
      },
    },
  },
  {
    name: 'flat-fab-layout',
    setup: { layout: 'flat', illumination: 'ambient', cameraAzEl: [0, 10], settleMs: 1000 },
    required: true,
    expect: {
      claim:
        'Fab-inspection layout: the 6 plates laid out head-on in a labeled 2x3 grid with their gold patterns and foil keep-out rims; seams and hinge hidden.',
      signature:
        'Six flat rectangular plates in a 2x3 grid, each with a gold pattern and a blank/metallic rim, with face-name labels; no 3D box.',
      failModes: [
        'Plates still assembled as a box',
        'Plates overlapping in the grid',
        'Seam beads or hinge visible in flat mode',
      ],
      native: {
        canvasNotBlank: true,
        layoutEquals: 'flat',
      },
    },
  },
];

export function allScenes(): BoxVisualScene[] {
  return BOX_SCENES;
}
