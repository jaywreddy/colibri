import * as THREE from 'three';
import type { BoxSpec, FaceId } from '../api';
import { platePlacements } from '../assembly';

/**
 * Tilt-progression capture: render ONE face in isolation across a tilt sweep,
 * at a scale where the optical effect is actually resolved.
 *
 * WHY THIS EXISTS AND WHY IT SUPERSAMPLES. The live viewport cannot show these
 * effects. The two-plane renderer filters each layer independently and then
 * composites, so it computes <front>*<back>; the physical part (and the eye)
 * integrate the product, <front*back>. The correlation between the layers IS
 * the switch and IS the moire, so as soon as the lattice falls below a screen
 * pixel the preview loses precisely the term the real object keeps — measured
 * on the shipping box, the frame fringes survive at about 1% contrast in the
 * overview, and the A/B interlace collapses to a static 50/50 blend.
 *
 * Supersampling is the honest fix, not magnification: render at a pitch that
 * RESOLVES the lattice (so the product is formed correctly per sample) and then
 * average down. That average is the same integration a pupil performs, so the
 * downsampled tile is a fair prediction of what a person sees — no periods are
 * altered, no physics is faked, and the tilt angles remain the true ones.
 *
 * The captured tiles feed two independent judgements, deliberately kept apart:
 *   - MEASURED: the frame-to-frame contrast actually rendered here.
 *   - PREDICTED: the perceptual budget computed from the fabricated geometry
 *     (backend app/readability.py), which never looks at these pixels.
 * Agreement between them is the signal worth trusting.
 */

/** Angles (deg) that cover a barrier's full behaviour: swap ~2.5, alias ~10. */
export const DEFAULT_ANGLES_DEG = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6];

export type ProgressionOptions = {
  /** Offscreen render pitch. Must resolve the lattice — see the note above. */
  renderPx?: number;
  /** Displayed tile size; renderPx/tilePx is the supersampling factor. */
  tilePx?: number;
  /** Field of view across the tile, in mm of the real plate. */
  fieldMm?: number;
  angles?: number[];
};

export type ProgressionFrame = {
  angleDeg: number;
  dataUrl: string;
  /** Mean luminance 0..255 of the tile. */
  mean: number;
};

export type ProgressionResult = {
  faceId: FaceId;
  slug: string;
  fieldMm: number;
  supersample: number;
  frames: ProgressionFrame[];
  /** Largest mean-brightness swing across the sweep (Michelson). */
  measuredContrast: number;
  /** Fraction of pixels that change by >2% between the extreme tilts. */
  changedFrac: number;
  /**
   * EFFECT STRENGTH: mean |dL| between the extreme tilts, as a fraction of mean
   * luminance. This is the honest headline for an image SWAP — mean brightness
   * barely moves when A and B have similar overall lightness, so
   * `measuredContrast` can read ~0% on a face where half the pixels invert.
   */
  effectStrength: number;
};

type StudioLike = {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  root: THREE.Group;
  controls: { enabled: boolean };
};

/** Hide everything that is not the target face; returns a restore function. */
function isolateFace(root: THREE.Group, faceId: FaceId): () => void {
  const changed: THREE.Object3D[] = [];
  root.traverse((o) => {
    const ud = o.userData as { faceId?: string; kind?: string; prop?: boolean };
    const isOtherFace = ud.faceId !== undefined && ud.faceId !== faceId;
    // Metalwork and display props belong to the assembled box, not to a single
    // plate under test — they would only occlude and tint the measurement.
    const isFurniture =
      ud.prop === true ||
      (ud.kind !== undefined && ud.kind !== 'plate-outer' && ud.kind !== 'plate-inner');
    if ((isOtherFace || isFurniture) && o.visible) {
      o.visible = false;
      changed.push(o);
    }
  });
  return () => {
    for (const o of changed) o.visible = true;
  };
}

/**
 * Capture the sweep. Renders offscreen at `renderPx`, downsamples to `tilePx`.
 *
 * The camera orbits in the face's own tangent frame about its UP axis, which is
 * the axis the barrier switches along (uSwitchAxis = 0 -> the face's +X), so a
 * positive angle walks the back layer the way a wrist tilt would.
 */
export async function captureTiltProgression(
  studio: StudioLike,
  spec: BoxSpec,
  faceId: FaceId,
  slug: string,
  opts: ProgressionOptions = {}
): Promise<ProgressionResult> {
  const renderPx = opts.renderPx ?? 2048;
  const tilePx = opts.tilePx ?? 340;
  const fieldMm = opts.fieldMm ?? 30;
  const angles = opts.angles ?? DEFAULT_ANGLES_DEG;

  const { scene, renderer, root } = studio;
  const placement = platePlacements(spec).find((p) => p.face === faceId);
  if (!placement) throw new Error(`no placement for face ${faceId}`);

  // World frame of the face. root.scale maps mm -> scene units.
  const sc = root.scale.x;
  const mm = (um: number) => (um / 1000) * sc;
  const center = new THREE.Vector3(
    mm(placement.center_um[0]),
    mm(placement.center_um[1]),
    mm(placement.center_um[2])
  );
  const euler = new THREE.Euler(
    placement.rotation[0],
    placement.rotation[1],
    placement.rotation[2],
    'XYZ'
  );
  const up = new THREE.Vector3(0, 1, 0).applyEuler(euler);
  const normal = new THREE.Vector3(...placement.outward);

  // Frame the requested field of view with a 45-deg camera.
  const fovDeg = 45;
  const fieldScene = (fieldMm / 1000) * 1000 * sc; // mm -> scene units
  const dist = fieldScene / 2 / Math.tan((fovDeg * Math.PI) / 360);
  const cam = new THREE.PerspectiveCamera(fovDeg, 1, 0.001, 100);

  // Save renderer state; the offscreen size and pixel ratio are ours alone.
  const savedSize = new THREE.Vector2();
  renderer.getSize(savedSize);
  const savedRatio = renderer.getPixelRatio();
  const restoreFace = isolateFace(root, faceId);
  const controlsWere = studio.controls.enabled;
  studio.controls.enabled = false;

  const full = document.createElement('canvas');
  full.width = renderPx;
  full.height = renderPx;
  const fullCtx = full.getContext('2d')!;
  const pixels = new Uint8Array(renderPx * renderPx * 4);
  const frames: ProgressionFrame[] = [];
  let firstTile: Uint8ClampedArray | null = null;
  let lastTile: Uint8ClampedArray | null = null;

  try {
    renderer.setPixelRatio(1);
    renderer.setSize(renderPx, renderPx, false);
    cam.aspect = 1;
    cam.updateProjectionMatrix();

    for (const angleDeg of angles) {
      const dir = normal.clone().applyAxisAngle(up, (angleDeg * Math.PI) / 180);
      cam.position.copy(center).addScaledVector(dir, dist);
      cam.up.copy(up);
      cam.lookAt(center);
      cam.updateMatrixWorld(true);
      renderer.render(scene, cam);

      const gl = renderer.getContext();
      gl.readPixels(0, 0, renderPx, renderPx, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      // readPixels is bottom-up; flip while blitting into the 2D canvas.
      const img = fullCtx.createImageData(renderPx, renderPx);
      for (let y = 0; y < renderPx; y++) {
        const src = (renderPx - 1 - y) * renderPx * 4;
        img.data.set(pixels.subarray(src, src + renderPx * 4), y * renderPx * 4);
      }
      fullCtx.putImageData(img, 0, 0);

      // Downsample = the supersample average (the pupil-like integration).
      const tile = document.createElement('canvas');
      tile.width = tilePx;
      tile.height = tilePx;
      const tctx = tile.getContext('2d')!;
      tctx.imageSmoothingEnabled = true;
      tctx.imageSmoothingQuality = 'high';
      tctx.drawImage(full, 0, 0, tilePx, tilePx);

      const data = tctx.getImageData(0, 0, tilePx, tilePx).data;
      let sum = 0;
      for (let i = 0; i < tilePx * tilePx; i++) {
        sum += 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
      }
      if (!firstTile) firstTile = new Uint8ClampedArray(data);
      lastTile = new Uint8ClampedArray(data);
      frames.push({
        angleDeg,
        dataUrl: tile.toDataURL('image/png'),
        mean: sum / (tilePx * tilePx),
      });
    }
  } finally {
    renderer.setPixelRatio(savedRatio);
    renderer.setSize(savedSize.x, savedSize.y, false);
    restoreFace();
    studio.controls.enabled = controlsWere;
  }

  // MEASURED effect strength, from the tiles themselves.
  const means = frames.map((f) => f.mean);
  const hi = Math.max(...means);
  const lo = Math.min(...means);
  const measuredContrast = hi + lo > 0 ? (hi - lo) / (hi + lo) : 0;

  let changed = 0;
  let effectStrength = 0;
  if (firstTile && lastTile) {
    const n = tilePx * tilePx;
    let mad = 0;
    let lumSum = 0;
    for (let i = 0; i < n; i++) {
      const a = 0.299 * firstTile[i * 4] + 0.587 * firstTile[i * 4 + 1] + 0.114 * firstTile[i * 4 + 2];
      const b = 0.299 * lastTile[i * 4] + 0.587 * lastTile[i * 4 + 1] + 0.114 * lastTile[i * 4 + 2];
      if (Math.abs(a - b) > 0.02 * 255) changed++;
      mad += Math.abs(a - b);
      lumSum += 0.5 * (a + b);
    }
    changed /= n;
    effectStrength = lumSum > 0 ? mad / lumSum : 0;
  }

  return {
    faceId,
    slug,
    fieldMm,
    supersample: renderPx / tilePx,
    frames,
    measuredContrast,
    changedFrac: changed,
    effectStrength,
  };
}
