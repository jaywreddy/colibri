/**
 * Effects-harness helpers — pixel-level instrumentation for the physical
 * honesty suite (effectsPhysical.spec.ts).
 *
 * Everything here reads the live WebGL back buffer through window.__studio
 * (render + readPixels done atomically in one page.evaluate). Captures are
 * downsampled grayscale grids, small enough to ship over the CDP wire and
 * diff in Node, plus full-resolution PNG dumps for the vision layer and for
 * humans diagnosing a failure.
 *
 * Coordinate note: gl.readPixels is bottom-left origin. All captures share
 * the convention, so diff metrics are orientation-agnostic; PNG dumps go
 * through canvas.toDataURL (top-left) and are for eyes only.
 */
import { type Page } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

export type Roi = { x: number; y: number; w: number; h: number };

export type GrayFrame = {
  /** Downsampled grayscale samples, row-major. */
  data: number[];
  cols: number;
  rows: number;
  /** Mean per-channel values over the ROI (for hue assertions). */
  meanRgb: [number, number, number];
};

export type DiffMetrics = {
  /** Mean absolute difference, 0..255. */
  mad: number;
  /** Fraction of samples whose |delta| > 12. */
  changedFrac: number;
  /** Pearson correlation of the two sample vectors (1 = identical structure). */
  corr: number;
};

/** Read the drawing-buffer size (device pixels, not CSS pixels). */
export async function drawingBufferSize(page: Page): Promise<{ w: number; h: number }> {
  return await page.evaluate(() => {
    const s = (window as any).__studio;
    const c = s.renderer.domElement as HTMLCanvasElement;
    return { w: c.width, h: c.height };
  });
}

/**
 * Screen-space bounding box (drawing-buffer pixels, bottom-left origin) of
 * one face's pattern plane. Found by traversing the scene for the mesh bound
 * to that face's ShaderMaterial, projecting its PlaneGeometry corners, and
 * insetting the box to stay clear of foil strips and antialiased edges.
 * Returns null when the face is back-facing or off-screen.
 */
export async function facePlateROI(
  page: Page,
  faceId: string,
  insetFrac = 0.18
): Promise<Roi | null> {
  return await page.evaluate(
    ([fid, inset]) => {
      const s = (window as any).__studio;
      const shader = s.faces[fid].shader;
      let mesh: any = null;
      s.scene.traverse((o: any) => {
        if (!mesh && o.isMesh && o.material === shader) mesh = o;
      });
      if (!mesh) return null;
      mesh.updateWorldMatrix(true, false);
      const geom = mesh.geometry;
      geom.computeBoundingBox();
      const bb = geom.boundingBox;
      const corners = [
        [bb.min.x, bb.min.y],
        [bb.min.x, bb.max.y],
        [bb.max.x, bb.min.y],
        [bb.max.x, bb.max.y],
      ];
      const canvas = s.renderer.domElement as HTMLCanvasElement;
      const W = canvas.width;
      const H = canvas.height;
      s.camera.updateMatrixWorld();
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const [cx, cy] of corners) {
        // localToWorld then project — done with plain math to avoid needing
        // the THREE namespace: v_clip = P * V * M * v_local.
        const v = { x: cx, y: cy, z: 0 };
        const m = mesh.matrixWorld.elements;
        const wx = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12];
        const wy = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13];
        const wz = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14];
        const cam = s.camera;
        const e = cam.matrixWorldInverse.elements;
        const p = cam.projectionMatrix.elements;
        const vx = e[0] * wx + e[4] * wy + e[8] * wz + e[12];
        const vy = e[1] * wx + e[5] * wy + e[9] * wz + e[13];
        const vz = e[2] * wx + e[6] * wy + e[10] * wz + e[14];
        const clipX = p[0] * vx + p[4] * vy + p[8] * vz + p[12];
        const clipY = p[1] * vx + p[5] * vy + p[9] * vz + p[13];
        const clipW = p[3] * vx + p[7] * vy + p[11] * vz + p[15];
        if (clipW <= 0) return null; // behind the camera
        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        const sx = ((ndcX + 1) / 2) * W;
        const sy = ((ndcY + 1) / 2) * H; // bottom-left origin, matches readPixels
        minX = Math.min(minX, sx);
        maxX = Math.max(maxX, sx);
        minY = Math.min(minY, sy);
        maxY = Math.max(maxY, sy);
      }
      const bw = maxX - minX;
      const bh = maxY - minY;
      const ix = bw * (inset as number);
      const iy = bh * (inset as number);
      const roi = {
        x: Math.max(0, Math.round(minX + ix)),
        y: Math.max(0, Math.round(minY + iy)),
        w: Math.round(bw - 2 * ix),
        h: Math.round(bh - 2 * iy),
      };
      if (roi.w < 8 || roi.h < 8) return null;
      roi.w = Math.min(roi.w, W - roi.x);
      roi.h = Math.min(roi.h, H - roi.y);
      return roi;
    },
    [faceId, insetFrac] as const
  );
}

/**
 * Capture a downsampled grayscale grid of an ROI (or the whole buffer when
 * roi is null). Render + readPixels happen atomically in-page. maxSamples
 * bounds the wire payload (~96x96 grid by default).
 */
export async function captureGray(
  page: Page,
  roi: Roi | null,
  maxSamples = 96
): Promise<GrayFrame> {
  return await page.evaluate(
    ([r, maxN]) => {
      const s = (window as any).__studio;
      s.renderer.render(s.scene, s.camera);
      const gl = s.renderer.getContext() as WebGLRenderingContext;
      const canvas = s.renderer.domElement as HTMLCanvasElement;
      const rx = r ? r.x : 0;
      const ry = r ? r.y : 0;
      const rw = r ? r.w : canvas.width;
      const rh = r ? r.h : canvas.height;
      const buf = new Uint8Array(rw * rh * 4);
      gl.readPixels(rx, ry, rw, rh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      const cols = Math.min(maxN as number, rw);
      const rows = Math.min(maxN as number, rh);
      const data: number[] = new Array(cols * rows);
      let sr = 0;
      let sg = 0;
      let sb = 0;
      for (let j = 0; j < rows; j++) {
        const py = Math.floor((j * rh) / rows);
        for (let i = 0; i < cols; i++) {
          const px = Math.floor((i * rw) / cols);
          const o = (py * rw + px) * 4;
          const R = buf[o];
          const G = buf[o + 1];
          const B = buf[o + 2];
          data[j * cols + i] = (R + G + B) / 3;
          sr += R;
          sg += G;
          sb += B;
        }
      }
      const n = cols * rows;
      return {
        data,
        cols,
        rows,
        meanRgb: [sr / n, sg / n, sb / n] as [number, number, number],
      };
    },
    [roi, maxSamples] as const
  );
}

/** Diff two equally-shaped GrayFrames. */
export function diffFrames(a: GrayFrame, b: GrayFrame): DiffMetrics {
  if (a.data.length !== b.data.length) {
    throw new Error(`diffFrames: shape mismatch ${a.data.length} vs ${b.data.length}`);
  }
  const n = a.data.length;
  let sumAbs = 0;
  let changed = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < n; i++) {
    const d = a.data[i] - b.data[i];
    sumAbs += Math.abs(d);
    if (Math.abs(d) > 12) changed++;
    sa += a.data[i];
    sb += b.data[i];
  }
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a.data[i] - ma;
    const db = b.data[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(va * vb);
  return {
    mad: sumAbs / n,
    changedFrac: changed / n,
    corr: denom > 1e-9 ? cov / denom : 1,
  };
}

/** Dump the full canvas to a PNG on disk (vision layer + human diagnosis). */
export async function dumpFramePng(page: Page, outDir: string, name: string): Promise<string> {
  await fs.mkdir(outDir, { recursive: true });
  const dataUrl = (await page.evaluate(() => {
    const s = (window as any).__studio;
    s.renderer.render(s.scene, s.camera);
    return (s.renderer.domElement as HTMLCanvasElement).toDataURL('image/png');
  })) as string;
  const p = path.join(outDir, `${name.replace(/[^a-z0-9._-]/gi, '_')}.png`);
  await fs.writeFile(p, Buffer.from(dataUrl.split(',', 2)[1] ?? '', 'base64'));
  return p;
}

/**
 * Scale the geometric two-plane gap of one face. The merged renderer places
 * the BACK gold layer on a REAL inner plane at the paraxial air gap T/n below
 * the outer plane — parallax emerges from perspective across that gap, not
 * from a thickness uniform. Scaling the gap is therefore the honest substrate
 * manipulation: factor 0 registers the layers (parallax collapses), factor 1
 * restores the design gap. Returns the design gap in scene mm.
 */
export async function scaleBackPlaneGap(
  page: Page,
  faceId: string,
  factor: number
): Promise<number> {
  return await page.evaluate(
    ([fid, k]) => {
      const s = (window as any).__studio;
      const front = s.faces[fid].shader;
      const back = s.faces[fid].shaderBack;
      let outerZ: number | null = null;
      let inner: any = null;
      s.scene.traverse((o: any) => {
        if (o.isMesh && o.material === front && outerZ === null) outerZ = o.position.z;
        if (o.isMesh && o.material === back && !inner) inner = o;
      });
      if (outerZ === null || !inner) throw new Error(`two-plane meshes not found for ${fid}`);
      const ud = inner.userData as { __designGap?: number };
      if (ud.__designGap === undefined) ud.__designGap = outerZ - inner.position.z;
      inner.position.z = outerZ - ud.__designGap * (k as number);
      return ud.__designGap;
    },
    [faceId, factor] as const
  );
}

/** Set a shader uniform on one face (numbers only). Returns the old value. */
export async function setFaceScalarUniform(
  page: Page,
  faceId: string,
  name: string,
  value: number
): Promise<number> {
  return await page.evaluate(
    ([fid, n, v]) => {
      const u = (window as any).__studio.faces[fid].shader.uniforms[n];
      const old = u.value as number;
      u.value = v;
      return old;
    },
    [faceId, name, value] as const
  );
}

/**
 * Assign a pattern slug to one face through the store (triggers the app's
 * debounced regen) and wait until BoxScene logs `face_texture_bound` for the
 * new slug/recipe. This exercises the REAL pipeline: backend mask generation
 * -> manifest -> texture bind.
 */
export async function assignFacePattern(
  page: Page,
  faceId: string,
  slug: string,
  expectRecipe: string,
  timeoutMs = 90_000
): Promise<void> {
  await page.evaluate(
    ([fid, sl]) => {
      (window as unknown as { __log?: unknown[] }).__log = [];
      (window as any).__studio.store.getState().patchFace(fid, {
        pattern_slug: sl,
        pattern_params: {},
      });
    },
    [faceId, slug] as const
  );
  await page.waitForFunction(
    ([fid, sl, rec]) => {
      const buf = ((window as unknown as { __log?: any[] }).__log ?? []) as any[];
      return buf.some(
        (e) =>
          e.type === 'face_texture_bound' &&
          e.face === fid &&
          e.slug === sl &&
          e.recipe === rec
      );
    },
    [faceId, slug, expectRecipe] as const,
    { timeout: timeoutMs }
  );
}

/** Read which recipe id a face's shader is currently running. */
export async function faceRecipeId(page: Page, faceId: string): Promise<number> {
  return await page.evaluate(
    (fid) => (window as any).__studio.faces[fid].shader.uniforms.uRecipe.value as number,
    faceId
  );
}

/** Let the damped lid/camera/controls settle: N quiet RAF frames. */
export async function settle(page: Page, ms = 700): Promise<void> {
  await page.waitForTimeout(ms);
}

/** Current |lid pivot rotation| in degrees, or null in flat layout. */
export async function lidRotationDeg(page: Page): Promise<number | null> {
  return await page.evaluate(() => {
    const p = (window as any).__studio.lidPivot;
    return p ? Math.abs((p.rotation.x * 180) / Math.PI) : null;
  });
}
