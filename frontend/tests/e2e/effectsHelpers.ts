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
    async ([fid, k]) => {
      const s = (window as any).__studio;
      const raf = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
      const apply = (): number => {
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
      };
      // Apply-and-verify: the scene's rebuild is rAF-coalesced, so a rebuild
      // queued by the preceding store patch can land AFTER this mutation and
      // replace the plane meshes at the design gap. Re-apply until the value
      // survives two animation frames (fresh meshes lack __designGap, so a
      // clobbered apply recomputes it from the rebuilt scene — safe).
      let design = apply();
      for (let i = 0; i < 20; i += 1) {
        await raf();
        await raf();
        design = apply();
        const front = s.faces[fid].shader;
        const back = s.faces[fid].shaderBack;
        let outerZ: number | null = null;
        let innerZ: number | null = null;
        s.scene.traverse((o: any) => {
          if (o.isMesh && o.material === front && outerZ === null) outerZ = o.position.z;
          if (o.isMesh && o.material === back && innerZ === null) innerZ = o.position.z;
        });
        if (
          outerZ !== null &&
          innerZ !== null &&
          Math.abs(outerZ - innerZ - design * (k as number)) < 1e-9
        ) {
          break;
        }
      }
      // Direct scene mutation is invisible to the dirty-flag render loop —
      // book frames explicitly or the next capture reads a stale buffer.
      s.requestRender?.();
      return design;
    },
    [faceId, factor] as const
  );
}

/**
 * Gap factor used by the "collapse the gap" anti-cheat probes: the near-
 * registration state the parallax and reveal scenarios measure against.
 *
 * Deliberately NOT 0. At exactly 0 the inner plane lands on the outer plane's
 * z, which corrupts the measurement two ways that have nothing to do with
 * layer registration:
 *   (a) two opaque alphaToCoverage meshes (renderOrder 2 and 0) end up at
 *       identical depth, so the frame changes by depth fighting;
 *   (b) the inner plane leaves the glass slab. The outer pattern plane sits at
 *       T/2 + EPS_PATTERN_MM (BoxScene, EPS = 0.03 mm) while the slab spans
 *       +/-T/2, so the inner plane is inside the slab — and thus keeps its
 *       transmission-backdrop treatment — only while the gap exceeds EPS.
 *
 * 0.12 clears both for the suite's substrate (the default box: T = 500 um,
 * n = 1.46 -> design gap 0.3425 mm, so a 12% gap is 41 um, comfortably above
 * the 30 um containment floor and ~60 depth-buffer steps of separation). The
 * general condition is factor > EPS_PATTERN_MM * n / T; revisit this constant
 * if the effects suite ever runs on a much thinner substrate.
 *
 * 12% of the design gap is still REGISTERED for everything measured here: at
 * the scenarios' oblique views it leaves ~7 um of in-plane shift against the
 * 60 um centerpiece barrier period (0.12 period) and ~12% of the carrier
 * reveal's half-period, i.e. the cross-layer effect has collapsed.
 */
export const GAP_COLLAPSE_FACTOR = 0.12;

/**
 * Resolve true-pitch micro-patterns for a probe: dolly the camera in by
 * `radiusScale` and raise the renderer pixel ratio. Both preserve every tilt
 * angle (that is the point — the shader no longer magnifies the centerpiece
 * pitch, so at the default view the 60 µm comb is sub-pixel and its fringes
 * average away; zoom is the HONEST way to see them). Returns a restore
 * function. ROIs must be (re)computed after calling this.
 */
export async function zoomForMicroPatterns(
  page: Page,
  opts: { radiusScale?: number; pixelRatio?: number } = {}
): Promise<() => Promise<void>> {
  const { radiusScale = 0.45, pixelRatio = 2.5 } = opts;
  const prev = await page.evaluate(
    ([rs, pr]) => {
      const s = (window as any).__studio;
      const cam = s.camera;
      const t = s.controls?.target ?? { x: 0, y: 0, z: 0 };
      const prevRatio = s.renderer.getPixelRatio();
      cam.position.set(
        t.x + (cam.position.x - t.x) * (rs as number),
        t.y + (cam.position.y - t.y) * (rs as number),
        t.z + (cam.position.z - t.z) * (rs as number)
      );
      cam.lookAt(t.x, t.y, t.z);
      s.controls?.update?.();
      s.renderer.setPixelRatio(pr as number);
      // setPixelRatio needs a setSize to reallocate the drawing buffer.
      const el = s.renderer.domElement as HTMLCanvasElement;
      s.renderer.setSize(el.clientWidth, el.clientHeight, false);
      s.requestRender?.();
      return { prevRatio, invScale: 1 / (rs as number) };
    },
    [radiusScale, pixelRatio] as const
  );
  return async () => {
    await page.evaluate(
      ([ratio, inv]) => {
        const s = (window as any).__studio;
        const cam = s.camera;
        const t = s.controls?.target ?? { x: 0, y: 0, z: 0 };
        cam.position.set(
          t.x + (cam.position.x - t.x) * (inv as number),
          t.y + (cam.position.y - t.y) * (inv as number),
          t.z + (cam.position.z - t.z) * (inv as number)
        );
        cam.lookAt(t.x, t.y, t.z);
        s.controls?.update?.();
        s.renderer.setPixelRatio(ratio as number);
        const el = s.renderer.domElement as HTMLCanvasElement;
        s.renderer.setSize(el.clientWidth, el.clientHeight, false);
        s.requestRender?.();
      },
      [prev.prevRatio, prev.invScale] as const
    );
  };
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
      const s = (window as any).__studio;
      const u = s.faces[fid].shader.uniforms[n];
      const old = u.value as number;
      u.value = v;
      // Uniform pokes bypass React/state — book frames so captures see them.
      s.requestRender?.();
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

/**
 * The six wall ids, in the order BoxScene builds them. Mirrors FACE_IDS in
 * src/api.ts — copied rather than imported because api.ts pulls in logger.ts,
 * which reads `import.meta.env`, and this module is evaluated by Playwright in
 * Node where that is not defined.
 */
export const FACE_IDS = ['front', 'back', 'top', 'bottom', 'left', 'right'] as const;

/**
 * Structural state of one face's TWO plane materials, cross-checked against
 * the box manifest the store holds.
 *
 * This is the cheap half of the texture-driven axiom: `maskW > 1` proves a
 * real backend mask PNG is bound (the pre-bind placeholder is a 1x1
 * DataTexture, BoxScene::makeBlankTexture), and `maskMatchesManifest` proves
 * the bound image URL is the one THIS face's manifest declared — the outer
 * plane must carry `files.front_png`, the inner plane `files.back_png`.
 *
 * LITERAL faces bind the same axiom to a different pair of slots. Their two
 * layers are composited in ONE pass on the OUTER plane (the eye integrates the
 * PRODUCT of the two layers' transmissions, which two independently filtered
 * planes cannot express — see plate.frag::runLiteralLayer), so the inner PATTERN
 * plane is hidden and the back raster arrives as `uBackCoverage` on the OUTER
 * material. `maskBackW` / `maskBackMatchesManifest` therefore read that uniform
 * on a literal face and `uFront` of the inner plane on a procedural one: same
 * question — "is THIS face's declared back mask actually feeding the pixels?" —
 * asked of whichever slot carries it. `visibleBack` is reported as-is (false on
 * a literal face, by construction); `literal` says which contract applies.
 */
export type FaceRenderState = {
  face: string;
  /** render_recipe as shipped by the backend for this face (null if no entry). */
  manifestRecipe: string | null;
  /** recipe_data.literal — the face draws the fabricated raster, one-pass. */
  literal: boolean;
  /** uRecipe on the outer / inner plane material. */
  recipe: number;
  recipeBack: number;
  visible: boolean;
  /** Inner PATTERN plane visibility. Always false on a literal face. */
  visibleBack: boolean;
  /** Bound uFront image width per plane; 1 = still the blank placeholder. */
  maskW: number;
  /**
   * Width of the BACK layer's bound raster: `uBackCoverage` on the outer
   * material for a literal face, `uFront` on the inner plane otherwise.
   */
  maskBackW: number;
  maskMatchesManifest: boolean;
  maskBackMatchesManifest: boolean;
  /** substrate the manifest shipped for this face (drives the T/n plane gap). */
  thicknessUm: number | null;
  n: number | null;
  /** uThicknessUm / uN as actually bound on the outer plane. */
  uThicknessUm: number;
  uN: number;
  /** recipe_data flags: a blank face binds nothing; a single-ply face has no inner plane. */
  blank: boolean;
  singlePly: boolean;
};

/** Read the two-plane structural state of all six faces in one round-trip. */
export async function allFaceRenderState(page: Page): Promise<FaceRenderState[]> {
  return await page.evaluate((ids) => {
    const s = (window as any).__studio;
    const bm = s.store.getState().boxManifest as any;
    const imgOf = (mat: any, uniform: string): { w: number; src: string } => {
      const img = mat?.uniforms?.[uniform]?.value?.image;
      return {
        w: Number(img?.width ?? 0),
        src: typeof img?.src === 'string' ? (img.src as string) : '',
      };
    };
    const declares = (src: string, p: unknown): boolean =>
      typeof p === 'string' && p.length > 0 && src.length > 0 && src.endsWith(p);
    return (ids as readonly string[]).map((fid) => {
      const rt = s.faces?.[fid];
      const fm = bm?.faces?.[fid] ?? null;
      const literal = !!fm?.recipe_data?.literal;
      const outer = imgOf(rt?.shader, 'uFront');
      // Where this face's BACK layer actually feeds the pixels: the composite
      // uniform on the outer material (literal) or the inner plane's own mask.
      const inner = literal
        ? imgOf(rt?.shader, 'uBackCoverage')
        : imgOf(rt?.shaderBack, 'uFront');
      return {
        face: fid,
        blank: !!fm?.recipe_data?.blank,
        singlePly: !!fm?.recipe_data?.single_ply,
        literal,
        manifestRecipe: (fm?.render_recipe as string | undefined) ?? null,
        recipe: Number(rt?.shader?.uniforms?.uRecipe?.value ?? -1),
        recipeBack: Number(rt?.shaderBack?.uniforms?.uRecipe?.value ?? -1),
        visible: !!rt?.shader?.visible,
        visibleBack: !!rt?.shaderBack?.visible,
        maskW: outer.w,
        maskBackW: inner.w,
        // A literal face binds the fabricated-geometry raster instead of the
        // level-coded mask; either is THIS face's declared image.
        maskMatchesManifest:
          declares(outer.src, fm?.files?.front_png) || declares(outer.src, fm?.files?.literal_front),
        maskBackMatchesManifest:
          declares(inner.src, fm?.files?.back_png) || declares(inner.src, fm?.files?.literal_back),
        thicknessUm:
          typeof fm?.substrate?.thickness_um === 'number'
            ? (fm.substrate.thickness_um as number)
            : null,
        n: typeof fm?.substrate?.n === 'number' ? (fm.substrate.n as number) : null,
        uThicknessUm: Number(rt?.shader?.uniforms?.uThicknessUm?.value ?? -1),
        uN: Number(rt?.shader?.uniforms?.uN?.value ?? -1),
      };
    });
  }, FACE_IDS);
}

/**
 * Wait until every one of the six faces has the real masks THE CURRENT MANIFEST
 * DECLARES bound on BOTH planes. helpers.ts::waitForBoxTextures only covers the
 * front face, so without this the five other walls can still be on the 1x1
 * blank when a spec starts asserting.
 *
 * The manifest-URL half of the predicate matters because the bind is two-step
 * and async — the store takes the new manifest, then twelve PNG loads resolve —
 * so a face can legitimately be showing the PREVIOUS design's mask for a beat
 * after a regen. Waiting for agreement makes the caller's structural assertions
 * a check on the settled state instead of a race.
 *
 * The default cap is deliberately well under the 60 s per-test budget
 * (playwright.config.ts) so a stuck face reports as THIS wait failing — with
 * the predicate in the message, and the frontend log (face_texture_retry /
 * face_texture_failed / face_recipe_unsupported) attached by the fixture —
 * instead of as an opaque test timeout.
 */
export async function waitForAllFaceMasks(page: Page, timeoutMs = 30_000): Promise<void> {
  await page.waitForFunction(
    (ids) => {
      const s = (window as any).__studio;
      if (!s?.faces) return false;
      const bm = s.store?.getState?.().boxManifest;
      if (!bm) return false;
      const bound = (mat: any, uniform: string, declared: unknown): boolean => {
        const img = mat?.uniforms?.[uniform]?.value?.image;
        if (Number(img?.width ?? 0) <= 1) return false;
        const src = typeof img?.src === 'string' ? (img.src as string) : '';
        return typeof declared === 'string' && declared.length > 0 && src.endsWith(declared);
      };
      return (ids as readonly string[]).every((fid) => {
        const rt = s.faces[fid];
        const files = bm.faces?.[fid]?.files;
        const literal = !!bm.faces?.[fid]?.recipe_data?.literal;
        // literal faces may legitimately drop an EMPTY layer (blank / bare inner ply)
        const boundOr = (mat: any, u: string, a: unknown, b: unknown) =>
          bound(mat, u, a) || bound(mat, u, b);
        if (literal) {
          const rd = bm.faces?.[fid]?.recipe_data ?? {};
          if (rd.blank) return true; // bare glass: nothing to bind on either layer
          // ONE-PASS COMPOSITE: both layers live on the outer material (uFront =
          // the outer chrome, uBackCoverage = the inner). The inner PATTERN plane
          // is hidden and binds nothing that reaches a pixel, so waiting on it
          // would wait on a slot the renderer no longer reads.
          const outerOk = boundOr(rt?.shader, 'uFront', files?.front_png, files?.literal_front);
          const innerOk =
            rd.single_ply ||
            boundOr(rt?.shader, 'uBackCoverage', files?.back_png, files?.literal_back);
          return outerOk && innerOk;
        }
        return (
          bound(rt?.shader, 'uFront', files?.front_png) &&
          bound(rt?.shaderBack, 'uFront', files?.back_png)
        );
      });
    },
    FACE_IDS,
    { timeout: timeoutMs }
  );
}

/**
 * The substrate the given face's manifest shipped — the numbers BoxScene turns
 * into the physical inner-plane gap (T/n). Returns null when the box manifest
 * has no entry for that face.
 */
export async function faceSubstrate(
  page: Page,
  faceId: string
): Promise<{ thicknessUm: number; n: number } | null> {
  return await page.evaluate((fid) => {
    const bm = (window as any).__studio.store.getState().boxManifest as any;
    const sub = bm?.faces?.[fid]?.substrate;
    if (!sub || typeof sub.thickness_um !== 'number' || typeof sub.n !== 'number') return null;
    return { thicknessUm: sub.thickness_um as number, n: sub.n as number };
  }, faceId);
}

/**
 * A fixed wall-clock wait.
 *
 * Use this ONLY where the duration is the point — the measurement interval
 * between two frames in a time-invariance / turntable comparison, or letting an
 * animation run. For "wait until the scene has stopped moving" use
 * waitForStableFrame, which measures quiescence instead of guessing at it.
 */
export async function settle(page: Page, ms = 700): Promise<void> {
  await page.waitForTimeout(ms);
}

export type SettleResult = {
  /** True when the quiet streak was reached before the deadline. */
  stable: boolean;
  waitedMs: number;
  /** Metrics of the last frame pair compared. */
  last: DiffMetrics | null;
};

/**
 * Bounded wait for frame QUIESCENCE instead of a fixed sleep.
 *
 * Several assertions in the suite demand a genuinely still frame (the
 * time-invariance and turntable-off bounds are changedFrac <= 0.005), yet the
 * waits in front of them were wall-clock guesses. OrbitControls' damping decay
 * is frame-rate dependent, so on a loaded host a fixed 900 ms can leave
 * residual sub-pixel drift that trips those bounds. This polls small
 * downsampled full-canvas captures and returns as soon as TWO consecutive
 * frame pairs are quiet, so the typical cost is well under the budget it
 * replaces; `maxMs` keeps the worst case at the old fixed duration.
 *
 * Thresholds sit between the suite's determinism noise floor (same-state
 * recapture is asserted at mad <= 0.5) and the motion bounds it feeds
 * (changedFrac <= 0.005), so "quiet" is stricter than "passes" but still
 * reachable under MSAA dither. Never throws: the scenario's own metrics remain
 * the gate, and a non-stable return is reported in the log for diagnosis.
 */
export async function waitForStableFrame(
  page: Page,
  maxMs = 900,
  opts: { samples?: number; epsMad?: number; epsChanged?: number; pollMs?: number } = {}
): Promise<SettleResult> {
  const samples = opts.samples ?? 24;
  const epsMad = opts.epsMad ?? 0.8;
  const epsChanged = opts.epsChanged ?? 0.0015;
  const pollMs = opts.pollMs ?? 60;
  const t0 = Date.now();
  const deadline = t0 + maxMs;
  let prev = await captureGray(page, null, samples);
  let last: DiffMetrics | null = null;
  let quiet = 0;
  while (Date.now() < deadline) {
    await page.waitForTimeout(pollMs);
    const cur = await captureGray(page, null, samples);
    last = diffFrames(prev, cur);
    prev = cur;
    quiet = last.mad <= epsMad && last.changedFrac <= epsChanged ? quiet + 1 : 0;
    if (quiet >= 2) return { stable: true, waitedMs: Date.now() - t0, last };
  }
  console.log(
    `[effects] waitForStableFrame: not quiescent within ${maxMs}ms ` +
      `(last mad ${last ? last.mad.toFixed(3) : 'n/a'}, changedFrac ${
        last ? last.changedFrac.toFixed(4) : 'n/a'
      })`
  );
  return { stable: false, waitedMs: Date.now() - t0, last };
}

/** Current |lid pivot rotation| in degrees, or null in flat layout. */
export async function lidRotationDeg(page: Page): Promise<number | null> {
  return await page.evaluate(() => {
    const p = (window as any).__studio.lidPivot;
    return p ? Math.abs((p.rotation.x * 180) / Math.PI) : null;
  });
}
