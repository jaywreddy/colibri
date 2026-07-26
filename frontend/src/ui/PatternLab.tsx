import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import {
  generatePattern,
  getDefault,
  type FaceId,
  type PatternManifest,
} from '../api';
import { log } from '../logger';
import { FACE_IDS, useStore } from '../store';
import {
  DEFAULT_LASER_COLOR,
  compositeParallax,
  grayFromRgba,
  parallaxShiftUm,
  tiltForShiftUm,
  type Illumination2D,
  type ImageDataLike,
} from '../lab/composite2d';
import { FACE_LABELS } from './FacesPanel';
import { Button, KIT, NumberRow, SelectRow, SliderRow, SubHeader } from './kit';

/** Target canvas width in px; height follows the pattern's extent aspect. */
const CANVAS_W = 420;
const TILT_MAX_DEG = 30;
/**
 * Linear downscale of the composite grid while a tilt drag is in flight.
 * compositeParallax is O(pixels) of JS math per frame (~176k px at 420², with
 * a bilinear back-mask fetch each), which cannot keep up with pointermove at
 * full resolution; a half-size grid is a quarter of that work and the drag is
 * a coarse gesture. The full-resolution composite lands on pointer release,
 * so what the designer measures is never the reduced one.
 */
const DRAG_DIVISOR = 2;

const clampTilt = (deg: number): number =>
  Math.max(-TILT_MAX_DEG, Math.min(TILT_MAX_DEG, deg));

/**
 * Pull the human-readable message out of an api.ts error string.
 *
 * Failed requests arrive as `"<fn>: <status> <raw body>"` and FastAPI bodies
 * are `{"detail": "..."}`. The generate route's 422 path passes the ValueError
 * text through verbatim (ParamSpec bounds, the litho floor, the 400k
 * lattice-budget guidance) — that detail IS the copy written for the user, so
 * show it alone. Anything unparseable falls through unchanged.
 */
function readableError(message: string): string {
  const brace = message.indexOf('{');
  if (brace < 0) return message;
  try {
    const body = JSON.parse(message.slice(brace)) as { detail?: unknown };
    if (typeof body.detail === 'string' && body.detail.trim()) return body.detail;
  } catch {
    /* not a JSON body — show what we got */
  }
  return message;
}

type MaskPair = { front: ImageDataLike; back: ImageDataLike };
type MaskGrids = { full: MaskPair; low: MaskPair };

/**
 * Downsample a mask PNG into single-channel grids, one per requested size.
 * This is the only DOM-touching step of the 2D pipeline — the math itself
 * (src/lab/composite2d.ts) stays canvas-free for unit tests.
 *
 * Every size shares ONE decode: the drag-resolution grid is derived from the
 * same decoded bitmap as the full one, so carrying two resolutions costs a
 * second drawImage, not a second network fetch + decode.
 */
async function loadMaskGrids(
  url: string,
  sizes: readonly (readonly [number, number])[]
): Promise<ImageDataLike[]> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const cv = document.createElement('canvas');
  return sizes.map(([w, h]) => {
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable');
    ctx.drawImage(img, 0, 0, w, h);
    return grayFromRgba(ctx.getImageData(0, 0, w, h).data, w, h);
  });
}

/**
 * Pattern Lab — a lightweight 2D dual-layer preview for pattern development.
 *
 * Loads a pattern's front/back litho masks, downsamples them to the canvas
 * grid, and recomposites live as you tilt (sliders or canvas drag), switch
 * illumination, or override substrate thickness / index. Line-spacing
 * experimentation: the pattern's numeric params are editable and
 * "Regenerate" round-trips POST /patterns/generate for fresh masks.
 *
 * Lab state is local (tilt, illumination, substrate overrides, params) — the
 * 3D scene is untouched until you press "Apply to <face>", which is the one
 * write into the box spec: it stamps the studied slug + tuned params onto a
 * face and lets App's debounced regen pick it up. Unmounting the panel (the
 * lab toggle) discards everything else, so apply before closing.
 */
export default function PatternLab() {
  const catalog = useStore((s) => s.catalog);
  const faces = useStore((s) => s.boxSpec.faces);
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const setSelectedFace = useStore((s) => s.setSelectedFace);
  const patchFace = useStore((s) => s.patchFace);
  const thumbnails = useStore((s) => s.thumbnails);
  const faceSlug = useStore((s) => s.boxSpec.faces[s.selectedFaceId]?.pattern_slug);
  const labSlug = useStore((s) => s.labSlug);
  const setLabSlug = useStore((s) => s.setLabSlug);
  const setLabOpen = useStore((s) => s.setLabOpen);

  // Default to the selected face's pattern until the user picks one.
  const slug = labSlug ?? faceSlug ?? catalog[0]?.slug ?? null;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Scale-up buffer for the reduced-resolution drag composite (putImageData
  // ignores the 2D transform, so the small grid has to go through drawImage).
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);
  // Latest pointer-derived tilt awaiting its animation frame. Pointermove
  // fires far faster than a composite takes, so moves coalesce here and only
  // the last one per frame becomes React state (and thus a repaint).
  const pendingTiltRef = useRef<{ x: number; y: number } | null>(null);
  const tiltRafRef = useRef<number | null>(null);
  // Monotonic request token, mirroring App.tsx's lastReqIdRef: every studied-
  // pattern change AND every Regenerate bumps it, and a response only lands if
  // its token is still current. Without it a slow cold generate (~2 s+) for the
  // previous slug applies its manifest over the newly selected pattern, leaving
  // the dropdown and the substrate/param readouts describing different patterns.
  const reqTokenRef = useRef(0);

  const [manifest, setManifest] = useState<PatternManifest | null>(null);
  const [masks, setMasks] = useState<MaskGrids | null>(null);
  /** True between pointerdown and pointerup on the canvas (drag resolution). */
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [regenBusy, setRegenBusy] = useState(false);
  // Load failures blank the canvas, so they belong ON the canvas; generation
  // failures belong next to the Regenerate button that caused them.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [regenError, setRegenError] = useState<string | null>(null);
  const [appliedTo, setAppliedTo] = useState<string | null>(null);

  const [tiltX, setTiltX] = useState(0);
  const [tiltY, setTiltY] = useState(0);
  const [illum, setIllum] = useState<Illumination2D>('ambient');
  // Local substrate overrides — seeded from the manifest, editable so
  // spacing/thickness interplay can be felt without touching the box spec.
  const [thicknessUm, setThicknessUm] = useState(500);
  const [n, setN] = useState(1.46);
  const [params, setParams] = useState<Record<string, number>>({});

  const descriptor = useMemo(
    () => catalog.find((c) => c.slug === slug) ?? null,
    [catalog, slug]
  );
  const numericParams = useMemo(
    () => descriptor?.params.filter((p) => p.type === 'float' || p.type === 'int') ?? [],
    [descriptor]
  );

  // Adopt a manifest: substrate readouts and param values reset to what the
  // backend actually materialized (generate merges class defaults).
  const applyManifest = (m: PatternManifest) => {
    setManifest(m);
    setThicknessUm(m.substrate.thickness_um);
    setN(m.substrate.n);
    const desc = useStore.getState().catalog.find((c) => c.slug === m.slug);
    const next: Record<string, number> = {};
    for (const p of desc?.params ?? []) {
      if (p.type !== 'float' && p.type !== 'int') continue;
      const v = m.params[p.name] ?? p.default;
      if (typeof v === 'number') next[p.name] = v;
    }
    setParams(next);
  };

  // Load the default variant whenever the studied pattern changes. The token
  // bump also invalidates any Regenerate still in flight for the old slug.
  useEffect(() => {
    if (!slug) return;
    let cancelled = false;
    const token = ++reqTokenRef.current;
    setLoading(true);
    setLoadError(null);
    setRegenError(null);
    getDefault(slug)
      .then((m) => {
        if (cancelled || reqTokenRef.current !== token) return;
        applyManifest(m);
        log('lab_pattern_loaded', { slug, variant: m.variant });
      })
      .catch((e) => {
        if (cancelled || reqTokenRef.current !== token) return;
        setLoadError(readableError((e as Error).message));
        log('lab_pattern_load_failed', { slug, error: (e as Error).message });
      })
      .finally(() => {
        if (!cancelled && reqTokenRef.current === token) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  const canvasH = useMemo(() => {
    if (!manifest) return CANVAS_W;
    const [ex, ey] = manifest.extent_um;
    return Math.max(64, Math.min(CANVAS_W * 2, Math.round((CANVAS_W * ey) / ex)));
  }, [manifest]);

  // Downsample both masks to the canvas grid (and the drag grid) whenever the
  // manifest changes.
  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    setMasks(null);
    const lowW = Math.max(1, Math.round(CANVAS_W / DRAG_DIVISOR));
    const lowH = Math.max(1, Math.round(canvasH / DRAG_DIVISOR));
    const sizes = [
      [CANVAS_W, canvasH],
      [lowW, lowH],
    ] as const;
    Promise.all([
      loadMaskGrids(manifest.files.front_png, sizes),
      loadMaskGrids(manifest.files.back_png, sizes),
    ])
      .then(([[front, frontLow], [back, backLow]]) => {
        if (!cancelled) {
          setMasks({ full: { front, back }, low: { front: frontLow, back: backLow } });
        }
      })
      .catch((e) => {
        if (!cancelled) setLoadError(readableError((e as Error).message));
      });
    return () => {
      cancelled = true;
    };
  }, [manifest, canvasH]);

  const shift = useMemo(
    () => parallaxShiftUm(tiltX, tiltY, thicknessUm, n),
    [tiltX, tiltY, thicknessUm, n]
  );

  // --- Zone calibration ----------------------------------------------------
  // Switch/reveal patterns (slit barriers, carrier reveals) live in shift
  // "zones" that repeat with the slit/carrier period p: the flip or reveal
  // completes within the FIRST half-period of back-mask shift and larger
  // tilts alias into repeated zones (±14° through 500 um silica is already
  // ~84 um ≈ 1-4 periods for p = 20-80 um, landing arbitrarily mid-zone).
  // When the manifest advertises its period we surface the shift measured in
  // periods and offer exact first-zone quick-sets (±p/2) via the inverse
  // Snell formula, so the effect is read at its design angle instead of an
  // aliased one.
  const zone = useMemo(() => {
    const rd = (manifest?.recipe_data ?? {}) as Record<string, unknown>;
    const slitP = rd.slit_period_um;
    const carrierP = rd.carrier_period_um;
    const p = slitP ?? carrierP;
    if (typeof p !== 'number' || !(p > 0)) return null;
    // Switch-axis convention matches the generators: 0° = vertical stripes,
    // shift along +x. Honor an axis hint when the manifest carries one.
    const axisRaw = rd.slit_axis_deg ?? rd.switch_axis_deg;
    return {
      periodUm: p,
      axisDeg: typeof axisRaw === 'number' ? axisRaw : 0,
      // Where the effect PEAKS, in fractions of the period. Empirically
      // audited: a slit barrier (open slit straddling the channel boundary)
      // completes its image switch at ±p/4 of back-mask shift — ±p/2 lands
      // back on a 50/50 blend. Carrier reveals peak at ±p/2 (anti-phase).
      switchFrac: typeof slitP === 'number' ? 0.25 : 0.5,
    };
  }, [manifest]);

  // Component of the parallax shift along the pattern's switch axis, in um.
  const zoneShiftUm = useMemo(() => {
    if (!zone) return 0;
    const rad = (zone.axisDeg * Math.PI) / 180;
    return shift.dxUm * Math.cos(rad) + shift.dyUm * Math.sin(rad);
  }, [zone, shift]);

  /** Set the tilt so the along-axis back-mask shift is exactly `sUm`. */
  const setZoneShift = (sUm: number) => {
    if (!zone) return;
    const rad = (zone.axisDeg * Math.PI) / 180;
    setTiltX(clampTilt(tiltForShiftUm(sUm * Math.cos(rad), thicknessUm, n)));
    setTiltY(clampTilt(tiltForShiftUm(sUm * Math.sin(rad), thicknessUm, n)));
  };

  // Live recomposite — pure 2D, no WebGL. um -> px through the extent so the
  // on-screen slide matches the physical shift fraction of the plate.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !masks || !manifest) return;
    // Drag runs on the reduced grid; every other trigger (sliders, zone
    // quick-sets, illumination, a fresh manifest, pointer release) is
    // full-resolution, so a measured readout always matches full-res pixels.
    const grid = dragging ? masks.low : masks.full;
    const dxPx = (shift.dxUm / manifest.extent_um[0]) * grid.front.width;
    const dyPx = (shift.dyUm / manifest.extent_um[1]) * grid.front.height;
    const rgba = compositeParallax(
      grid.front,
      grid.back,
      dxPx,
      dyPx,
      illum,
      DEFAULT_LASER_COLOR
    );
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    // Fresh allocation keeps ImageData happy under TS 5.7's ArrayBuffer-only
    // ImageDataArray type (compositeParallax returns ArrayBufferLike-typed).
    const px = new Uint8ClampedArray(rgba.length);
    px.set(rgba);
    const image = new ImageData(px, grid.front.width, grid.front.height);
    if (image.width === cv.width && image.height === cv.height) {
      ctx.putImageData(image, 0, 0);
      return;
    }
    // putImageData ignores the canvas transform and never scales, so the
    // reduced-resolution frame is staged and blitted up.
    const scratch =
      scratchRef.current ?? (scratchRef.current = document.createElement('canvas'));
    scratch.width = image.width;
    scratch.height = image.height;
    const sctx = scratch.getContext('2d');
    if (!sctx) return;
    sctx.putImageData(image, 0, 0);
    ctx.drawImage(scratch, 0, 0, cv.width, cv.height);
  }, [masks, manifest, shift, illum, dragging]);

  /**
   * Commit a pointer-derived tilt at most once per animation frame. Chromium
   * coalesces pointermove to the frame rate at best; without this every move
   * event that DOES arrive triggers a full composite + React render.
   */
  const queueTilt = (x: number, y: number) => {
    pendingTiltRef.current = { x, y };
    if (tiltRafRef.current !== null) return;
    tiltRafRef.current = requestAnimationFrame(() => {
      tiltRafRef.current = null;
      const p = pendingTiltRef.current;
      pendingTiltRef.current = null;
      if (!p) return;
      setTiltX(p.x);
      setTiltY(p.y);
    });
  };

  /** Apply any queued tilt immediately (drag end) and drop the pending frame. */
  const flushTilt = () => {
    if (tiltRafRef.current !== null) {
      cancelAnimationFrame(tiltRafRef.current);
      tiltRafRef.current = null;
    }
    const p = pendingTiltRef.current;
    pendingTiltRef.current = null;
    if (!p) return;
    setTiltX(p.x);
    setTiltY(p.y);
  };

  // A queued frame must not fire into an unmounted component.
  useEffect(
    () => () => {
      if (tiltRafRef.current !== null) cancelAnimationFrame(tiltRafRef.current);
    },
    []
  );

  // Drag on the canvas to tilt: pointer position maps linearly to tilt.
  const tiltFromPointer = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    queueTilt(clampTilt(nx * TILT_MAX_DEG), clampTilt(ny * TILT_MAX_DEG));
  };

  /** End a canvas drag: full-resolution repaint at the final tilt. */
  const endDrag = () => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    // Both setStates land in one React batch, so the release paints once.
    flushTilt();
    setDragging(false);
  };

  const regenerate = () => {
    if (!slug || regenBusy) return;
    const token = ++reqTokenRef.current;
    setRegenBusy(true);
    setRegenError(null);
    log('lab_regen_start', { slug, params });
    generatePattern(slug, params)
      .then((m) => {
        // generatePattern takes no AbortSignal, so a superseded response is
        // dropped here rather than cancelled on the wire.
        if (reqTokenRef.current !== token) {
          log('lab_regen_stale', { slug, variant: m.variant });
          return;
        }
        applyManifest(m);
        log('lab_regen_done', { slug, variant: m.variant });
      })
      .catch((e) => {
        const msg = (e as Error).message;
        log('lab_regen_failed', { slug, error: msg });
        if (reqTokenRef.current !== token) return;
        setRegenError(readableError(msg));
      })
      .finally(() => {
        if (reqTokenRef.current === token) setRegenBusy(false);
      });
  };

  // --- Lab -> box ----------------------------------------------------------
  // The one write out of the lab: stamp the studied slug + tuned params onto a
  // face so the tuning survives closing the panel (App's debounced regen picks
  // the spec change up). Non-numeric params (choice/bool) are not editable
  // here, so they are preserved only when the face already studies this slug —
  // on a slug change they are dropped exactly as FaceEditor's picker does,
  // letting the backend re-merge class defaults.
  const applyToFace = () => {
    const face = faces[selectedFaceId];
    if (!slug || !face) return;
    const patternParams =
      face.pattern_slug === slug
        ? { ...face.pattern_params, ...params }
        : { ...params };
    patchFace(selectedFaceId, { pattern_slug: slug, pattern_params: patternParams });
    log('lab_apply_to_face', { faceId: selectedFaceId, slug, params });
    setAppliedTo(FACE_LABELS[selectedFaceId]);
  };

  // Any later edit (or a fresh load/regen, which resets params) invalidates the
  // "applied" confirmation — what is on the face is no longer what is on screen.
  useEffect(() => {
    setAppliedTo(null);
  }, [slug, params, selectedFaceId]);

  return (
    <div
      data-testid="pattern-lab"
      style={{
        position: 'fixed',
        top: 48,
        right: 0,
        bottom: 0,
        width: 470,
        background: KIT.panel,
        borderLeft: `1px solid ${KIT.divider}`,
        boxShadow: '-8px 0 24px rgba(0,0,0,0.45)',
        zIndex: 30,
        overflowY: 'auto',
        padding: 12,
        fontSize: 12,
        color: KIT.text,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          marginBottom: 8,
        }}
      >
        <SubHeader>PATTERN LAB — 2D DUAL-LAYER PREVIEW</SubHeader>
        <Button testId="lab-close" title="Close the lab" onClick={() => setLabOpen(false)}>
          ✕
        </Button>
      </div>

      <SelectRow
        label="Pattern"
        value={slug ?? ''}
        options={catalog.map((c) => ({ value: c.slug, label: c.name }))}
        onChange={(v) => setLabSlug(v)}
        testId="lab-pattern"
      />
      {descriptor && (
        // Same context the FaceEditor grid gives: thumbnail + description, so
        // picking a pattern to study isn't blind. Only already-loaded
        // thumbnails are shown — the lab never kicks off the 16-slug sweep
        // itself (that's ~2 s of server-side generation each).
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 8 }}>
          {typeof thumbnails[descriptor.slug] === 'string' && (
            <img
              src={thumbnails[descriptor.slug] as string}
              alt={descriptor.name}
              style={{
                width: 44,
                height: 44,
                objectFit: 'cover',
                borderRadius: 4,
                border: `1px solid ${KIT.border}`,
                background: '#0b0d10',
                flex: '0 0 auto',
              }}
            />
          )}
          <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.4 }}>
            {descriptor.description}
          </div>
        </div>
      )}

      <div style={{ position: 'relative', marginBottom: 8 }}>
        <canvas
          data-testid="lab-canvas"
          ref={canvasRef}
          width={CANVAS_W}
          height={canvasH}
          onPointerDown={(e) => {
            draggingRef.current = true;
            setDragging(true);
            e.currentTarget.setPointerCapture(e.pointerId);
            tiltFromPointer(e);
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) tiltFromPointer(e);
          }}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          style={{
            width: '100%',
            border: `1px solid ${KIT.border}`,
            borderRadius: 4,
            background: '#0b0d10',
            cursor: 'crosshair',
            touchAction: 'none',
            display: 'block',
          }}
        />
        {(loading || !masks) && !loadError && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: 0.6,
            }}
          >
            Loading masks…
          </div>
        )}
        {loadError && (
          // ON the canvas, not at the panel bottom: a failed load leaves the
          // canvas blank or stale, and that is where the user is looking.
          <div
            data-testid="lab-load-error"
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              padding: 12,
              background: 'rgba(11,13,16,0.88)',
              color: KIT.error,
              lineHeight: 1.4,
            }}
          >
            Could not load this pattern: {loadError}
          </div>
        )}
      </div>
      <div style={{ opacity: 0.5, fontSize: 11, marginBottom: 8 }}>
        Drag on the preview to tilt; the back mask slides by the Snell parallax.
      </div>

      <SliderRow
        label="Tilt X"
        value={tiltX}
        min={-TILT_MAX_DEG}
        max={TILT_MAX_DEG}
        step={0.5}
        unit="°"
        decimals={1}
        onChange={(v) => setTiltX(clampTilt(v))}
        testId="lab-tilt-x"
      />
      <SliderRow
        label="Tilt Y"
        value={tiltY}
        min={-TILT_MAX_DEG}
        max={TILT_MAX_DEG}
        step={0.5}
        unit="°"
        decimals={1}
        onChange={(v) => setTiltY(clampTilt(v))}
        testId="lab-tilt-y"
      />
      <div data-testid="lab-shift-readout" style={{ opacity: 0.7, marginBottom: 8 }}>
        Back-mask shift: Δx {shift.dxUm.toFixed(2)} um · Δy {shift.dyUm.toFixed(2)} um
      </div>
      {zone && (
        <>
          <div data-testid="lab-zone-readout" style={{ opacity: 0.7, marginBottom: 6 }}>
            Zone shift: {zoneShiftUm >= 0 ? '+' : ''}
            {zoneShiftUm.toFixed(2)} um = {zoneShiftUm / zone.periodUm >= 0 ? '+' : ''}
            {(zoneShiftUm / zone.periodUm).toFixed(2)} × period (p ={' '}
            {zone.periodUm.toFixed(0)} um)
          </div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
            <Button
              testId="lab-zone-neg"
              title={`Tilt so the back mask slides exactly −p×${zone.switchFrac} along the switch axis (first-zone peak)`}
              onClick={() => setZoneShift(-zone.periodUm * zone.switchFrac)}
            >
              {zone.switchFrac === 0.25 ? '−p/4' : '−p/2'}
            </Button>
            <Button
              testId="lab-zone-zero"
              title="Head-on registration (zero shift)"
              onClick={() => setZoneShift(0)}
            >
              0
            </Button>
            <Button
              testId="lab-zone-pos"
              title={`Tilt so the back mask slides exactly +p×${zone.switchFrac} along the switch axis (first-zone peak)`}
              onClick={() => setZoneShift(zone.periodUm * zone.switchFrac)}
            >
              {zone.switchFrac === 0.25 ? '+p/4' : '+p/2'}
            </Button>
          </div>
          <div style={{ opacity: 0.5, fontSize: 11, marginBottom: 8 }}>
            Slit-barrier switches peak at ±p/4 of shift; carrier reveals at ±p/2.
            Both alias every full period — the quick-sets land the tilt exactly on
            the first-zone peak.
          </div>
        </>
      )}

      <SelectRow
        label="Illumination"
        value={illum}
        options={[
          { value: 'ambient', label: 'ambient' },
          { value: 'laser', label: 'laser' },
          { value: 'backlight', label: 'backlight' },
        ]}
        onChange={(v) => setIllum(v as Illumination2D)}
        testId="lab-illum"
      />

      <div style={{ borderTop: `1px solid ${KIT.divider}`, margin: '8px 0' }} />
      <SubHeader>SUBSTRATE (LOCAL OVERRIDE)</SubHeader>
      {manifest && (
        <div style={{ opacity: 0.6, fontSize: 11, marginBottom: 6 }}>
          Manifest: {manifest.substrate.material} · t ={' '}
          {manifest.substrate.thickness_um.toFixed(0)} um · n ={' '}
          {manifest.substrate.n.toFixed(2)}
        </div>
      )}
      <NumberRow
        label="Thickness"
        value={thicknessUm}
        min={0}
        step={50}
        unit="um"
        onChange={setThicknessUm}
        testId="lab-thickness"
      />
      <NumberRow
        label="Refractive index n"
        value={n}
        min={1}
        max={3}
        step={0.01}
        onChange={setN}
        testId="lab-n"
      />

      <div style={{ borderTop: `1px solid ${KIT.divider}`, margin: '8px 0' }} />
      <SubHeader>LINE SPACING / PARAMS</SubHeader>
      {numericParams.length === 0 && (
        <div style={{ opacity: 0.5, marginBottom: 8 }}>
          This pattern has no numeric params.
        </div>
      )}
      {numericParams.map((p) => (
        <NumberRow
          key={p.name}
          label={p.unit ? `${p.label} (${p.unit})` : p.label}
          value={params[p.name] ?? Number(p.default)}
          min={p.min}
          max={p.max}
          step={p.step ?? (p.type === 'int' ? 1 : 0.1)}
          onChange={(v) =>
            setParams((prev) => ({
              ...prev,
              [p.name]: p.type === 'int' ? Math.round(v) : v,
            }))
          }
          testId={`lab-param-${p.name}`}
        />
      ))}
      <Button
        testId="lab-regenerate"
        title="POST /patterns/generate with the params above, then recomposite"
        onClick={regenerate}
        disabled={regenBusy || loading || !slug}
      >
        {regenBusy ? 'Regenerating…' : 'Regenerate'}
      </Button>

      {/* Adjacent to the button that caused it, and verbatim: the backend's 422
          detail is written for the user (param bounds, litho floor, the 400k
          lattice budget's "increase period_um or shrink extent_um" guidance). */}
      {regenError && (
        <div
          data-testid="lab-error"
          style={{ color: KIT.error, marginTop: 8, lineHeight: 1.4 }}
        >
          {regenError}
        </div>
      )}

      <div style={{ borderTop: `1px solid ${KIT.divider}`, margin: '8px 0' }} />
      <SubHeader>APPLY TO THE BOX</SubHeader>
      <SelectRow
        label="Target face"
        value={selectedFaceId}
        options={FACE_IDS.filter((fid) => faces[fid]).map((fid) => ({
          value: fid,
          label: `${FACE_LABELS[fid]} — ${
            catalog.find((c) => c.slug === faces[fid]?.pattern_slug)?.name ??
            faces[fid]?.pattern_slug ??
            ''
          }`,
        }))}
        onChange={(v) => {
          // labSlug === null means "follow the selected face", so retargeting
          // would otherwise reload a different pattern and wipe the tuning the
          // user is about to apply. Pin the studied slug first.
          if (labSlug === null && slug) setLabSlug(slug);
          setSelectedFace(v as FaceId);
        }}
        testId="lab-apply-face"
      />
      <div style={{ opacity: 0.5, fontSize: 11, marginBottom: 8 }}>
        This is the studio's selected face — changing it here also switches the
        Faces rail editor.
      </div>
      <Button
        testId="lab-apply"
        title="Stamp the studied pattern + the params above onto this face of the box"
        onClick={applyToFace}
        disabled={!slug || regenBusy || !faces[selectedFaceId]}
      >
        Apply to {FACE_LABELS[selectedFaceId]}
      </Button>
      {appliedTo && (
        <div data-testid="lab-applied" style={{ opacity: 0.7, marginTop: 8 }}>
          Applied to {appliedTo} — the 3D preview is regenerating.
        </div>
      )}
      <div style={{ opacity: 0.5, fontSize: 11, marginTop: 8 }}>
        Everything else here (tilt, illumination, substrate overrides) is local to
        the lab and is discarded when the panel closes — apply first.
      </div>
    </div>
  );
}
