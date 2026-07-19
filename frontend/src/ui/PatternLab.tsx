import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { generatePattern, getDefault, type PatternManifest } from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import {
  DEFAULT_LASER_COLOR,
  compositeParallax,
  grayFromRgba,
  parallaxShiftUm,
  tiltForShiftUm,
  type Illumination2D,
  type ImageDataLike,
} from '../lab/composite2d';
import { Button, KIT, NumberRow, SelectRow, SliderRow, SubHeader } from './kit';

/** Target canvas width in px; height follows the pattern's extent aspect. */
const CANVAS_W = 420;
const TILT_MAX_DEG = 30;

const clampTilt = (deg: number): number =>
  Math.max(-TILT_MAX_DEG, Math.min(TILT_MAX_DEG, deg));

/**
 * Downsample a mask PNG into a single-channel grid at the lab canvas
 * resolution. This is the only DOM-touching step of the 2D pipeline — the
 * math itself (src/lab/composite2d.ts) stays canvas-free for unit tests.
 */
async function loadMaskGrid(url: string, w: number, h: number): Promise<ImageDataLike> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.drawImage(img, 0, 0, w, h);
  return grayFromRgba(ctx.getImageData(0, 0, w, h).data, w, h);
}

/**
 * Pattern Lab — a lightweight 2D dual-layer preview for pattern development.
 *
 * Loads a pattern's front/back litho masks, downsamples them to the canvas
 * grid, and recomposites live as you tilt (sliders or canvas drag), switch
 * illumination, or override substrate thickness / index. Line-spacing
 * experimentation: the pattern's numeric params are editable and
 * "Regenerate" round-trips POST /patterns/generate for fresh masks. All lab
 * state is local — the 3D scene's illumination and face specs are untouched.
 */
export default function PatternLab() {
  const catalog = useStore((s) => s.catalog);
  const faceSlug = useStore((s) => s.boxSpec.faces[s.selectedFaceId]?.pattern_slug);
  const labSlug = useStore((s) => s.labSlug);
  const setLabSlug = useStore((s) => s.setLabSlug);
  const setLabOpen = useStore((s) => s.setLabOpen);

  // Default to the selected face's pattern until the user picks one.
  const slug = labSlug ?? faceSlug ?? catalog[0]?.slug ?? null;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);

  const [manifest, setManifest] = useState<PatternManifest | null>(null);
  const [masks, setMasks] = useState<{ front: ImageDataLike; back: ImageDataLike } | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [regenBusy, setRegenBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  // Load the default variant whenever the studied pattern changes.
  useEffect(() => {
    if (!slug) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getDefault(slug)
      .then((m) => {
        if (cancelled) return;
        applyManifest(m);
        log('lab_pattern_loaded', { slug, variant: m.variant });
      })
      .catch((e) => {
        if (cancelled) return;
        setError((e as Error).message);
        log('lab_pattern_load_failed', { slug, error: (e as Error).message });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
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

  // Downsample both masks to the canvas grid whenever the manifest changes.
  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    setMasks(null);
    Promise.all([
      loadMaskGrid(manifest.files.front_png, CANVAS_W, canvasH),
      loadMaskGrid(manifest.files.back_png, CANVAS_W, canvasH),
    ])
      .then(([front, back]) => {
        if (!cancelled) setMasks({ front, back });
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
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
    const dxPx = (shift.dxUm / manifest.extent_um[0]) * masks.front.width;
    const dyPx = (shift.dyUm / manifest.extent_um[1]) * masks.front.height;
    const rgba = compositeParallax(
      masks.front,
      masks.back,
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
    ctx.putImageData(new ImageData(px, masks.front.width, masks.front.height), 0, 0);
  }, [masks, manifest, shift, illum]);

  // Drag on the canvas to tilt: pointer position maps linearly to tilt.
  const tiltFromPointer = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    setTiltX(clampTilt(nx * TILT_MAX_DEG));
    setTiltY(clampTilt(ny * TILT_MAX_DEG));
  };

  const regenerate = () => {
    if (!slug || regenBusy) return;
    setRegenBusy(true);
    setError(null);
    log('lab_regen_start', { slug, params });
    generatePattern(slug, params)
      .then((m) => {
        applyManifest(m);
        log('lab_regen_done', { slug, variant: m.variant });
      })
      .catch((e) => {
        setError((e as Error).message);
        log('lab_regen_failed', { slug, error: (e as Error).message });
      })
      .finally(() => setRegenBusy(false));
  };

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

      <div style={{ position: 'relative', marginBottom: 8 }}>
        <canvas
          data-testid="lab-canvas"
          ref={canvasRef}
          width={CANVAS_W}
          height={canvasH}
          onPointerDown={(e) => {
            draggingRef.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            tiltFromPointer(e);
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) tiltFromPointer(e);
          }}
          onPointerUp={() => {
            draggingRef.current = false;
          }}
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
        {(loading || !masks) && !error && (
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

      {error && (
        <div data-testid="lab-error" style={{ color: KIT.error, marginTop: 8 }}>
          {error}
        </div>
      )}
    </div>
  );
}
