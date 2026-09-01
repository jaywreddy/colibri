import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchCollage,
  listPatterns,
  type CollageSheet,
  type PatternDescriptor,
} from '../api';
import { FACE_IDS, type FaceId } from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import { KIT } from './kit';

/**
 * Collage view: every catalogue pattern, composited across a fan of view
 * angles, so the effects can be confirmed by eye instead of taken on trust.
 *
 * This is deliberately NOT a render of the 3D scene. The WebGL path filters
 * each layer independently and so computes <front>*<back>, which loses the
 * cross term that IS the effect the moment the lattice drops below a screen
 * pixel — the reason the live viewport shows the frame moiré at ~1% contrast
 * and the A/B interlace as a static blend. The backend forms the product at
 * raster resolution and area-averages afterwards, which is the order the eye
 * integrates in. A pattern that does nothing on these strips does nothing in
 * glass.
 *
 * SCOPE: these strips sweep the PATTERN's own front/back pair, at the periods
 * that actually get written. They are not the composed plate. A composed plate
 * wraps the pattern in a frame carrier — so a single-layer pattern reads as NO
 * EFFECT here while its plate still carries a frame moiré. The composed-plate
 * raster cannot be substituted: it is a coarse preview (33 um pixels around a
 * 22 um carrier), so sweeping it would measure the preview's aliasing rather
 * than the part.
 *
 * Sheets load ONE AT A TIME on purpose. Each is a full sweep of full-raster
 * composites, and this host bugchecks when two heavy computes overlap.
 */

const SPANS = [3, 6, 12] as const;
const STEP_CHOICES = [7, 9, 13] as const;
const ILLUMS = ['ambient', 'backlight', 'laser'] as const;
const AXES = ['auto', 'x', 'y'] as const;

/**
 * Strength bands. A sweep that moves less than a couple of percent of mean
 * luminance is not something a person will notice across a wrist tilt; above
 * ~15% it reads as an unmistakable change of image.
 */
const STRONG = 0.15;
const WEAK = 0.02;

type Row = {
  slug: string;
  name: string;
  sheet?: CollageSheet;
  error?: string;
  state: 'idle' | 'loading' | 'done' | 'error';
};

function verdictOf(row: Row): { label: string; color: string; note: string } {
  if (row.state === 'error') return { label: 'FAILED', color: '#e06c6c', note: row.error ?? '' };
  if (!row.sheet) return { label: '—', color: KIT.border, note: '' };
  const { effect_strength: s, changed_frac: c } = row.sheet.metrics;
  if (s >= STRONG) {
    return { label: 'STRONG', color: '#6fcf7f', note: `${Math.round(c * 100)}% of pixels change` };
  }
  if (s >= WEAK) {
    return { label: 'WEAK', color: KIT.accent, note: `${Math.round(c * 100)}% of pixels change` };
  }
  return {
    label: 'STATIC',
    color: '#c98b3a',
    note: 'single-layer art — the plate’s frame carrier still shimmers, and is not in this sheet',
  };
}

export default function CollageView() {
  const open = useStore((s) => s.collageOpen);
  const setOpen = useStore((s) => s.setCollageOpen);
  const boxManifest = useStore((s) => s.boxManifest);

  // Which of the box's six faces currently carry each pattern — the question
  // behind this view is whether THE BOX reads, not just whether the catalogue
  // does, so a pattern that no face uses should be visibly incidental.
  const facesFor = (slug: string): string[] =>
    FACE_IDS.filter((f) => boxManifest?.faces?.[f as FaceId]?.spec.pattern_slug === slug);

  const [rows, setRows] = useState<Row[]>([]);
  const [spanDeg, setSpanDeg] = useState<number>(6);
  const [steps, setSteps] = useState<number>(9);
  const [illum, setIllum] = useState<string>('ambient');
  const [axis, setAxis] = useState<'auto' | 'x' | 'y'>('auto');
  const [running, setRunning] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const cancelled = useRef(false);
  // Object URLs are only reclaimed when revoked; a re-sweep of 17 sheets would
  // otherwise leak every previous blob for the lifetime of the tab.
  const urls = useRef<string[]>([]);
  const revokeAll = () => {
    urls.current.forEach((u) => URL.revokeObjectURL(u));
    urls.current = [];
  };

  useEffect(() => {
    if (!open) {
      cancelled.current = true;
      return;
    }
    if (rows.length) return;
    let dead = false;
    listPatterns()
      .then((ps: PatternDescriptor[]) => {
        if (dead) return;
        setRows(ps.map((p) => ({ slug: p.slug, name: p.name, state: 'idle' })));
      })
      .catch((e: Error) => !dead && setCatalogError(e.message));
    return () => {
      dead = true;
    };
  }, [open, rows.length]);

  useEffect(() => () => revokeAll(), []);

  const runAll = useCallback(async () => {
    if (!rows.length) return;
    cancelled.current = false;
    setRunning(true);
    revokeAll();
    setRows((rs) => rs.map((r) => ({ ...r, sheet: undefined, error: undefined, state: 'idle' })));

    for (const row of rows) {
      if (cancelled.current) break;
      setRows((rs) => rs.map((r) => (r.slug === row.slug ? { ...r, state: 'loading' } : r)));
      try {
        // Sequential by construction — see the note at the top of the file.
        const sheet = await fetchCollage(row.slug, {
          spanDeg,
          steps,
          illum,
          axis,
          tilePx: 130,
          cols: steps,
          // The row header already names the pattern, axis and light.
          showTitle: false,
        });
        urls.current.push(sheet.url);
        setRows((rs) => rs.map((r) => (r.slug === row.slug ? { ...r, sheet, state: 'done' } : r)));
      } catch (e) {
        setRows((rs) =>
          rs.map((r) =>
            r.slug === row.slug ? { ...r, error: (e as Error).message, state: 'error' } : r
          )
        );
      }
    }
    setRunning(false);
    log('collage_swept', { patterns: rows.length, spanDeg, steps, illum, axis });
  }, [rows, spanDeg, steps, illum, axis]);

  if (!open) return null;

  const done = rows.filter((r) => r.state === 'done');
  const strong = done.filter((r) => (r.sheet?.metrics.effect_strength ?? 0) >= STRONG).length;
  const dead = done.filter((r) => (r.sheet?.metrics.effect_strength ?? 0) < WEAK).length;

  const chip = <T extends string | number>(
    value: T,
    current: T,
    set: (v: T) => void,
    label?: string
  ) => (
    <button
      key={String(value)}
      onClick={() => set(value)}
      disabled={running}
      style={{
        padding: '3px 9px',
        fontSize: 11,
        background: value === current ? KIT.accent : '#141820',
        color: value === current ? '#101216' : KIT.text,
        border: `1px solid ${value === current ? KIT.accent : KIT.border}`,
        borderRadius: 4,
        cursor: running ? 'default' : 'pointer',
        opacity: running ? 0.5 : 1,
      }}
    >
      {label ?? String(value)}
    </button>
  );

  return (
    <div
      data-testid="collage-view"
      style={{
        // Window-level, not scoped to the centre column: a strip of 9-13 tiles
        // needs the full width to stay legible.
        position: 'fixed',
        inset: 0,
        background: '#0b0d10',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '10px 14px',
          borderBottom: `1px solid ${KIT.border}`,
          background: '#0f1218',
          flexWrap: 'wrap',
        }}
      >
        <strong style={{ fontSize: 13 }}>Tilt collage</strong>
        <span style={{ fontSize: 11, opacity: 0.6, maxWidth: 430 }}>
          Every pattern composited across a fan of view angles. The product is formed at raster
          resolution, so this shows what the eye integrates — not what the 3D preview can draw.
          These are the pattern&rsquo;s own two layers; the frame carrier a composed plate adds
          around them is not included.
        </span>

        <span style={{ fontSize: 11, opacity: 0.7, marginLeft: 8 }}>span</span>
        {SPANS.map((v) => chip(v, spanDeg, setSpanDeg, `±${v}°`))}
        <span style={{ fontSize: 11, opacity: 0.7 }}>steps</span>
        {STEP_CHOICES.map((v) => chip(v, steps, setSteps))}
        <span style={{ fontSize: 11, opacity: 0.7 }}>light</span>
        {ILLUMS.map((v) => chip(v, illum, setIllum))}
        <span style={{ fontSize: 11, opacity: 0.7 }}>tilt</span>
        {AXES.map((v) => chip(v, axis, setAxis))}

        <button
          data-testid="collage-run"
          onClick={runAll}
          disabled={running || !rows.length}
          style={{
            padding: '5px 14px',
            fontSize: 12,
            background: running ? '#141820' : KIT.accent,
            color: running ? KIT.text : '#101216',
            border: `1px solid ${KIT.accent}`,
            borderRadius: 4,
            cursor: running || !rows.length ? 'default' : 'pointer',
          }}
        >
          {running ? 'Sweeping…' : done.length ? 'Re-sweep' : 'Sweep all patterns'}
        </button>

        {done.length > 0 && (
          <span data-testid="collage-summary" style={{ fontSize: 11, opacity: 0.8 }}>
            {done.length} swept · <b style={{ color: '#6fcf7f' }}>{strong} strong</b> ·{' '}
            <b style={{ color: '#c98b3a' }}>{dead} static</b>
          </span>
        )}

        <div style={{ flex: 1 }} />
        <button
          data-testid="collage-close"
          onClick={() => {
            cancelled.current = true;
            setOpen(false);
          }}
          style={{
            padding: '4px 12px',
            fontSize: 12,
            background: '#141820',
            color: KIT.text,
            border: `1px solid ${KIT.border}`,
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          Close
        </button>
      </header>

      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: 14 }}>
        {catalogError && (
          <div style={{ color: '#e06c6c', fontSize: 12 }}>Catalog failed: {catalogError}</div>
        )}
        {!running && !done.length && !catalogError && (
          <div style={{ fontSize: 12, opacity: 0.6, padding: 8 }}>
            {rows.length} patterns ready. Each sweep composites the full raster once per angle and
            runs on its own — the host cannot take two heavy computes at once, so this is
            sequential and takes a moment.
          </div>
        )}

        {rows.map((row) => {
          const v = verdictOf(row);
          const m = row.sheet?.metrics;
          return (
            <section
              key={row.slug}
              data-testid={`collage-row-${row.slug}`}
              style={{
                marginBottom: 14,
                border: `1px solid ${KIT.border}`,
                borderRadius: 6,
                background: '#0f1218',
                opacity: row.state === 'idle' ? 0.45 : 1,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '7px 10px',
                  borderBottom: row.sheet ? `1px solid ${KIT.border}` : 'none',
                  fontSize: 12,
                  flexWrap: 'wrap',
                }}
              >
                <b>{row.name}</b>
                <code style={{ fontSize: 10, opacity: 0.5 }}>{row.slug}</code>
                {facesFor(row.slug).map((f) => (
                  <span
                    key={f}
                    title={`this pattern is on the ${f} face of the current box`}
                    style={{
                      padding: '1px 6px',
                      borderRadius: 3,
                      fontSize: 10,
                      background: '#1d2434',
                      border: `1px solid ${KIT.border}`,
                    }}
                  >
                    {f}
                  </span>
                ))}
                <span
                  data-testid={`collage-verdict-${row.slug}`}
                  style={{
                    padding: '1px 7px',
                    borderRadius: 3,
                    fontSize: 10,
                    fontWeight: 700,
                    letterSpacing: 0.4,
                    color: v.color,
                    border: `1px solid ${v.color}`,
                  }}
                >
                  {v.label}
                </span>
                {row.state === 'loading' && (
                  <span style={{ fontSize: 11, opacity: 0.7 }}>compositing…</span>
                )}
                {m && (
                  <span style={{ fontSize: 11, opacity: 0.75 }}>
                    strength {(m.effect_strength * 100).toFixed(1)}% · tilts about{' '}
                    <b>{m.axis}</b>
                    {m.axis_mode === 'auto' ? ' (chosen)' : ''} · strongest between{' '}
                    {m.peak_pair_deg[0]}° and {m.peak_pair_deg[1]}°
                  </span>
                )}
                <div style={{ flex: 1 }} />
                <span style={{ fontSize: 10, opacity: 0.55 }}>{v.note}</span>
              </div>

              {row.sheet && (
                <img
                  src={row.sheet.url}
                  alt={`${row.name} across ${steps} view angles`}
                  style={{ display: 'block', width: '100%', height: 'auto' }}
                />
              )}
              {row.error && (
                <div style={{ color: '#e06c6c', fontSize: 11, padding: '6px 10px' }}>
                  {row.error}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
