import { useEffect, useRef, useState } from 'react';
import { FACE_IDS, type FaceId } from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import { captureTiltProgression, type ProgressionResult } from '../scene/tiltProgression';
import { Button, KIT } from './kit';

/**
 * Tilt-progression view: each face rendered in isolation across a tilt sweep,
 * at a scale that actually resolves its optical structure, beside a verdict on
 * whether the effect will be visible to a human eye on the fabricated part.
 *
 * The two judgements shown per face are independent on purpose:
 *   MEASURED  — contrast actually rendered in the strip below it. Comes from
 *               supersampled captures, because the live viewport cannot show
 *               these effects at all (it filters the two layers separately and
 *               loses the correlation term that IS the effect).
 *   PREDICTED — the perceptual budget from the FABRICATED geometry
 *               (GET /sim/readability/<box>): lane/grating subtense against
 *               acuity, pupil and diffraction blur against the switch
 *               separation, beat size against acuity. It never looks at a
 *               rendered pixel.
 * When they agree, believe them. When they disagree, the predicted one is the
 * one about the real object.
 */

type Check = {
  name: string;
  value: number | null;
  unit: string;
  limit: number;
  passes: boolean;
  note: string;
};
type FaceReport = {
  face: string;
  kind: string;
  passes: boolean;
  checks: Check[];
  summary: Record<string, number | string | null>;
};
type Verdict = {
  passes: boolean;
  faces: FaceReport[];
  viewing: { distance_mm: number; pupil_mm: number };
  glass: { thickness_um: number; n: number };
};

const FIELD_PRESETS = [
  { label: 'Whole face', mm: 46 },
  { label: 'Centrepiece', mm: 30 },
  { label: 'Detail', mm: 10 },
];

export default function ProgressionView() {
  const open = useStore((s) => s.progressionOpen);
  const setOpen = useStore((s) => s.setProgressionOpen);
  const boxSpec = useStore((s) => s.boxSpec);
  const boxManifest = useStore((s) => s.boxManifest);

  const [fieldMm, setFieldMm] = useState(30);
  const [distanceMm, setDistanceMm] = useState(300);
  const [results, setResults] = useState<Record<string, ProgressionResult>>({});
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelled = useRef(false);

  // Predicted verdict comes from the backend, keyed on the live box.
  useEffect(() => {
    if (!open || !boxManifest) return;
    let live = true;
    (async () => {
      try {
        const r = await fetch(
          `/sim/readability/${encodeURIComponent(boxManifest.id)}?distance_mm=${distanceMm}`
        );
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = (await r.json()) as Verdict;
        if (live) setVerdict(j);
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    })();
    return () => {
      live = false;
    };
  }, [open, boxManifest, distanceMm]);

  useEffect(() => {
    if (!open) cancelled.current = true;
  }, [open]);

  const runAll = async (): Promise<void> => {
    const studio = (window as unknown as { __studio?: Parameters<typeof captureTiltProgression>[0] })
      .__studio;
    if (!studio || !boxManifest) return;
    cancelled.current = false;
    setError(null);
    const next: Record<string, ProgressionResult> = {};
    for (const fid of FACE_IDS) {
      if (cancelled.current) break;
      const fm = boxManifest.faces[fid as FaceId];
      if (!fm) continue;
      setBusy(fid);
      try {
        // Yield so the "capturing…" label paints before the blocking renders.
        await new Promise((r) => setTimeout(r, 30));
        const res = await captureTiltProgression(
          studio,
          boxSpec,
          fid as FaceId,
          fm.spec.pattern_slug,
          { fieldMm }
        );
        next[fid] = res;
        setResults({ ...next });
      } catch (e) {
        setError(`${fid}: ${(e as Error).message}`);
      }
    }
    setBusy(null);
    log('progression_captured', { faces: Object.keys(next).length, fieldMm });
  };

  if (!open) return null;

  const faceVerdicts = (fid: string): FaceReport[] =>
    (verdict?.faces ?? []).filter((f) => f.face === fid);

  return (
    <div
      data-testid="progression-view"
      style={{
        position: 'absolute',
        inset: 0,
        background: 'rgba(8,10,14,0.97)',
        zIndex: 40,
        display: 'flex',
        flexDirection: 'column',
        color: KIT.text,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '10px 14px',
          borderBottom: `1px solid ${KIT.border}`,
          flexWrap: 'wrap',
        }}
      >
        <strong style={{ fontSize: 13 }}>Tilt progression</strong>
        <span style={{ opacity: 0.6, fontSize: 11 }}>
          each face alone, swept −6°…+6°, supersampled so the lattice resolves
        </span>
        <div style={{ flex: 1 }} />
        <label style={{ fontSize: 11, opacity: 0.8 }}>
          Field&nbsp;
          <select
            data-testid="progression-field"
            value={fieldMm}
            onChange={(e) => setFieldMm(Number(e.target.value))}
            style={{ background: KIT.field, color: KIT.text, border: `1px solid ${KIT.border}` }}
          >
            {FIELD_PRESETS.map((f) => (
              <option key={f.mm} value={f.mm}>
                {f.label} ({f.mm} mm)
              </option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 11, opacity: 0.8 }}>
          Viewing&nbsp;
          <select
            value={distanceMm}
            onChange={(e) => setDistanceMm(Number(e.target.value))}
            style={{ background: KIT.field, color: KIT.text, border: `1px solid ${KIT.border}` }}
          >
            {[250, 300, 400, 500].map((d) => (
              <option key={d} value={d}>
                {d} mm
              </option>
            ))}
          </select>
        </label>
        <Button onClick={runAll} testId="progression-run">
          {busy ? `Capturing ${busy}…` : 'Capture all faces'}
        </Button>
        <Button onClick={() => setOpen(false)} testId="progression-close">
          Close
        </Button>
      </div>

      {error && (
        <div style={{ padding: '6px 14px', color: KIT.error, fontSize: 11 }}>{error}</div>
      )}

      <div style={{ flex: 1, overflow: 'auto', padding: 14 }}>
        {Object.keys(results).length === 0 && !busy && (
          <div style={{ opacity: 0.65, fontSize: 12, maxWidth: 720, lineHeight: 1.5 }}>
            Press <strong>Capture all faces</strong>. Each face is rendered alone at{' '}
            {2048 / 340 >= 6 ? '6×' : '4×'} supersampling and swept through tilt, then averaged
            down — that average is the same integration your eye performs, so the strip predicts
            what a person sees rather than what the live viewport can draw. The verdict beside
            each strip is computed independently from the fabricated pitches, not from these
            pixels.
          </div>
        )}

        {FACE_IDS.map((fid) => {
          const res = results[fid];
          if (!res) return null;
          const reports = faceVerdicts(fid);
          // Judge on EFFECT STRENGTH, not mean brightness: an A/B swap can
          // invert half the face while the mean barely moves.
          const measuredVisible = res.effectStrength >= 0.02;
          return (
            <div
              key={fid}
              data-testid={`progression-row-${fid}`}
              style={{ marginBottom: 22, borderBottom: `1px solid ${KIT.border}`, paddingBottom: 14 }}
            >
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 6 }}>
                <strong style={{ textTransform: 'uppercase', fontSize: 12, letterSpacing: 1 }}>
                  {fid}
                </strong>
                <span style={{ opacity: 0.65, fontSize: 11 }}>{res.slug}</span>
                <span style={{ opacity: 0.5, fontSize: 11 }}>
                  {res.fieldMm} mm field · {res.supersample.toFixed(1)}× supersample
                </span>
              </div>

              <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
                <div style={{ display: 'flex', gap: 4, overflowX: 'auto', paddingBottom: 6 }}>
                  {res.frames.map((f) => (
                    <figure key={f.angleDeg} style={{ margin: 0, textAlign: 'center' }}>
                      <img
                        src={f.dataUrl}
                        width={150}
                        height={150}
                        alt={`${fid} at ${f.angleDeg}°`}
                        style={{ display: 'block', borderRadius: 3, border: `1px solid ${KIT.border}` }}
                      />
                      <figcaption style={{ fontSize: 10, opacity: 0.6, marginTop: 2 }}>
                        {f.angleDeg > 0 ? '+' : ''}
                        {f.angleDeg}°
                      </figcaption>
                    </figure>
                  ))}
                </div>

                <div style={{ minWidth: 300, fontSize: 11 }}>
                  <div style={{ marginBottom: 6 }}>
                    <span style={{ opacity: 0.6 }}>Measured in these tiles: </span>
                    <strong style={{ color: measuredVisible ? KIT.accent : KIT.error }}>
                      {(res.effectStrength * 100).toFixed(1)}% effect strength
                    </strong>
                    <span style={{ opacity: 0.6 }}>
                      {' '}({(res.changedFrac * 100).toFixed(0)}% of pixels change −6°→+6°,
                      mean brightness swing {(res.measuredContrast * 100).toFixed(1)}%)
                    </span>
                  </div>
                  {reports.map((r) => (
                    <div key={r.kind} style={{ marginBottom: 6 }}>
                      <div style={{ opacity: 0.75, marginBottom: 2 }}>
                        Predicted for the fabricated part — <strong>{r.kind}</strong>{' '}
                        <span style={{ color: r.passes ? KIT.accent : KIT.error }}>
                          {r.passes ? 'reads' : 'at risk'}
                        </span>
                      </div>
                      {r.checks.map((c) => (
                        <div
                          key={c.name}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 8,
                            opacity: c.passes ? 0.6 : 1,
                            color: c.passes ? KIT.text : KIT.error,
                          }}
                          title={c.note}
                        >
                          <span>{c.name}</span>
                          <span style={{ fontVariantNumeric: 'tabular-nums' }}>
                            {c.value === null ? '—' : c.value.toFixed(2)} {c.unit}
                            <span style={{ opacity: 0.5 }}> / {c.limit}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
