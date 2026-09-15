import { useEffect, useMemo, useRef, useState } from 'react';
import {
  generateBox,
  getBox,
  listBoxes,
  listPatterns,
  type BoxManifest,
  type BoxSpec,
} from './api';
import { validateBox } from './assembly';
import { log } from './logger';
import { useStore, LID_MAX_DEG } from './store';
import BoxScene from './scene/BoxScene';
import BuildPanel from './ui/BuildPanel';
import FacesPanel from './ui/FacesPanel';
import { BUTTON_STYLE, INPUT_STYLE, KIT } from './ui/kit';

function useDebounce<T extends (...args: never[]) => void>(fn: T, ms: number): T {
  const timer = useRef<number | null>(null);
  const latest = useRef(fn);
  latest.current = fn;
  return useMemo(
    () =>
      ((...args: Parameters<T>) => {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => latest.current(...args), ms);
      }) as T,
    [ms]
  );
}

/**
 * Order-independent serialization of any JSON-ish value.
 *
 * The live spec is built by object literals while a manifest's spec came back
 * through the backend's own `to_dict` key order, so plain JSON.stringify would
 * report two identical designs as different. Sorting keys at every level makes
 * the two comparable.
 */
function canonicalJson(v: unknown): string {
  if (v === undefined) return 'null'; // an absent knob and an unset one are one design
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`;
  const obj = v as Record<string, unknown>;
  return `{${Object.keys(obj)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonicalJson(obj[k])}`)
    .join(',')}}`;
}

/**
 * The regenerate identity of a box spec: every persisted field that changes
 * what the backend would produce. Hinge, bead and finish don't change masks,
 * but they DO change the saved manifest — ASSEMBLY.md's hinge cut list and
 * finish come from it — so they participate too. Hinge/foil-only regens are
 * cheap: the backend per-face plate caches hit and only the manifest assembly
 * block is recomputed.
 *
 * `label` is deliberately OUT (naming a preset must not invalidate the held
 * manifest), as are lid angle and layout, which are view-only state.
 */
function specRegenKey(spec: BoxSpec): string {
  return canonicalJson({
    w: spec.width_um,
    d: spec.depth_um,
    h: spec.height_um,
    glass: spec.glass,
    foil: spec.foil,
    hinge: spec.hinge,
    carrier_pitch_um: spec.carrier_pitch_um,
    faces: spec.faces,
  });
}

/**
 * Banner text for a thrown request error. api.ts already turns HTTP failures
 * into the endpoint's own `detail` sentence; what's left to translate is
 * fetch's opaque network TypeError, which a user reads as gibberish even
 * though it's the one case the app recovers from by itself.
 */
function friendlyError(e: unknown): string {
  const err = e as Error;
  if (
    err.name === 'TypeError' ||
    /failed to fetch|networkerror|load failed/i.test(err.message)
  ) {
    return 'Backend not responding on :8765 — retrying automatically';
  }
  return err.message;
}

/**
 * Ring Box Studio — single-purpose, box-first studio screen.
 *
 * Any persisted spec change (dims, glass, foil, hinge, faces) triggers a
 * debounced POST /boxes/generate with a stale-response guard, keeping the
 * held manifest — and therefore the six composed plates the scene samples —
 * in sync with the on-screen design. Hinge/bead/finish edits still update the
 * scene instantly via src/assembly.ts; their regen only refreshes the manifest
 * (per-face plate caches hit, no mask recompute). Lid angle and layout are
 * view-only and never hit the backend.
 *
 * The FAB BUNDLE is no longer built from here. A cold export is six plate
 * composes and six DRC-healed fab masks — minutes of the host's one heavy
 * compute slot — and pulling it through a browser download bought nothing but
 * a progress strip and a staleness gate. It is a CLI step now, run against the
 * box id this screen shows, so the archive can never be a stale echo of a
 * design the user has since changed.
 */
export default function App() {
  const boxSpec = useStore((s) => s.boxSpec);
  const boxManifest = useStore((s) => s.boxManifest);
  const setBoxManifest = useStore((s) => s.setBoxManifest);
  const setBoxSpec = useStore((s) => s.setBoxSpec);
  const catalog = useStore((s) => s.catalog);
  const setCatalog = useStore((s) => s.setCatalog);
  const lidTargetDeg = useStore((s) => s.lidTargetDeg);
  const setLidTargetDeg = useStore((s) => s.setLidTargetDeg);
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);
  const autoRotate = useStore((s) => s.autoRotate);
  const setAutoRotate = useStore((s) => s.setAutoRotate);
  const inspectMode = useStore((s) => s.inspectMode);
  const setInspectMode = useStore((s) => s.setInspectMode);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Mirrors regenFailsRef > 0 for the banner: the app self-heals on a 5 s
  // heartbeat, and a user who can't see that reloads or restarts servers.
  const [retrying, setRetrying] = useState(false);
  const [savedBoxes, setSavedBoxes] = useState<BoxManifest[]>([]);
  const [presetName, setPresetName] = useState('');
  const lastReqIdRef = useRef(0);
  // Consecutive regen failures — drives the backend-warmup retry backoff.
  const regenFailsRef = useRef(0);
  // regenKey of the spec we POSTed for the manifest currently held: what
  // actually produced the manifest on screen, independent of how the backend's
  // normalize_face_dims spells the spec back.
  const builtFromKeyRef = useRef<string | null>(null);

  // Pattern catalog — fetched at boot for the face editors. Retries with
  // backoff: under the combined `app` launcher Vite is ready in ~0.5 s while
  // uvicorn takes a few seconds, so the first fetches can hit a dead proxy.
  // Without retry the app sits on blank (black) faces forever.
  useEffect(() => {
    if (catalog.length > 0) return;
    let cancelled = false;
    let timer: number | null = null;
    let attempt = 0;
    const load = () => {
      listPatterns()
        .then((pts) => {
          if (cancelled) return;
          setCatalog(pts);
          log('catalog_loaded', { count: pts.length, attempt });
        })
        .catch((e) => {
          if (cancelled) return;
          attempt += 1;
          // Never give up: a dev-server restart can bring the backend back
          // minutes later, and a capped retry left the app on black faces
          // forever. Settle into a gentle 5 s heartbeat after the first burst.
          if (attempt % 10 === 0) {
            log('catalog_load_retrying', { error: (e as Error).message, attempt });
          }
          timer = window.setTimeout(load, Math.min(500 * attempt, 5000));
        });
    };
    load();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [catalog.length, setCatalog]);

  const validationErrors = useMemo(() => validateBox(boxSpec), [boxSpec]);

  // See specRegenKey for what participates and why.
  const regenKey = useMemo(() => specRegenKey(boxSpec), [boxSpec]);

  const regen = useDebounce(async () => {
    const spec = useStore.getState().boxSpec;
    const errors = validateBox(spec);
    if (errors.length > 0) {
      setError(`${errors.length} spec error${errors.length === 1 ? '' : 's'} — ${errors[0]}`);
      log('box_regen_skipped_invalid', { errors });
      return;
    }
    const key = specRegenKey(spec);
    const reqId = ++lastReqIdRef.current;
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    log('box_regen_start', { reqId });
    try {
      const m = await generateBox(spec);
      // Drop stale responses if another request fired since we started.
      if (lastReqIdRef.current !== reqId) {
        log('box_regen_stale', { reqId });
        return;
      }
      builtFromKeyRef.current = key;
      setBoxManifest(m);
      regenFailsRef.current = 0;
      setRetrying(false);
      log('box_regen_done', {
        id: m.id,
        duration_ms: Math.round(performance.now() - t0),
      });
    } catch (e) {
      const err = e as Error;
      if (lastReqIdRef.current === reqId) {
        setError(friendlyError(err));
        setRetrying(true);
        // Backend-warmup retry: under the combined `app` launcher the first
        // generate can race uvicorn's startup (proxy 500/ECONNREFUSED), and a
        // dev-server restart can take the backend down for minutes. Never give
        // up — settle into a 5 s heartbeat; the stale-response guard makes
        // overlapping retries harmless and a success resets the counter. (A
        // genuinely invalid spec never reaches here: regen() validates first.)
        regenFailsRef.current += 1;
        const delay = Math.min(600 * regenFailsRef.current, 5000);
        if (regenFailsRef.current <= 3 || regenFailsRef.current % 10 === 0) {
          log('box_regen_retry', { attempt: regenFailsRef.current, delay_ms: delay });
        }
        window.setTimeout(() => {
          if (lastReqIdRef.current === reqId) regen();
        }, delay);
      }
      log('box_regen_failed', { error: err.message });
    } finally {
      if (lastReqIdRef.current === reqId) setBusy(false);
    }
  }, 400);

  useEffect(() => {
    regen();
  }, [regenKey, regen]);

  useEffect(() => {
    listBoxes().then(setSavedBoxes).catch(() => {});
  }, [boxManifest]);

  const savePreset = async () => {
    const name = presetName.trim();
    if (!name) {
      setError('Give the preset a name to save it.');
      return;
    }
    // Slug must satisfy the backend's box_id safety rules: [a-z0-9-] only,
    // no leading '-', max 64 chars — anything else 400s (path-traversal guard).
    const slug = name
      .toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/[^a-z0-9-]/g, '')
      .replace(/^-+|-+$/g, '')
      .slice(0, 64);
    if (!slug) {
      setError('Preset name must contain at least one letter or digit.');
      return;
    }
    try {
      const spec = { ...useStore.getState().boxSpec, label: name };
      const m = await generateBox(spec, { boxId: slug });
      // Naming a preset doesn't change the design, so this manifest is just as
      // exportable as the scratch one it replaces (`label` is out of the key).
      builtFromKeyRef.current = specRegenKey(spec);
      setBoxManifest(m);
      setSavedBoxes(await listBoxes());
      log('box_saved', { id: m.id, name: m.name });
    } catch (e) {
      setError(friendlyError(e));
    }
  };

  const loadPreset = async (id: string) => {
    if (!id) return;
    try {
      const m = await getBox(id);
      // No POST produced this one — staleness falls back to the manifest's own
      // spec, which setBoxSpec is about to mirror into the live spec.
      builtFromKeyRef.current = null;
      setBoxSpec(m.spec);
      setBoxManifest(m);
      setPresetName(m.name);
      log('box_loaded', { id: m.id });
    } catch (e) {
      setError(friendlyError(e));
    }
  };


  return (
    <div
      style={{
        display: 'grid',
        gridTemplateRows: '48px 1fr',
        height: '100vh',
        width: '100vw',
      }}
    >
      <header
        style={{
          borderBottom: '1px solid #22262d',
          display: 'flex',
          alignItems: 'center',
          padding: '0 16px',
          gap: 12,
          background: '#0f1218',
        }}
      >
        <div style={{ fontWeight: 700, letterSpacing: 0.3 }}>Ring Box Studio</div>
        <div style={{ fontSize: 11, opacity: 0.6 }}>
          Fused-silica plates · gold-on-quartz masks · copper foil + solder
        </div>
        <div style={{ flex: 1 }} />
        {busy && (
          <span data-testid="regen-status" style={{ fontSize: 12, opacity: 0.7 }}>
            Regenerating…
          </span>
        )}
        {error && (
          <span
            data-testid="regen-error"
            title={error}
            style={{
              color: KIT.error,
              fontSize: 12,
              maxWidth: 340,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {error}
            {retrying && (
              <span data-testid="regen-retrying" style={{ opacity: 0.75 }}>
                {' · retrying…'}
              </span>
            )}
          </span>
        )}
        <input
          placeholder="Preset name"
          value={presetName}
          onChange={(e) => setPresetName(e.target.value)}
          style={{ ...INPUT_STYLE, width: 140 }}
          data-testid="preset-name"
        />
        <button onClick={savePreset} style={BUTTON_STYLE} data-testid="preset-save">
          Save
        </button>
        <select
          value=""
          onChange={(e) => loadPreset(e.target.value)}
          style={{ ...INPUT_STYLE, maxWidth: 150 }}
          data-testid="preset-load"
        >
          <option value="">Load preset…</option>
          {savedBoxes.map((b) => (
            <option key={b.id} value={b.id}>
              {b.name}
            </option>
          ))}
        </select>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '280px 1fr 340px',
          height: '100%',
          width: '100%',
          minHeight: 0,
        }}
      >
        {/* Left: Build panel */}
        <aside
          style={{
            borderRight: '1px solid #22262d',
            background: '#0f1218',
            overflowY: 'auto',
            minHeight: 0,
          }}
        >
          <BuildPanel validationErrors={validationErrors} />
        </aside>

        {/* Center: 3D scene + lid/layout bar */}
        <main
          style={{
            background: '#0b0d10',
            position: 'relative',
            minWidth: 0,
            minHeight: 0,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
            <BoxScene />
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '8px 12px',
              borderTop: '1px solid #22262d',
              background: '#0f1218',
              fontSize: 12,
            }}
          >
            <span style={{ opacity: 0.7 }}>Lid</span>
            <input
              data-testid="lid-slider"
              type="range"
              min={0}
              max={LID_MAX_DEG}
              step={1}
              value={lidTargetDeg}
              onChange={(e) => {
                const deg = parseFloat(e.target.value);
                log('lid_changed', { deg });
                setLidTargetDeg(deg);
              }}
              style={{ width: 180 }}
            />
            <span style={{ width: 36, opacity: 0.7 }}>{lidTargetDeg.toFixed(0)}°</span>
            <button
              data-testid="lid-toggle"
              onClick={() => {
                const next = lidTargetDeg > 0 ? 0 : 110;
                log('lid_changed', { deg: next, toggle: true });
                setLidTargetDeg(next);
              }}
              style={BUTTON_STYLE}
            >
              {lidTargetDeg > 0 ? 'Close' : 'Open'}
            </button>
            <button
              data-testid="autorotate-toggle"
              aria-pressed={autoRotate}
              title="Slowly spin the box on a turntable"
              onClick={() => {
                log('autorotate_toggled', { on: !autoRotate });
                setAutoRotate(!autoRotate);
              }}
              style={{
                ...BUTTON_STYLE,
                borderColor: autoRotate ? KIT.accent : KIT.border,
              }}
            >
              {autoRotate ? '◉ Auto-rotate' : '○ Auto-rotate'}
            </button>
            <button
              data-testid="inspect-toggle"
              aria-pressed={inspectMode}
              title="Lock head-on to the selected face; drag rocks the view ±8° with a live tilt / parallax readout"
              disabled={layout === 'flat'}
              onClick={() => {
                log('inspect_toggled', { on: !inspectMode });
                setInspectMode(!inspectMode);
              }}
              style={{
                ...BUTTON_STYLE,
                borderColor: inspectMode ? KIT.accent : KIT.border,
                opacity: layout === 'flat' ? 0.5 : 1,
              }}
            >
              {inspectMode ? '◉ Inspect' : '○ Inspect'}
            </button>
            <div style={{ flex: 1 }} />
            <div
              role="tablist"
              style={{
                display: 'flex',
                background: '#141820',
                border: '1px solid #2a2f36',
                borderRadius: 6,
                padding: 2,
              }}
            >
              {(['assembled', 'flat'] as const).map((l) => (
                <button
                  key={l}
                  data-testid={`layout-${l}`}
                  role="tab"
                  aria-selected={layout === l}
                  onClick={() => {
                    log('layout_changed', { layout: l });
                    setLayout(l);
                  }}
                  style={{
                    padding: '4px 12px',
                    background: layout === l ? '#1d2434' : 'transparent',
                    border: 'none',
                    color: '#e8eaed',
                    fontSize: 12,
                    cursor: 'pointer',
                    borderRadius: 4,
                    textTransform: 'capitalize',
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
        </main>

        {/* Right: Faces panel */}
        <aside
          style={{
            borderLeft: '1px solid #22262d',
            background: '#0f1218',
            overflowY: 'auto',
            minHeight: 0,
          }}
        >
          <FacesPanel />
        </aside>
      </div>
    </div>
  );
}
