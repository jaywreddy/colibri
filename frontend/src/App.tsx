import { useEffect, useMemo, useRef, useState } from 'react';
import {
  generateBox,
  getBox,
  listBoxes,
  listPatterns,
  type BoxManifest,
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
 * Ring Box Studio — single-purpose, box-first studio screen.
 *
 * Any persisted spec change (dims, glass, foil, hinge, faces) triggers a
 * debounced POST /boxes/generate with a stale-response guard, keeping the
 * saved manifest — and the "Export fab bundle" zip built from it — in sync
 * with the on-screen design. Hinge/bead/finish edits still update the scene
 * instantly via src/assembly.ts; their regen only refreshes the manifest
 * (per-face plate caches hit, no mask recompute). Lid angle and layout are
 * view-only and never hit the backend.
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

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedBoxes, setSavedBoxes] = useState<BoxManifest[]>([]);
  const [presetName, setPresetName] = useState('');
  const lastReqIdRef = useRef(0);

  // Pattern catalog — fetched once at boot for the face editors.
  useEffect(() => {
    if (catalog.length > 0) return;
    listPatterns()
      .then((pts) => {
        setCatalog(pts);
        log('catalog_loaded', { count: pts.length });
      })
      .catch((e) => log('catalog_load_failed', { error: (e as Error).message }));
  }, [catalog.length, setCatalog]);

  const validationErrors = useMemo(() => validateBox(boxSpec), [boxSpec]);

  // Every persisted spec field participates in the regen key. Hinge, bead
  // and finish don't change masks, but they DO change the saved manifest —
  // ASSEMBLY.md's hinge cut list and finish come from it — so excluding them
  // would let "Export fab bundle" ship a bundle that disagrees with the UI.
  // Hinge/foil-only regens are cheap: the backend per-face plate caches hit
  // and only the manifest assembly block is recomputed. Lid angle and layout
  // are view-only state and stay out.
  const regenKey = useMemo(
    () =>
      JSON.stringify({
        w: boxSpec.width_um,
        d: boxSpec.depth_um,
        h: boxSpec.height_um,
        glass: boxSpec.glass,
        foil: boxSpec.foil,
        hinge: boxSpec.hinge,
        faces: boxSpec.faces,
      }),
    [boxSpec]
  );

  const regen = useDebounce(async () => {
    const spec = useStore.getState().boxSpec;
    const errors = validateBox(spec);
    if (errors.length > 0) {
      setError(errors[0]);
      log('box_regen_skipped_invalid', { errors });
      return;
    }
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
      setBoxManifest(m);
      log('box_regen_done', {
        id: m.id,
        duration_ms: Math.round(performance.now() - t0),
      });
    } catch (e) {
      const err = e as Error;
      if (lastReqIdRef.current === reqId) setError(err.message);
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
      const m = await generateBox(
        { ...useStore.getState().boxSpec, label: name },
        { boxId: slug }
      );
      setBoxManifest(m);
      setSavedBoxes(await listBoxes());
      log('box_saved', { id: m.id, name: m.name });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadPreset = async (id: string) => {
    if (!id) return;
    try {
      const m = await getBox(id);
      setBoxSpec(m.spec);
      setBoxManifest(m);
      setPresetName(m.name);
      log('box_loaded', { id: m.id });
    } catch (e) {
      setError((e as Error).message);
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
              color: '#ff8888',
              fontSize: 12,
              maxWidth: 320,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {error}
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
        {boxManifest && (
          <a
            href={`/export/box/${boxManifest.id}/fab.zip`}
            download
            data-testid="export-fab"
            style={{ ...BUTTON_STYLE, textDecoration: 'none' }}
          >
            Export fab bundle
          </a>
        )}
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
          <div style={{ flex: 1, minHeight: 0 }}>
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
