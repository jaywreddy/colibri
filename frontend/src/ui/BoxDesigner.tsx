import { useEffect, useMemo, useRef, useState } from 'react';
import {
  FACE_IDS,
  generateBox,
  listBoxes,
  type BoxManifest,
  type FaceId,
} from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import BoxScene from '../scene/BoxScene';
import FaceEditor from './FaceEditor';
import IlluminationPanel from './IlluminationPanel';

const FACE_LABELS: Record<FaceId, string> = {
  front: 'Front',
  back: 'Back',
  top: 'Top',
  bottom: 'Bottom',
  left: 'Left',
  right: 'Right',
};

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
 * Box-mode root view: left rail of face thumbnails + box dimension controls,
 * center 3D BoxScene, right rail with FaceEditor + illumination controls.
 *
 * Triggers a debounced backend regenerate whenever the BoxSpec changes; the
 * resulting BoxManifest binds to the scene and to the thumbnails. Per-face
 * plates are cached by content hash on the backend, so only faces whose spec
 * actually changed re-render.
 */
export default function BoxDesigner() {
  const boxSpec = useStore((s) => s.boxSpec);
  const boxManifest = useStore((s) => s.boxManifest);
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const setSelectedFace = useStore((s) => s.setSelectedFace);
  const setBoxManifest = useStore((s) => s.setBoxManifest);
  const patchBoxSpec = useStore((s) => s.patchBoxSpec);
  const [flat, setFlat] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedBoxes, setSavedBoxes] = useState<BoxManifest[]>([]);
  const [boxLabel, setBoxLabel] = useState('');
  const lastReqIdRef = useRef(0);

  const regen = useDebounce(async () => {
    const reqId = ++lastReqIdRef.current;
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    try {
      const m = await generateBox(boxSpec);
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
      setError(err.message);
      log('box_regen_failed', { error: err.message });
    } finally {
      if (lastReqIdRef.current === reqId) setBusy(false);
    }
  }, 400);

  useEffect(() => {
    regen();
  }, [boxSpec, regen]);

  useEffect(() => {
    listBoxes().then(setSavedBoxes).catch(() => {});
  }, [boxManifest]);

  const saveCurrent = async () => {
    if (!boxLabel.trim()) {
      setError('Give the box a name to save it.');
      return;
    }
    try {
      const m = await generateBox(
        { ...boxSpec, label: boxLabel.trim() },
        { boxId: boxLabel.trim().toLowerCase().replace(/\s+/g, '-') }
      );
      setBoxManifest(m);
      setSavedBoxes(await listBoxes());
      log('box_saved', { id: m.id, name: m.name });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '220px 1fr 320px',
        gridTemplateRows: '1fr',
        height: '100%',
        width: '100%',
      }}
    >
      {/* Left rail: face picker + box dims + saved boxes */}
      <aside
        style={{
          borderRight: '1px solid #22262d',
          padding: 12,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7 }}>FACES</div>
        <div
          data-testid="face-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 6,
          }}
        >
          {FACE_IDS.map((fid) => {
            const fm = boxManifest?.faces[fid];
            const active = fid === selectedFaceId;
            return (
              <button
                key={fid}
                data-face={fid}
                data-testid={`face-thumb-${fid}`}
                onClick={() => setSelectedFace(fid)}
                style={{
                  padding: 4,
                  border: active ? '1px solid #8ab4ff' : '1px solid #2a2f36',
                  background: active ? '#1d2434' : '#141820',
                  borderRadius: 6,
                  color: '#e8eaed',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 2,
                  fontSize: 11,
                }}
              >
                <div
                  style={{
                    width: '100%',
                    aspectRatio: '1 / 1',
                    background: '#0b0d10',
                    borderRadius: 3,
                    backgroundImage: fm?.files.thumbnail
                      ? `url(${fm.files.thumbnail})`
                      : 'none',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                  }}
                />
                <div>{FACE_LABELS[fid]}</div>
              </button>
            );
          })}
        </div>

        <div style={{ borderTop: '1px solid #22262d', paddingTop: 12 }}>
          <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7, marginBottom: 8 }}>
            BOX DIMENSIONS (μm)
          </div>
          {(['width_um', 'height_um', 'depth_um'] as const).map((k) => (
            <label key={k} style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, marginBottom: 8 }}>
              <span>
                {k.replace('_um', '').toUpperCase()}
                <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>
                  {boxSpec[k].toFixed(0)}
                </span>
              </span>
              <input
                type="range"
                min={1500}
                max={12000}
                step={250}
                value={boxSpec[k]}
                onChange={(e) => patchBoxSpec({ [k]: parseFloat(e.target.value) })}
              />
            </label>
          ))}
        </div>

        <div style={{ borderTop: '1px solid #22262d', paddingTop: 12 }}>
          <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7, marginBottom: 8 }}>
            SAVED BOXES
          </div>
          <input
            placeholder="Box name to save"
            value={boxLabel}
            onChange={(e) => setBoxLabel(e.target.value)}
            style={{
              background: '#141820',
              border: '1px solid #2a2f36',
              color: '#e8eaed',
              padding: '4px 6px',
              borderRadius: 4,
              fontSize: 12,
              width: '100%',
              marginBottom: 6,
            }}
          />
          <button
            onClick={saveCurrent}
            style={{
              width: '100%',
              padding: '6px',
              background: '#1d2434',
              border: '1px solid #2a2f36',
              borderRadius: 4,
              color: '#e8eaed',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            Save current box
          </button>
          {savedBoxes.length > 0 && (
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {savedBoxes.map((b) => (
                <button
                  key={b.id}
                  onClick={() => {
                    useStore.getState().setBoxSpec({
                      width_um: b.dimensions_um.width,
                      height_um: b.dimensions_um.height,
                      depth_um: b.dimensions_um.depth,
                      faces: b.spec.faces,
                      label: b.name,
                    });
                    setBoxManifest(b);
                    log('box_loaded', { id: b.id });
                  }}
                  style={{
                    textAlign: 'left',
                    padding: '4px 6px',
                    background: '#141820',
                    border: '1px solid #2a2f36',
                    borderRadius: 4,
                    color: '#e8eaed',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  {b.name}
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* Center: 3D scene + flatten/export bar */}
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
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '6px 12px',
            borderBottom: '1px solid #22262d',
            background: '#0f1218',
            fontSize: 12,
          }}
        >
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={flat}
              onChange={(e) => setFlat(e.target.checked)}
              data-testid="flatten-toggle"
            />
            Flatten (2×3 grid)
          </label>
          <div style={{ flex: 1 }} />
          {busy && <span style={{ opacity: 0.7 }}>Regenerating…</span>}
          {error && (
            <span style={{ color: '#ff8888', maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {error}
            </span>
          )}
          {boxManifest && (
            <a
              href={`/export/box/${boxManifest.id}/fab.zip`}
              download
              style={{
                padding: '4px 10px',
                background: '#1d2434',
                border: '1px solid #2a2f36',
                borderRadius: 4,
                color: '#e8eaed',
                fontSize: 12,
                textDecoration: 'none',
              }}
            >
              Export fab bundle
            </a>
          )}
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <BoxScene manifest={boxManifest} flat={flat} onFaceClick={setSelectedFace} />
        </div>
      </main>

      {/* Right rail: face editor + illumination */}
      <aside
        style={{
          borderLeft: '1px solid #22262d',
          background: '#0f1218',
          overflowY: 'auto',
        }}
      >
        <FaceEditor />
        <div style={{ borderTop: '1px solid #22262d' }} />
        <IlluminationPanel />
      </aside>
    </div>
  );
}
