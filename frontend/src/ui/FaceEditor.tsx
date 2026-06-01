import { useStore } from '../store';
import { log } from '../logger';
import type { FaceId } from '../api';
import FrameControls from './FrameControls';

const ROW: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  fontSize: 12,
};

const INPUT: React.CSSProperties = {
  background: '#141820',
  border: '1px solid #2a2f36',
  color: '#e8eaed',
  borderRadius: 4,
  padding: '4px 6px',
  fontSize: 12,
};

const FACE_LABELS: Record<FaceId, string> = {
  front: 'Front',
  back: 'Back',
  top: 'Top',
  bottom: 'Bottom',
  left: 'Left',
  right: 'Right',
};

/**
 * Right-rail editor for one face: pattern picker + the central pattern's
 * params + frame dials. Parameter changes write to the store; the parent
 * BoxDesigner debounces regeneration.
 */
export default function FaceEditor() {
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const face = useStore((s) => s.boxSpec.faces[selectedFaceId]);
  const catalog = useStore((s) => s.catalog);
  const patchFace = useStore((s) => s.patchFace);

  if (!face) {
    return <div style={{ padding: 12, opacity: 0.6 }}>No face selected.</div>;
  }

  const descriptor = catalog.find((c) => c.slug === face.pattern_slug);

  return (
    <div
      data-testid="face-editor"
      style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}
    >
      <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7 }}>
        EDITING: {FACE_LABELS[selectedFaceId].toUpperCase()}
      </div>

      <label style={ROW}>
        <span>Central pattern</span>
        <select
          value={face.pattern_slug}
          onChange={(e) => {
            const slug = e.target.value;
            log('face_pattern_changed', { faceId: selectedFaceId, slug });
            // Reset central pattern params on pattern change to defaults; the
            // backend will re-merge with class defaults during materialize.
            patchFace(selectedFaceId, { pattern_slug: slug, pattern_params: {} });
          }}
          style={INPUT}
        >
          {catalog.map((c) => (
            <option key={c.slug} value={c.slug}>
              {c.name}
            </option>
          ))}
        </select>
      </label>

      {descriptor && (
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.4 }}>{descriptor.description}</div>
      )}

      {/* Central pattern params */}
      {descriptor && descriptor.params.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7 }}>PATTERN</div>
          {descriptor.params.map((p) => {
            const v =
              (face.pattern_params[p.name] as number | string | boolean | undefined) ??
              (p.default as number | string | boolean);
            if (p.type === 'choice') {
              return (
                <label key={p.name} style={ROW}>
                  <span>
                    {p.label}
                    {p.unit ? <em style={{ opacity: 0.5 }}> ({p.unit})</em> : null}
                  </span>
                  <select
                    value={String(v)}
                    onChange={(e) =>
                      patchFace(selectedFaceId, {
                        pattern_params: {
                          ...face.pattern_params,
                          [p.name]: e.target.value,
                        },
                      })
                    }
                    style={INPUT}
                  >
                    {p.choices?.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              );
            }
            if (p.type === 'bool') {
              return (
                <label key={p.name} style={ROW}>
                  <span>{p.label}</span>
                  <input
                    type="checkbox"
                    checked={Boolean(v)}
                    onChange={(e) =>
                      patchFace(selectedFaceId, {
                        pattern_params: {
                          ...face.pattern_params,
                          [p.name]: e.target.checked,
                        },
                      })
                    }
                  />
                </label>
              );
            }
            return (
              <label key={p.name} style={ROW}>
                <span>
                  {p.label}
                  {p.unit ? <em style={{ opacity: 0.5 }}> ({p.unit})</em> : null}
                  <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>
                    {Number(v).toFixed(p.type === 'int' ? 0 : 2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step ?? (p.type === 'int' ? 1 : 0.01)}
                  value={Number(v)}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const value = p.type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
                    patchFace(selectedFaceId, {
                      pattern_params: { ...face.pattern_params, [p.name]: value },
                    });
                  }}
                />
              </label>
            );
          })}
        </div>
      )}

      <FrameControls faceId={selectedFaceId} />
    </div>
  );
}
