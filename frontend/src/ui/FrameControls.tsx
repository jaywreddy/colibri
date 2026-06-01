import { useStore } from '../store';
import { log } from '../logger';
import type { FaceId, FrameSpec } from '../api';

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

/**
 * Frame dials for one face — algorithm picker, theme picker, density/bloom/
 * foliage sliders, and the seed control. All edits go to `patchFaceFrame`;
 * the parent (FaceEditor) debounces regeneration of the box manifest.
 */
export default function FrameControls({ faceId }: { faceId: FaceId }) {
  const face = useStore((s) => s.boxSpec.faces[faceId]);
  const patchFaceFrame = useStore((s) => s.patchFaceFrame);

  if (!face) {
    return <div style={{ opacity: 0.6, padding: 12 }}>No frame on this face.</div>;
  }
  const f = face.frame;

  const setNum = <K extends keyof FrameSpec>(key: K, value: number) => {
    log('frame_param_changed', { faceId, key, value });
    patchFaceFrame(faceId, { [key]: value } as Partial<FrameSpec>);
  };

  return (
    <div data-testid="frame-controls" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 11, letterSpacing: 1.5, opacity: 0.7 }}>FRAME</div>

      <label style={ROW}>
        <span>Algorithm</span>
        <select
          value={f.algorithm}
          onChange={(e) =>
            patchFaceFrame(faceId, { algorithm: e.target.value as FrameSpec['algorithm'] })
          }
          style={INPUT}
        >
          <option value="colonize">Space colonization</option>
        </select>
      </label>

      <label style={ROW}>
        <span>Theme</span>
        <select
          value={f.theme}
          onChange={(e) =>
            patchFaceFrame(faceId, { theme: e.target.value as FrameSpec['theme'] })
          }
          style={INPUT}
        >
          <option value="esmeralda">Esmeralda (Colombian)</option>
        </select>
      </label>

      <label style={ROW}>
        <span>
          Density
          <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>{f.density.toFixed(2)}</span>
        </span>
        <input
          type="range"
          min={0.4}
          max={1.8}
          step={0.05}
          value={f.density}
          onChange={(e) => setNum('density', parseFloat(e.target.value))}
        />
      </label>

      <label style={ROW}>
        <span>
          Bloom
          <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>{f.bloom.toFixed(2)}</span>
        </span>
        <input
          type="range"
          min={0}
          max={1.6}
          step={0.05}
          value={f.bloom}
          onChange={(e) => setNum('bloom', parseFloat(e.target.value))}
        />
      </label>

      <label style={ROW}>
        <span>
          Foliage
          <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>{f.foliage.toFixed(2)}</span>
        </span>
        <input
          type="range"
          min={0}
          max={1.4}
          step={0.05}
          value={f.foliage}
          onChange={(e) => setNum('foliage', parseFloat(e.target.value))}
        />
      </label>

      <label style={ROW}>
        <span>
          Frame band (μm){' '}
          <span style={{ float: 'right', opacity: 0.7, fontSize: 10 }}>
            {f.band_um == null ? 'auto' : f.band_um.toFixed(0)}
          </span>
        </span>
        <input
          type="range"
          min={500}
          max={6000}
          step={100}
          value={f.band_um ?? Math.round(0.12 * Math.min(face.width_um, face.height_um))}
          onChange={(e) => setNum('band_um', parseFloat(e.target.value))}
        />
      </label>

      <label style={ROW}>
        <span>Seed</span>
        <div style={{ display: 'flex', gap: 6 }}>
          <input
            type="number"
            value={f.seed}
            onChange={(e) => setNum('seed', parseInt(e.target.value, 10) || 0)}
            style={{ ...INPUT, flex: 1 }}
          />
          <button
            onClick={() => setNum('seed', Math.floor(Math.random() * 0x7fffffff))}
            style={{ ...INPUT, cursor: 'pointer', padding: '4px 10px' }}
            title="Random seed"
          >
            🎲
          </button>
        </div>
      </label>
    </div>
  );
}
