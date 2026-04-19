import { useStore } from '../store';

export default function ManifestPanel() {
  const m = useStore((s) => s.manifest);
  if (!m) return null;
  return (
    <div
      style={{
        padding: '10px 12px',
        borderTop: '1px solid #22262d',
        fontSize: 11,
        opacity: 0.85,
        lineHeight: 1.5,
      }}
    >
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <span>
          <strong>Extent:</strong> {m.extent_um[0].toFixed(0)} × {m.extent_um[1].toFixed(0)} μm
        </span>
        <span>
          <strong>Pitch:</strong> {m.pixel_pitch_um.toFixed(2)} μm/px
        </span>
        <span>
          <strong>Min feat:</strong> {m.min_feature_um.toFixed(1)} μm
        </span>
        <span>
          <strong>Substrate:</strong> {m.substrate.thickness_um} μm {m.substrate.material} (n={m.substrate.n})
        </span>
      </div>
      {Object.keys(m.extra).length > 0 && (
        <div style={{ marginTop: 4, opacity: 0.75 }}>
          {Object.entries(m.extra).map(([k, v]) => (
            <span key={k} style={{ marginRight: 10 }}>
              <em>{k}:</em>{' '}
              {typeof v === 'number' ? (v as number).toFixed(3) : String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
