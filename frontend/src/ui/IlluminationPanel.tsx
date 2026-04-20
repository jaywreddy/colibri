import { log } from '../logger';
import { useStore } from '../store';

export default function IlluminationPanel() {
  const illumination = useStore((s) => s.illumination);
  const setIllumination = useStore((s) => s.setIllumination);
  const laserColor = useStore((s) => s.laserColor);
  const setLaserColor = useStore((s) => s.setLaserColor);
  const az = useStore((s) => s.lightAzimuthDeg);
  const el = useStore((s) => s.lightElevationDeg);
  const setLight = useStore((s) => s.setLight);
  const manifest = useStore((s) => s.manifest);
  const zSlice = useStore((s) => s.zSlice);
  const setZSlice = useStore((s) => s.setZSlice);

  const recipe = manifest?.render_recipe ?? null;
  const rd = (manifest?.recipe_data ?? {}) as Record<string, unknown>;
  const isCarpet = recipe === 'near_field_carpet';
  // For the slider readout — e.g. "z = 1200 μm" with a hint for z_T or f.
  const zMinUm = Number(rd.z_min_um ?? 0) || 0;
  const zMaxUm = Number(rd.z_max_um ?? 0) || 0;
  const zAbsUm = zMinUm + zSlice * (zMaxUm - zMinUm);
  const zTalbotUm = Number(rd.talbot_distance_um ?? 0) || 0;
  const zFocalUm = Number(rd.focal_length_um ?? 0) || 0;

  return (
    <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={hdr}>ILLUMINATION</div>
        <div style={{ display: 'flex', gap: 6 }}>
          {(['ambient', 'laser', 'backlight'] as const).map((m) => (
            <button
              key={m}
              data-mode={m}
              onClick={() => {
                log('illumination_changed', { from: illumination, to: m });
                setIllumination(m);
              }}
              style={{
                ...btn,
                background: illumination === m ? '#1d2434' : '#141820',
                borderColor: illumination === m ? '#8ab4ff' : '#2a2f36',
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {isCarpet && (
        <div>
          <div style={hdr}>Z-SLICE (NEAR-FIELD CARPET)</div>
          <label style={lbl}>
            <span>
              z = {zAbsUm.toFixed(0)} μm
              <span style={{ float: 'right', opacity: 0.7 }}>
                {zTalbotUm
                  ? `z_T = ${zTalbotUm.toFixed(0)} μm`
                  : zFocalUm
                  ? `f = ${zFocalUm.toFixed(0)} μm`
                  : ''}
              </span>
            </span>
            <input
              data-testid="z-slice"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={zSlice}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                log('z_slice_changed', { from: zSlice, to: v });
                setZSlice(v);
              }}
            />
          </label>
        </div>
      )}

      {illumination === 'laser' && (
        <div>
          <div style={hdr}>LASER</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {(['red', 'green', 'blue'] as const).map((c) => (
              <button
                key={c}
                data-color={c}
                onClick={() => {
                  log('laser_color_changed', { from: laserColor, to: c });
                  setLaserColor(c);
                }}
                style={{
                  ...btn,
                  background: laserColor === c ? '#1d2434' : '#141820',
                  borderColor: laserColor === c ? '#8ab4ff' : '#2a2f36',
                }}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      )}

      <div>
        <div style={hdr}>LIGHT DIRECTION</div>
        <label style={lbl}>
          Azimuth <span style={{ float: 'right', opacity: 0.7 }}>{az.toFixed(0)}°</span>
          <input
            data-testid="light-az"
            type="range"
            min={-180}
            max={180}
            step={1}
            value={az}
            onChange={(e) => {
              const nextAz = parseFloat(e.target.value);
              log('light_moved', { az: nextAz, el });
              setLight(nextAz, el);
            }}
          />
        </label>
        <label style={lbl}>
          Elevation <span style={{ float: 'right', opacity: 0.7 }}>{el.toFixed(0)}°</span>
          <input
            data-testid="light-el"
            type="range"
            min={5}
            max={89}
            step={1}
            value={el}
            onChange={(e) => {
              const nextEl = parseFloat(e.target.value);
              log('light_moved', { az, el: nextEl });
              setLight(az, nextEl);
            }}
          />
        </label>
      </div>
    </div>
  );
}

const hdr: React.CSSProperties = {
  fontSize: 11,
  letterSpacing: 1.5,
  opacity: 0.7,
  marginBottom: 6,
};
const lbl: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  fontSize: 12,
  marginBottom: 6,
};
const btn: React.CSSProperties = {
  flex: 1,
  border: '1px solid #2a2f36',
  borderRadius: 4,
  padding: '6px 8px',
  color: '#e8eaed',
  cursor: 'pointer',
  textTransform: 'capitalize',
  fontSize: 12,
};
