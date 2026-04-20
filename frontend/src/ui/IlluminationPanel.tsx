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
  const engine = useStore((s) => s.engine);
  const setEngine = useStore((s) => s.setEngine);

  return (
    <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={hdr}>ENGINE</div>
        <select
          data-testid="engine-select"
          value={engine}
          onChange={(e) => {
            const next = e.target.value as typeof engine;
            log('engine_switched', { from: engine, to: next });
            setEngine(next);
          }}
          style={input}
        >
          <option value="stylized">Tier 1 — Stylized parallax</option>
          <option value="fraunhofer">Tier 2 — Fraunhofer FFT</option>
          <option value="waveprop">Tier 3 — Angular spectrum (scaffolded)</option>
        </select>
      </div>

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
const input: React.CSSProperties = {
  width: '100%',
  background: '#141820',
  border: '1px solid #2a2f36',
  color: '#e8eaed',
  borderRadius: 4,
  padding: '6px 8px',
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
