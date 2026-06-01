import { useEffect } from 'react';
import { listPatterns } from './api';
import { log } from './logger';
import Gallery from './ui/Gallery';
import ParameterPanel from './ui/ParameterPanel';
import IlluminationPanel from './ui/IlluminationPanel';
import ManifestPanel from './ui/ManifestPanel';
import PlateScene from './scene/PlateScene';
import BoxDesigner from './ui/BoxDesigner';
import { useStore } from './store';

export default function App() {
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const setCatalog = useStore((s) => s.setCatalog);
  const catalog = useStore((s) => s.catalog);

  // Catalog is needed in BOTH modes (Gallery for plate mode, FaceEditor for
  // box mode). Fetch once at app boot, regardless of which mode is active.
  useEffect(() => {
    if (catalog.length > 0) return;
    listPatterns()
      .then((pts) => {
        setCatalog(pts);
        log('catalog_loaded', { count: pts.length });
      })
      .catch((e) => log('catalog_load_failed', { error: (e as Error).message }));
  }, [catalog.length, setCatalog]);

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
        <div style={{ fontWeight: 700, letterSpacing: 0.3 }}>Optics Pattern Studio</div>
        <div style={{ fontSize: 11, opacity: 0.6 }}>
          Dual-layer gold-on-quartz · 500 μm substrate · 2 μm min feature
        </div>
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
          {(['box', 'plate'] as const).map((m) => (
            <button
              key={m}
              data-testid={`mode-${m}`}
              role="tab"
              aria-selected={mode === m}
              onClick={() => {
                log('mode_changed', { mode: m });
                setMode(m);
              }}
              style={{
                padding: '4px 12px',
                background: mode === m ? '#1d2434' : 'transparent',
                border: 'none',
                color: '#e8eaed',
                fontSize: 12,
                cursor: 'pointer',
                borderRadius: 4,
                textTransform: 'capitalize',
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </header>

      {mode === 'box' ? <BoxDesigner /> : <PlateLayout />}
    </div>
  );
}

function PlateLayout() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '260px 1fr 300px',
        gridTemplateRows: '1fr auto',
        gridTemplateAreas: `
          "gallery canvas controls"
          "gallery manifest controls"
        `,
        height: '100%',
        width: '100%',
      }}
    >
      <aside
        style={{
          gridArea: 'gallery',
          borderRight: '1px solid #22262d',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Gallery />
      </aside>
      <main
        style={{
          gridArea: 'canvas',
          background: '#0b0d10',
          position: 'relative',
          minWidth: 0,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        <PlateScene />
      </main>
      <section style={{ gridArea: 'manifest', background: '#0f1218' }}>
        <ManifestPanel />
      </section>
      <aside
        style={{
          gridArea: 'controls',
          borderLeft: '1px solid #22262d',
          overflowY: 'auto',
          background: '#0f1218',
        }}
      >
        <IlluminationPanel />
        <div style={{ borderTop: '1px solid #22262d' }} />
        <ParameterPanel />
      </aside>
    </div>
  );
}
