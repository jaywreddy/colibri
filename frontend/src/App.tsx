import Gallery from './ui/Gallery';
import ParameterPanel from './ui/ParameterPanel';
import IlluminationPanel from './ui/IlluminationPanel';
import ManifestPanel from './ui/ManifestPanel';
import PlateScene from './scene/PlateScene';

export default function App() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '260px 1fr 300px',
        gridTemplateRows: '48px 1fr auto',
        gridTemplateAreas: `
          "header header header"
          "gallery canvas controls"
          "gallery manifest controls"
        `,
        height: '100vh',
        width: '100vw',
      }}
    >
      <header
        style={{
          gridArea: 'header',
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
      </header>

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
