import { useStore } from '../store';

/**
 * Fourth-column panel shown beside the 3D canvas for recipes whose signature
 * physics can't live on the plate itself:
 *
 *   - `near_field_carpet` (tairona-talbot, muzo-emerald-zone) — shows the
 *     full vertical (x, z) propagation carpet. The z-slice slider in
 *     `IlluminationPanel` drives a horizontal indicator line so the user
 *     can see which slice the plate is currently sampling.
 *   - `far_field_hologram` (colibri-hologram, meridian-speckle) — will
 *     show the Fraunhofer RGB reconstruction in Phase E.
 *
 * For any other recipe the panel hides itself (returns null) so the canvas
 * reclaims that column.
 */
export default function SecondaryView() {
  const manifest = useStore((s) => s.manifest);
  const zSlice = useStore((s) => s.zSlice);
  const carpetAtlasUrl = useStore((s) => s.carpetAtlasUrl);
  const recipe = manifest?.render_recipe ?? null;
  const rd = (manifest?.recipe_data ?? {}) as Record<string, unknown>;

  if (recipe !== 'near_field_carpet' && recipe !== 'far_field_hologram') {
    return null;
  }

  // --- near_field_carpet: render the full carpet with a z-indicator line ---
  if (recipe === 'near_field_carpet') {
    const zMinUm = Number(rd.z_min_um ?? 0) || 0;
    const zMaxUm = Number(rd.z_max_um ?? 0) || 0;
    const zTalbotUm = Number(rd.talbot_distance_um ?? 0) || 0;
    const zFocalUm = Number(rd.focal_length_um ?? 0) || 0;
    // The carpet atlas is stacked top-to-bottom by z-slice; the indicator's
    // top offset as a percentage of the image height tracks zSlice.
    const indicatorTopPct = Math.max(0, Math.min(100, zSlice * 100));

    // Marker positions for z_T / f as a fraction of [z_min, z_max].
    const markerZT =
      zTalbotUm > 0 && zMaxUm > zMinUm
        ? ((zTalbotUm - zMinUm) / (zMaxUm - zMinUm)) * 100
        : null;
    const markerF =
      zFocalUm > 0 && zMaxUm > zMinUm
        ? ((zFocalUm - zMinUm) / (zMaxUm - zMinUm)) * 100
        : null;

    return (
      <aside
        data-testid="secondary-view"
        data-recipe={recipe}
        style={panel}
      >
        <div style={title}>Near-field carpet — propagation through z</div>
        <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
          {carpetAtlasUrl ? (
            <>
              <img
                data-testid="carpet-image"
                src={carpetAtlasUrl}
                alt="near-field carpet"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'fill',
                  // Nearest-neighbor keeps the per-slice rows crisp.
                  imageRendering: 'pixelated',
                }}
              />
              {/* Z-slice indicator */}
              <div
                data-testid="z-indicator"
                style={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  top: `${indicatorTopPct}%`,
                  height: 2,
                  background: '#8ab4ff',
                  boxShadow: '0 0 4px #8ab4ff',
                  pointerEvents: 'none',
                }}
              />
              {markerZT != null && (
                <div style={{ ...marker, top: `${markerZT}%` }} title="z_T">
                  <span style={markerLabel}>z_T</span>
                </div>
              )}
              {markerF != null && (
                <div style={{ ...marker, top: `${markerF}%` }} title="focal">
                  <span style={markerLabel}>f</span>
                </div>
              )}
            </>
          ) : (
            <div style={placeholder}>fetching carpet…</div>
          )}
        </div>
        <div style={footer}>
          z<sub>min</sub> {zMinUm.toFixed(0)} μm · z<sub>max</sub>{' '}
          {zMaxUm.toFixed(0)} μm
        </div>
      </aside>
    );
  }

  // --- far_field_hologram placeholder (Phase E will populate) --------------
  return (
    <aside
      data-testid="secondary-view"
      data-recipe={recipe}
      style={panel}
    >
      <div style={title}>Far-field reconstruction</div>
      <div style={placeholder}>coming in Phase E</div>
    </aside>
  );
}

const panel: React.CSSProperties = {
  gridArea: 'secondary',
  borderLeft: '1px solid #22262d',
  background: '#0b0d10',
  display: 'flex',
  flexDirection: 'column',
  padding: 12,
  gap: 8,
  minHeight: 0,
};
const title: React.CSSProperties = {
  fontSize: 11,
  letterSpacing: 1.2,
  opacity: 0.75,
  textTransform: 'uppercase',
};
const footer: React.CSSProperties = {
  fontSize: 11,
  opacity: 0.6,
};
const placeholder: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  opacity: 0.45,
  fontSize: 12,
};
const marker: React.CSSProperties = {
  position: 'absolute',
  left: 0,
  right: 0,
  height: 1,
  borderTop: '1px dashed #6b7280',
  pointerEvents: 'none',
};
const markerLabel: React.CSSProperties = {
  position: 'absolute',
  right: 2,
  top: -14,
  fontSize: 10,
  color: '#9ca3af',
  background: '#0b0d10',
  padding: '0 3px',
};
