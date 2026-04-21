import { useStore } from '../store';

/**
 * Fourth-column panel shown beside the 3D canvas for recipes whose signature
 * physics can't live on the plate itself:
 *
 *   - `near_field_carpet` (tairona-talbot, muzo-emerald-zone) — shows the
 *     full vertical (x, z) propagation carpet. The z-slice slider in
 *     `IlluminationPanel` drives a horizontal indicator line so the user
 *     can see which slice the plate is currently sampling.
 *   - `far_field_hologram` (colibri-hologram, meridian-speckle) — shows
 *     the merged-RGB Fraunhofer reconstruction returned by /sim/farfield.
 *     The plate itself shows the bare mask (recipe 4 routes to runStylized)
 *     because the actual "what you'd see on a screen" lives here, not
 *     in the shader.
 *
 * For any other recipe the panel hides itself (returns null) so the canvas
 * reclaims that column.
 */
export default function SecondaryView() {
  const manifest = useStore((s) => s.manifest);
  const zSlice = useStore((s) => s.zSlice);
  const carpetAtlasUrl = useStore((s) => s.carpetAtlasUrl);
  const farfieldUrl = useStore((s) => s.farfieldUrl);
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
    const layout = rd.carpet_layout === 'stripe' ? 'stripe' : 'tiles';
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

    // Stripe layout = canonical (x, z) Talbot diagram (tairona); tiles layout
    // = stack of 2D focal snapshots per z (muzo). The title and the footer
    // axis hint differ so the user reads the panel correctly.
    const carpetTitle =
      layout === 'stripe'
        ? 'Talbot carpet — x (horizontal) · z (vertical)'
        : 'Near-field carpet — propagation through z';

    return (
      <aside
        data-testid="secondary-view"
        data-recipe={recipe}
        data-carpet-layout={layout}
        style={panel}
      >
        <div style={title}>{carpetTitle}</div>
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

  // --- far_field_hologram: merged-RGB Fraunhofer reconstruction -----------
  // colibri-hologram: a hummingbird silhouette under white coherent light.
  // meridian-speckle: a flat-topped beam with color-smeared speckle grain.
  // The plate shader itself doesn't know about any of this — the plate
  // keeps showing the bare amplitude mask, and the "signature" physics
  // visualization is the DOM image right here. A styled "plate → projected"
  // arrow above the reconstruction emphasizes the optical relationship.
  const wavelengths = Array.isArray(rd.wavelengths_um) ? rd.wavelengths_um : [];
  const target = typeof rd.target === 'string' ? rd.target : null;
  return (
    <aside
      data-testid="secondary-view"
      data-recipe={recipe}
      style={panel}
    >
      <div style={title}>
        Far-field reconstruction
        <span style={{ float: 'right', opacity: 0.5, textTransform: 'none' }}>
          plate&nbsp;──→&nbsp;projected
        </span>
      </div>
      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        {farfieldUrl ? (
          <img
            data-testid="farfield-image"
            src={farfieldUrl}
            alt={`far-field reconstruction${target ? ` (${target})` : ''}`}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              // Log-stretched reconstructions look best with crisp pixels
              // at small atlas sizes; the actual PNG is ~128×128.
              imageRendering: 'pixelated',
              background: '#000',
            }}
          />
        ) : (
          <div style={placeholder}>fetching reconstruction…</div>
        )}
      </div>
      <div style={footer}>
        {wavelengths.length === 3
          ? `λ = ${wavelengths
              .map((w) => `${(Number(w) * 1000).toFixed(0)} nm`)
              .join(' · ')}`
          : 'coherent illumination'}
        {target && (
          <span style={{ float: 'right', opacity: 0.7 }}>target: {target}</span>
        )}
      </div>
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
