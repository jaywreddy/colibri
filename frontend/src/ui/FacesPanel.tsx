import { FACE_IDS, type FaceId } from '../api';
import { useStore } from '../store';
import { KIT, Section } from './kit';
import FaceEditor from './FaceEditor';
import IlluminationPanel from './IlluminationPanel';

/**
 * Human-readable face names — the single source for every label a user reads
 * (face grid, "EDITING:" banner, cut list). `top` is the box's lid, so it reads
 * "Lid" everywhere; the raw `top` id stays in the spec, manifests and fab
 * artifacts (CUTLIST.csv, per-face files).
 */
export const FACE_LABELS: Record<FaceId, string> = {
  front: 'Front',
  back: 'Back',
  top: 'Lid',
  bottom: 'Bottom',
  left: 'Left',
  right: 'Right',
};

/**
 * Right "Faces" rail: 6 face thumbnails (from the box manifest), the editor
 * for the selected face (pattern picker + params + frame), and illumination.
 */
export default function FacesPanel() {
  const boxManifest = useStore((s) => s.boxManifest);
  const metal = useStore((s) => s.boxSpec.metal ?? 'gold');
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const setSelectedFace = useStore((s) => s.setSelectedFace);

  return (
    <div data-testid="faces-panel">
      <Section title="Faces" testId="section-faces" persistId="faces">
        <div
          data-testid="face-grid"
          style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}
        >
          {FACE_IDS.map((fid) => {
            const fm = boxManifest?.faces[fid];
            const thumbUrl = fm?.files.thumbnails?.[metal] ?? fm?.files.thumbnail;
            const active = fid === selectedFaceId;
            return (
              <button
                key={fid}
                data-face={fid}
                data-testid={`face-thumb-${fid}`}
                onClick={() => setSelectedFace(fid)}
                style={{
                  padding: 4,
                  border: `1px solid ${active ? KIT.accent : KIT.border}`,
                  background: active ? KIT.raised : KIT.field,
                  borderRadius: 6,
                  color: KIT.text,
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
                    // The backend renders one chip PER LITHO METAL (the masks
                    // are metal-agnostic graylevel codes, so the palette is the
                    // chip's only colour choice). Picking the matching chip
                    // keeps the rail consistent with the 3D preview instead of
                    // showing gold beside a platinum-lined render; older
                    // manifests without the map fall back to the gold chip.
                    backgroundImage: thumbUrl ? `url(${thumbUrl})` : 'none',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                  }}
                />
                <div>{FACE_LABELS[fid]}</div>
              </button>
            );
          })}
        </div>
      </Section>
      <FaceEditor />
      <IlluminationPanel />
    </div>
  );
}
