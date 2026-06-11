import { useEffect, useState } from 'react';
import { useStore } from '../store';
import { log } from '../logger';
import FrameControls from './FrameControls';
import ParameterPanel from './ParameterPanel';
import { FACE_LABELS } from './FacesPanel';
import { Button, Disclosure, KIT, Shimmer, SubHeader } from './kit';

/**
 * Editor for the selected face: visual pattern picker (thumbnail cards) +
 * apply-to-all / shuffle-seed actions + the central pattern's params
 * (ParameterPanel) + frame dials (FrameControls). Parameter changes write to
 * the store; App debounces the box regeneration.
 *
 * Pattern thumbnails are fetched lazily whenever the picker opens (GET
 * /patterns/{slug}/default materializes server-side in ~2 s, then caches).
 * The store skips slugs that already loaded or are in flight, so repeat
 * calls are free — calling on EVERY open (plus when the catalog arrives
 * while the picker is open) is what retries slugs that failed earlier.
 */
export default function FaceEditor() {
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const face = useStore((s) => s.boxSpec.faces[selectedFaceId]);
  const catalog = useStore((s) => s.catalog);
  const thumbnails = useStore((s) => s.thumbnails);
  const loadThumbnails = useStore((s) => s.loadThumbnails);
  const patchFace = useStore((s) => s.patchFace);
  const applyFaceToAll = useStore((s) => s.applyFaceToAll);
  const shuffleFaceSeed = useStore((s) => s.shuffleFaceSeed);

  // True once the picker has been opened at least once this session. Covers
  // the boot race: picker opened before GET /patterns populated the catalog
  // (loadThumbnails no-ops on an empty catalog), so refetch when it lands.
  const [pickerOpened, setPickerOpened] = useState(false);
  useEffect(() => {
    if (pickerOpened && catalog.length > 0) void loadThumbnails();
  }, [pickerOpened, catalog.length, loadThumbnails]);

  if (!face) {
    return <div style={{ padding: 12, opacity: 0.6 }}>No face selected.</div>;
  }

  const descriptor = catalog.find((c) => c.slug === face.pattern_slug);

  return (
    <div
      data-testid="face-editor"
      style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}
    >
      <SubHeader>EDITING: {FACE_LABELS[selectedFaceId].toUpperCase()}</SubHeader>

      <Disclosure
        label={`Pattern — ${descriptor?.name ?? face.pattern_slug}`}
        testId="pattern-picker"
        onOpen={() => {
          setPickerOpened(true);
          void loadThumbnails();
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 6,
          }}
        >
          {catalog.map((c) => {
            const active = c.slug === face.pattern_slug;
            const thumb = thumbnails[c.slug];
            return (
              <button
                key={c.slug}
                data-testid={`pattern-card-${c.slug}`}
                title={c.description}
                aria-pressed={active}
                onClick={() => {
                  log('face_pattern_changed', { faceId: selectedFaceId, slug: c.slug });
                  // Reset central pattern params on pattern change; the
                  // backend re-merges class defaults during materialize.
                  patchFace(selectedFaceId, { pattern_slug: c.slug, pattern_params: {} });
                }}
                style={{
                  padding: 4,
                  border: `1px solid ${active ? KIT.accent : KIT.border}`,
                  background: active ? KIT.raised : KIT.field,
                  borderRadius: 6,
                  color: KIT.text,
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                  fontSize: 11,
                  textAlign: 'center',
                }}
              >
                {typeof thumb === 'string' ? (
                  <img
                    src={thumb}
                    alt={c.name}
                    style={{
                      width: '100%',
                      aspectRatio: '1 / 1',
                      objectFit: 'cover',
                      borderRadius: 4,
                      background: '#0b0d10',
                    }}
                  />
                ) : (
                  <Shimmer style={{ width: '100%', aspectRatio: '1 / 1' }} />
                )}
                <div style={{ lineHeight: 1.25 }}>{c.name}</div>
              </button>
            );
          })}
        </div>
      </Disclosure>

      {descriptor && (
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.4 }}>
          {descriptor.description}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6 }}>
        <Button
          testId="apply-all-faces"
          title="Copy this face's pattern + dials to all 6 faces (each keeps its own seed)"
          onClick={() => {
            log('face_apply_all', { from: selectedFaceId });
            applyFaceToAll(selectedFaceId);
          }}
          style={{ flex: 1 }}
        >
          Apply to all faces
        </Button>
        <Button
          testId="shuffle-seed"
          title="Randomize this face's frame seed"
          onClick={() => {
            shuffleFaceSeed(selectedFaceId);
            log('face_seed_shuffled', {
              faceId: selectedFaceId,
              seed: useStore.getState().boxSpec.faces[selectedFaceId]?.frame.seed,
            });
          }}
          style={{ flex: 1 }}
        >
          🎲 Shuffle seed
        </Button>
      </div>

      <ParameterPanel faceId={selectedFaceId} />

      <FrameControls faceId={selectedFaceId} />
    </div>
  );
}
