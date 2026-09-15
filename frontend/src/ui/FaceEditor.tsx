import { useEffect, useState } from 'react';
import { PICKER_SLUGS } from '../api';
import { useStore } from '../store';
import { log } from '../logger';
import PhotoChoice from './PhotoChoice';
import { FACE_LABELS } from './FacesPanel';
import { Button, FailedTile, KIT, Section, Shimmer } from './kit';

/**
 * Editor for the selected face: the pattern picker restricted to the box's own
 * five constructions, plus the photo choice on a photo wall.
 *
 * What is NOT here any more, and why: the frame dials (density, bloom,
 * foliage, band, seed, motif scale), "Apply to all faces", the seed shuffle
 * and the per-face single-ply toggle. Each of those re-bakes a fabricated
 * mask, and the design they belong to is settled in code — backend
 * `boxes.default_box_spec`, mirrored by api.ts::defaultBoxSpec — so exposing
 * them here only invited a preview that no longer describes the part being
 * made. The spec fields all survive; they are simply authored where the
 * decision is made.
 *
 * Pattern thumbnails are fetched lazily whenever the picker opens (GET
 * /patterns/{slug}/thumbnail is read-only: it serves the cached PNG or 404s,
 * never a generate). The store skips slugs that already loaded or are in
 * flight, so repeat calls are free — Section's onOpen fires on EVERY open,
 * which is what retries slugs that failed earlier. Failed slugs render as
 * retry tiles, never as an endless shimmer.
 */
export default function FaceEditor() {
  const selectedFaceId = useStore((s) => s.selectedFaceId);
  const face = useStore((s) => s.boxSpec.faces[selectedFaceId]);
  const catalog = useStore((s) => s.catalog);
  const thumbnails = useStore((s) => s.thumbnails);
  const metal = useStore((s) => s.boxSpec.metal ?? 'gold');
  const thumbnailErrors = useStore((s) => s.thumbnailErrors);
  const loadThumbnails = useStore((s) => s.loadThumbnails);
  const patchFace = useStore((s) => s.patchFace);

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

  // The picker offers the box's own constructions in PICKER_SLUGS order, not
  // whatever the catalog happens to list: a dev exemplar (or a slug a stale
  // backend still registers) must not be assignable to a wall from here.
  const offered = PICKER_SLUGS.map((slug) => catalog.find((c) => c.slug === slug)).filter(
    (c): c is NonNullable<typeof c> => c !== undefined
  );
  const descriptor = catalog.find((c) => c.slug === face.pattern_slug);
  const ready = offered.filter((c) => typeof thumbnails[c.slug] === 'string').length;
  const pending = offered.filter((c) => thumbnails[c.slug] === null).length;
  const failed = offered.filter((c) => thumbnailErrors[c.slug] !== undefined);

  return (
    <div data-testid="face-editor">
      <div
        style={{
          padding: '10px 12px 4px',
          fontSize: 11,
          letterSpacing: 1.5,
          opacity: 0.6,
        }}
      >
        EDITING: {FACE_LABELS[selectedFaceId].toUpperCase()}
      </div>

      {/* testId and persistId are UNCHANGED — `pattern-picker` is what the e2e
          specs target, and persistId keys the remembered open/closed state. */}
      <Section
        title="Face pattern"
        testId="pattern-picker"
        persistId="pattern"
        defaultOpen
        onOpen={() => {
          setPickerOpened(true);
          void loadThumbnails();
        }}
      >
        {pending > 0 && (
          <div
            data-testid="thumbnail-progress"
            style={{ fontSize: 11, opacity: 0.6, marginBottom: 8 }}
          >
            Rendering previews… {ready}/{offered.length}
          </div>
        )}
        <div
          data-testid="pattern-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 8,
          }}
        >
          {offered.map((c) => {
            const active = c.slug === face.pattern_slug;
            // Tint the catalog chip to the box's litho metal. The store's
            // map records AVAILABILITY (the read-only route 404s a cold slug);
            // the metal is a pure query on that same URL, and the route
            // re-colours the already-cached masks, so a chrome tile costs no
            // pattern generation. Gold needs no query — that is the route's
            // default and keeps its URL byte-identical to before.
            const base = thumbnails[c.slug];
            const thumb =
              typeof base === 'string' && metal !== 'gold'
                ? `${base}?metal=${encodeURIComponent(metal)}`
                : base;
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
                  padding: 5,
                  border: `2px solid ${active ? KIT.accent : KIT.border}`,
                  background: active ? KIT.raised : KIT.field,
                  borderRadius: 6,
                  color: KIT.text,
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 5,
                  fontSize: 11,
                  textAlign: 'center',
                  boxShadow: active ? `0 0 0 1px ${KIT.accent}` : 'none',
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
                ) : thumbnailErrors[c.slug] !== undefined ? (
                  <FailedTile
                    style={{ width: '100%', aspectRatio: '1 / 1' }}
                    title={`Preview failed: ${thumbnailErrors[c.slug]} — reopen this section or press Retry previews`}
                  />
                ) : (
                  <Shimmer style={{ width: '100%', aspectRatio: '1 / 1' }} />
                )}
                <div style={{ lineHeight: 1.25, fontWeight: active ? 600 : 400 }}>
                  {c.name}
                </div>
              </button>
            );
          })}
        </div>

        {descriptor && (
          <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.4, marginTop: 10 }}>
            {descriptor.description}
          </div>
        )}

        {failed.length > 0 && pending === 0 && (
          <div style={{ display: 'flex', marginTop: 10 }}>
            <Button
              testId="retry-thumbnails"
              title={`Refetch the ${failed.length} preview(s) whose generation failed`}
              onClick={() => {
                log('thumbnail_retry_clicked', { count: failed.length });
                void loadThumbnails();
              }}
              style={{ flex: 1, borderColor: KIT.error }}
            >
              ↻ Retry {failed.length} failed preview{failed.length === 1 ? '' : 's'}
            </Button>
          </div>
        )}
      </Section>

      <PhotoChoice faceId={selectedFaceId} />
    </div>
  );
}
