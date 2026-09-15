import { PHOTO_COLOUR_MODE, PHOTO_PARAM, PHOTO_PATTERN_SLUG, type FaceId } from '../api';
import { log } from '../logger';
import { useStore } from '../store';
import { Section, SelectRow } from './kit';

/**
 * The one design choice the visualizer still makes: WHICH prepared photograph
 * a photo wall carries.
 *
 * Everything else about a face — the pattern, the garland's dials, the band
 * width, the seed, the ply policy — is decided in code (backend
 * `boxes.default_box_spec`, mirrored by api.ts::defaultBoxSpec) because those
 * are fab decisions, not view decisions: changing one re-bakes a 2 µm
 * gold-on-quartz mask. Choosing between six photographs that are already
 * prepared, cropped and colour-authored is not.
 *
 * The choices come from the CATALOG, not from a list here: the backend's
 * `image` ParamSpec is built by `photo.available_photos()`, i.e. from the PNGs
 * that actually sit in app/assets/photos. A second copy of those names on this
 * side would offer a picture the backend cannot screen. `colour_mode` rides
 * along as 'authored' because every prepared photograph ships its
 * `<image>.colour.json` plan and the box is designed around them.
 *
 * Renders nothing on a non-photo face.
 */
export default function PhotoChoice({ faceId }: { faceId: FaceId }) {
  const face = useStore((s) => s.boxSpec.faces[faceId]);
  const catalog = useStore((s) => s.catalog);
  const patchFace = useStore((s) => s.patchFace);

  if (!face || face.pattern_slug !== PHOTO_PATTERN_SLUG) return null;

  const descriptor = catalog.find((c) => c.slug === PHOTO_PATTERN_SLUG);
  const choices = descriptor?.params.find((p) => p.name === PHOTO_PARAM)?.choices ?? [];
  // Catalog not in yet (boot race), or a backend whose photo folder is empty:
  // say so rather than rendering an empty select that silently does nothing.
  if (choices.length === 0) {
    return (
      <Section title="Photograph" testId="photo-choice" persistId="photo">
        <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5 }}>
          No prepared photographs listed by the backend yet.
        </div>
      </Section>
    );
  }

  const current = String(face.pattern_params[PHOTO_PARAM] ?? descriptor?.params
    .find((p) => p.name === PHOTO_PARAM)?.default ?? choices[0]);

  return (
    <Section title="Photograph" testId="photo-choice" persistId="photo" defaultOpen>
      <SelectRow
        label="Picture"
        value={current}
        options={choices.map((c) => ({ value: c, label: c }))}
        onChange={(image) => {
          log('face_photo_changed', { faceId, image });
          patchFace(faceId, {
            pattern_params: {
              ...face.pattern_params,
              [PHOTO_PARAM]: image,
              colour_mode: PHOTO_COLOUR_MODE,
            },
          });
        }}
        testId="photo-select"
      />
      <div style={{ fontSize: 11, opacity: 0.6, lineHeight: 1.5, marginTop: 2 }}>
        Halftoned as a gold line screen, with this picture&rsquo;s authored
        colour plan driving the diffraction sub-gratings.
      </div>
    </Section>
  );
}
