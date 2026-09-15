/**
 * Six-face assignment through the REAL UI (@faces).
 *
 * The workflow step "assign 6 face patterns" was previously covered only by
 * effectsHelpers::assignFacePattern, which calls `store.patchFace` inside
 * page.evaluate and is only ever invoked for the FRONT face. That skips the
 * entire user path — the face thumbnails (FacesPanel, `face-thumb-<id>`) and the
 * pattern cards (FaceEditor, `pattern-card-<slug>`, whose onClick also resets
 * pattern_params) — and it never touches the lid. A regression that bound the
 * wrong face, or that never fired for `top`, would leave the J+P monogram off
 * the engagement box lid with every other spec green.
 *
 * So this spec clicks: face thumbnail -> pattern card -> wait for the real
 * `face_texture_bound` for THAT face and slug, on two non-front faces, and then
 * asserts the six bound uFront textures are the six distinct plate PNGs the
 * manifest names — the assertion that actually kills a cross-face texture mixup.
 *
 * Slug choices matter: a card click for the slug a face ALREADY carries changes
 * nothing (same spec -> no regen -> no bind), so every target slug below is
 * asserted to differ from the face's current one first, and each is one of the
 * five slugs the picker offers (api.ts::PICKER_SLUGS) — i.e. a construction the
 * plate compositor really builds for this box.
 */
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { clearLog, expectLogEvent, waitForBoxTextures, waitForStudio } from './helpers';

/** Cold plate compose + mask load per assignment; be generous. */
const ASSIGN_TIMEOUT_MS = 180_000;

/**
 * Mirrors FACE_LABELS in src/ui/FacesPanel.tsx — the lid's spec id is `top` but
 * every label a user reads says "Lid". Asserting the banner is also how we know
 * FaceEditor has re-rendered for the newly selected face before we click a card.
 */
const FACE_LABEL: Record<string, string> = {
  front: 'Front',
  back: 'Back',
  top: 'Lid',
  bottom: 'Bottom',
  left: 'Left',
  right: 'Right',
};

/** The slug a face currently carries in the live spec. */
async function faceSlug(page: Page, faceId: string): Promise<string> {
  return (await page.evaluate(
    (fid) =>
      (window as any).__studio.store.getState().boxSpec.faces[fid]?.pattern_slug as string,
    faceId
  )) as string;
}

/** The pattern_params object a face currently carries. */
async function faceParams(page: Page, faceId: string): Promise<Record<string, unknown>> {
  return (await page.evaluate(
    (fid) => (window as any).__studio.store.getState().boxSpec.faces[fid]?.pattern_params ?? null,
    faceId
  )) as Record<string, unknown>;
}

type BoundMask = { src: string | null; wantSrc: string | null; slug: string };

/**
 * What each of the six faces currently has bound, next to what the held
 * manifest says it should be. One round-trip; safe to poll.
 *
 * `boundFront` (the URL BoxScene recorded when it uploaded the texture) rather
 * than the texture's own `image.src`: a LITERAL face — every production wall —
 * uploads a DataTexture decoded from the fabricated-chrome raster, and a
 * DataTexture has no src to compare. The manifest side follows the same
 * precedence the bind path uses (files.literal_front, else front_png).
 */
async function readBoundMasks(page: Page): Promise<Record<string, BoundMask>> {
  return (await page.evaluate(() => {
    const s = (window as any).__studio;
    const m = s?.store?.getState?.().boxManifest;
    const out: Record<string, BoundMask> = {};
    if (!m?.faces) return out;
    for (const fid of Object.keys(m.faces)) {
      const fm = m.faces[fid];
      out[fid] = {
        src: s.faces[fid]?.boundFront ?? null,
        wantSrc: fm?.files?.literal_front ?? fm?.files?.front_png ?? null,
        slug: fm?.spec?.pattern_slug ?? '',
      };
    }
    return out;
  })) as Record<string, BoundMask>;
}

/** True once every face's bound mask IS the one the manifest names. */
function allMasksMatchManifest(bound: Record<string, BoundMask>): boolean {
  const faces = Object.keys(bound);
  if (faces.length !== 6) return false;
  return faces.every((fid) => {
    const b = bound[fid];
    return !!b.src && !!b.wantSrc && b.src.endsWith(b.wantSrc);
  });
}

/**
 * Select a face in the Faces rail and assign `slug` by clicking its pattern
 * card, exactly as a user would. Returns once the new mask is bound to THAT
 * face. Fails fast (instead of hanging on a bind that will never come) if the
 * face already carries the slug.
 */
async function assignThroughUi(page: Page, faceId: string, slug: string): Promise<void> {
  const before = await faceSlug(page, faceId);
  expect(
    before,
    `[${faceId}] already carries '${slug}' — clicking its card produces no spec change, ` +
      `so no regen and no bind would ever fire. Pick a different slug.`
  ).not.toBe(slug);

  await page.getByTestId(`face-thumb-${faceId}`).click();
  await expect
    .poll(async () =>
      page.evaluate(() => (window as any).__studio.store.getState().selectedFaceId)
    )
    .toBe(faceId);
  // The editor must have re-rendered for THIS face before the card click — the
  // card's onClick captures selectedFaceId from the render that produced it.
  await expect(page.getByTestId('face-editor')).toContainText(
    `EDITING: ${FACE_LABEL[faceId].toUpperCase()}`
  );

  const card = page.getByTestId(`pattern-card-${slug}`);
  await expect(card).toBeVisible();
  await clearLog(page);
  await card.click();

  // The click itself is instrumented; then the debounced regen runs.
  await expectLogEvent(
    page,
    'face_pattern_changed',
    (e) => e.faceId === faceId && e.slug === slug
  );
  await expectLogEvent(page, 'box_regen_done', undefined, { timeout: ASSIGN_TIMEOUT_MS });
  await expectLogEvent(
    page,
    'face_texture_bound',
    (e) => e.face === faceId && e.slug === slug && e.recipe === 'foliage_moire',
    { timeout: ASSIGN_TIMEOUT_MS }
  );

  // The card the user pressed is the active one now.
  await expect(card).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId(`pattern-card-${before}`)).toHaveAttribute(
    'aria-pressed',
    'false'
  );
  expect(await faceSlug(page, faceId)).toBe(slug);
  // FaceEditor's card onClick clears pattern_params (the backend re-merges the
  // new class's defaults during materialize). Params are already {} on a
  // freshly booted spec, so this pins the reset rather than proving it.
  expect(await faceParams(page, faceId)).toEqual({});
}

test.describe('@faces per-face pattern assignment through the UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
    await waitForBoxTextures(page);
  });

  test('@faces the LID (top) takes a new pattern and comes back to the J+P monogram', async ({
    page,
  }) => {
    test.setTimeout(600_000);

    // The lid ships with monogram-jp, so assign something else first to prove
    // the click path really drives the top face...
    await assignThroughUi(page, 'top', 'globe-atlantic');

    // Let the five untouched faces finish rebinding (the bind effect loops all
    // six on every manifest change) so the stray check below sees the whole
    // regen, not just the face we waited on.
    await expect
      .poll(async () => allMasksMatchManifest(await readBoundMasks(page)), {
        timeout: ASSIGN_TIMEOUT_MS,
      })
      .toBe(true);

    // ...and no OTHER face picked up the lid's new slug (a cross-face mixup
    // would show here: all six face_texture_bound events for this regen are in
    // the buffer, which was cleared right before the click).
    const strays = await page.evaluate(() => {
      const buf = ((window as any).__log ?? []) as any[];
      return buf
        .filter(
          (e) =>
            e.type === 'face_texture_bound' && e.slug === 'globe-atlantic' && e.face !== 'top'
        )
        .map((e) => e.face);
    });
    // The FRONT wall also carries globe-atlantic, and every face rebinds on a
    // manifest change, so it is the one legitimate second binder of this slug.
    expect(
      strays.filter((f) => f !== 'front'),
      'globe-atlantic bound to a face other than the lid and the front wall'
    ).toEqual([]);

    // ...then put the engagement-box monogram back, through the same UI.
    await assignThroughUi(page, 'top', 'monogram-jp');
  });

  test('@faces a second non-front face rebinds and all six faces keep distinct masks', async ({
    page,
  }) => {
    test.setTimeout(600_000);

    await assignThroughUi(page, 'right', 'solid-gold');

    // Every face must end up bound to the mask the CURRENT manifest names for
    // it. The five untouched faces rebind too (the bind effect loops all six on
    // every manifest change), so allow a moment for their loads to land.
    await expect
      .poll(async () => allMasksMatchManifest(await readBoundMasks(page)), {
        timeout: ASSIGN_TIMEOUT_MS,
      })
      .toBe(true);

    const bound = await readBoundMasks(page);
    const faces = Object.keys(bound);
    expect(faces.length, `manifest faces: ${faces.join(',')}`).toBe(6);
    for (const [fid, b] of Object.entries(bound)) {
      expect(b.wantSrc, `[${fid}] manifest has no front_png`).toBeTruthy();
      expect(b.src, `[${fid}] uFront is not a loaded image`).toBeTruthy();
      // front_png is root-relative ('/data/plates/<hash>/front.png'); the loaded
      // image src is that path absolutized against the dev server.
      expect(
        b.src!.endsWith(b.wantSrc!),
        `[${fid}] bound mask '${b.src}' is not the manifest's '${b.wantSrc}'`
      ).toBe(true);
    }
    // Six plates, six distinct hashes: same-slug faces still differ (each face
    // has its own cut dims + frame seed), so a shared source means two faces are
    // sampling ONE texture — the mixup this spec exists to catch.
    const srcs = Object.values(bound).map((b) => b.src);
    expect(new Set(srcs).size, `bound mask sources: ${JSON.stringify(bound, null, 2)}`).toBe(6);
    // And the right wall really is the pattern we clicked.
    expect(bound.right.slug).toBe('solid-gold');
    expect(bound.top.slug).toBe('monogram-jp');
  });
});
