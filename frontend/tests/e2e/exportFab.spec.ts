/**
 * @export — the last step of the user journey: download the fab bundle.
 *
 * This is the app's actual deliverable (six plate folders with fine.gds +
 * previews, CUTLIST.csv, ASSEMBLY.md), and until this spec existed nothing
 * clicked the button. Three contracts are pinned here — staleness, progress and
 * content in one real flow, plus a fully mocked spec for the two job-path
 * failures (busy slot, dead worker) that must never run a second real export on
 * this host:
 *
 *  1. STALENESS. Export is a fetch-driven button that refuses while the held
 *     manifest could disagree with the screen (App.tsx::exportBlockedReason).
 *     We edit a manifest-only field — hinge ROD OD, which changes ASSEMBLY.md
 *     but no mask — through the real DOM control, and assert the button goes
 *     disabled with a visible reason until `box_regen_done`. That is the
 *     regression the audit called out: drop `hinge` from App.tsx::specRegenKey
 *     and the UI still looks right while the bundle ships the OLD hinge cut
 *     list. With hinge out of the key no POST fires and the button never goes
 *     disabled, so both assertions below fail.
 *
 *  1b. PROGRESS. Export is a polled JOB now (POST /export/box/{id}/fab/start,
 *     GET /export/jobs/{id}, GET /export/jobs/{id}/fab.zip), and the user's
 *     "not hung" contract is that the progress strip's TEXT MOVES while a real
 *     build runs — a static "the first export is slow" is exactly what this
 *     replaces. The spec samples `export-progress` throughout the export and
 *     requires at least two distinct readings whenever the export took long
 *     enough to have had a second poll (PROGRESS_CHANGE_FLOOR_MS); on a warm
 *     cache a job can legitimately finish inside one poll, and demanding a
 *     change there would be a flake, not a pass. The `export_progress` events
 *     are checked to be one-per-phase-change, never one-per-poll.
 *
 *  2. CONTENT. The zip we actually receive is unpacked in-process and its
 *     CUTLIST.csv rows are compared cell-for-cell against the on-screen
 *     cut-list testids, its ASSEMBLY.md against the edited rod OD, and its
 *     box.json spec against the live store — so "the bundle matches the
 *     screen" is measured, not assumed. On a 2 um gold-on-quartz run a
 *     mismatch here is an unrecoverable fab error.
 *
 * No zip library is in devDependencies (and adding one for a test isn't worth
 * a supply-chain entry), so `readZipDirectory`/`readZipEntry` below are a ~40
 * line central-directory reader over node:zlib's raw inflate. The backend
 * writes the archive with python zipfile ZIP_DEFLATED (api/export.py::
 * build_box_fab_zip, via the job worker), i.e. stored/deflated entries with
 * real sizes in the headers and no zip64 — exactly the subset handled here.
 *
 * Budget: a cold export composes any missing plate and then builds six
 * DRC-healed fine.gds masks SEQUENTIALLY server-side (machine constraint), so
 * the download can legitimately take minutes on this host. Hence the generous
 * test.setTimeout — kept well above every inner wait so a slow build fails
 * with the informative log dump from expectLogEvent, never a bare Playwright
 * timeout (the trap this suite hit before: a 90 s wait inside a 60 s test).
 */
import { readFile } from 'node:fs/promises';
import { inflateRawSync } from 'node:zlib';
import type { APIRequestContext, Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { clearLog, expectLogEvent, readLog, waitForStudio } from './helpers';

/** Faces in the order BuildPanel renders the cut-list rows. */
const FACES = ['bottom', 'top', 'front', 'back', 'left', 'right'] as const;

/** The first box generate of a run can be a full six-plate compose. */
const BOOT_REGEN_WAIT_MS = 180_000;
/** The hinge-only regen: every per-face plate cache hits, manifest only. */
const REGEN_WAIT_MS = 120_000;
/** The download itself: six sequential DRC-healed fab masks, one at a time. */
const DOWNLOAD_WAIT_MS = 300_000;
/** One-time cold fine-mask build (six faces through klayout DRC). */
const FINE_WARM_MS = 600_000;
/**
 * Above this, the export ran long enough that the 1 Hz poll MUST have produced
 * a second, different phase — so a frozen strip is a real bug. Below it (warm
 * caches: every plate hits, every fine.gds slot hits, the worker can finish
 * inside one poll interval) a single reading is legitimate and asserting on a
 * change would only buy flakes. 2.5 s = two poll intervals plus slack.
 */
const PROGRESS_CHANGE_FLOOR_MS = 2_500;
/** How often the spec samples the progress strip while an export runs. */
const PROGRESS_SAMPLE_MS = 200;
/**
 * Test budget = the sum of the inner waits plus slack, so a slow step always
 * fails with its own diagnostic (expectLogEvent dumps the log buffer) instead
 * of a bare Playwright timeout. This ceiling is only approached on a fully
 * cold backend/data cache; a warm one finishes the whole test in well under a
 * minute.
 */
const TEST_BUDGET_MS = BOOT_REGEN_WAIT_MS + REGEN_WAIT_MS + DOWNLOAD_WAIT_MS + 60_000;

// -----------------------------------------------------------------------------
// Minimal ZIP reader (central directory + raw inflate). See the header note.
// -----------------------------------------------------------------------------

type ZipEntry = {
  name: string;
  /** 0 = stored, 8 = deflate. Anything else is rejected. */
  method: number;
  compSize: number;
  uncompSize: number;
  localOffset: number;
};

const SIG_EOCD = 0x06054b50; // PK\x05\x06 — end of central directory
const SIG_CDIR = 0x02014b50; // PK\x01\x02 — central directory entry
const SIG_LOCAL = 0x04034b50; // PK\x03\x04 — local file header

/** Parse the central directory into name -> entry. */
function readZipDirectory(buf: Buffer): Map<string, ZipEntry> {
  if (buf.length < 22) throw new Error(`not a zip: only ${buf.length} bytes`);
  // The EOCD sits in the last 22 bytes + up to 64 KiB of archive comment.
  let eocd = -1;
  const floor = Math.max(0, buf.length - (0xffff + 22));
  for (let i = buf.length - 22; i >= floor; i--) {
    if (buf.readUInt32LE(i) === SIG_EOCD) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('not a zip: no end-of-central-directory record');
  const count = buf.readUInt16LE(eocd + 10);
  const cdOffset = buf.readUInt32LE(eocd + 16);
  if (count === 0xffff || cdOffset === 0xffffffff) {
    throw new Error('zip64 archive — this reader only handles the classic format');
  }
  const entries = new Map<string, ZipEntry>();
  let p = cdOffset;
  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== SIG_CDIR) {
      throw new Error(`corrupt central directory: bad signature at byte ${p}`);
    }
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const name = buf.subarray(p + 46, p + 46 + nameLen).toString('utf8');
    entries.set(name, {
      name,
      method: buf.readUInt16LE(p + 10),
      compSize: buf.readUInt32LE(p + 20),
      uncompSize: buf.readUInt32LE(p + 24),
      localOffset: buf.readUInt32LE(p + 42),
    });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

/** Decompressed bytes of one entry. Throws (listing the archive) if absent. */
function readZipEntry(buf: Buffer, entries: Map<string, ZipEntry>, name: string): Buffer {
  const e = entries.get(name);
  if (!e) {
    throw new Error(
      `zip entry '${name}' missing. Archive holds: ${[...entries.keys()].sort().join(', ')}`
    );
  }
  if (buf.readUInt32LE(e.localOffset) !== SIG_LOCAL) {
    throw new Error(`corrupt local header for '${name}' at byte ${e.localOffset}`);
  }
  // Name/extra lengths come from the LOCAL header: the extra field routinely
  // differs in size between the two copies.
  const nameLen = buf.readUInt16LE(e.localOffset + 26);
  const extraLen = buf.readUInt16LE(e.localOffset + 28);
  const start = e.localOffset + 30 + nameLen + extraLen;
  const raw = buf.subarray(start, start + e.compSize);
  if (e.method === 0) return Buffer.from(raw);
  if (e.method === 8) return inflateRawSync(raw);
  throw new Error(`unsupported compression method ${e.method} for '${name}'`);
}

const readZipText = (buf: Buffer, entries: Map<string, ZipEntry>, name: string): string =>
  readZipEntry(buf, entries, name).toString('utf8');

type CutRow = {
  width_mm: number;
  height_mm: number;
  thickness_mm: number;
  width_um: number;
  height_um: number;
};

/** CUTLIST.csv -> face -> row. Pins the header too (it is a fab interface). */
function parseCutlist(csv: string): Map<string, CutRow> {
  const lines = csv.trim().split(/\r?\n/);
  expect(lines[0]).toBe('face,width_mm,height_mm,thickness_mm,width_um,height_um');
  const rows = new Map<string, CutRow>();
  for (const line of lines.slice(1)) {
    const [face, w, h, t, wu, hu] = line.split(',');
    rows.set(face, {
      width_mm: Number(w),
      height_mm: Number(h),
      thickness_mm: Number(t),
      width_um: Number(wu),
      height_um: Number(hu),
    });
  }
  return rows;
}

/**
 * Pre-warm the per-plate fine.gds cache (api/export.py FINE_GDS_VERSION slots)
 * through the JOB api, best-effort.
 *
 * The FIRST export after a `just clean` builds six true-pitch masks through
 * klayout DRC — minutes on this host — and that one-time cold build is not the
 * UX contract these specs pin. Driving it here (and polling the job to
 * completion, so no worker is still holding the host-wide export slot when the
 * UI export starts) leaves the warmed slots valid for the assertions below: the
 * hinge edit does not change any plate hash.
 *
 * Every failure is swallowed: a warm run may answer 429/404 and the specs must
 * still be meaningful without the warm-up (just slower).
 */
async function prewarmFabMasks(request: APIRequestContext, boxId: string): Promise<void> {
  try {
    const start = await request.post(`/export/box/${boxId}/fab/start`, { timeout: 30_000 });
    if (!start.ok()) return;
    const { job_id: jobId } = (await start.json()) as { job_id: string };
    const deadline = Date.now() + FINE_WARM_MS;
    while (Date.now() < deadline) {
      const r = await request.get(`/export/jobs/${jobId}`, { timeout: 30_000 });
      if (!r.ok()) return;
      const s = (await r.json()) as { status: string };
      if (s.status !== 'running') return;
      await new Promise<void>((resolve) => {
        setTimeout(() => resolve(), 2000);
      });
    }
  } catch {
    /* best-effort warm-up — the specs below do not depend on it */
  }
}

/**
 * Sample the `export-progress` strip until `stop()` is called, collecting every
 * DISTINCT reading. This is the "not hung" measurement: what the user sees
 * change, read the way they read it (rendered text), not from the log buffer.
 */
function watchProgressText(page: Page): { readings: Set<string>; stop: () => Promise<void> } {
  const readings = new Set<string>();
  let running = true;
  const loop = (async () => {
    while (running) {
      const txt = await page
        .getByTestId('export-progress')
        .textContent({ timeout: 5_000 })
        .catch(() => null);
      if (txt && txt.trim()) readings.add(txt.trim());
      await page.waitForTimeout(PROGRESS_SAMPLE_MS).catch(() => undefined);
    }
  })();
  return {
    readings,
    stop: async () => {
      running = false;
      await loop.catch(() => undefined);
    },
  };
}

test.describe('@export fab bundle', () => {
  test('@export hinge edit blocks export until regen, then the zip matches the screen', async ({
    page,
    request,
  }) => {
    test.setTimeout(TEST_BUDGET_MS + FINE_WARM_MS);
    await page.goto('/');
    await waitForStudio(page);
    // Boot regen — the manifest the export button will serve.
    await expectLogEvent(page, 'box_regen_done', undefined, {
      timeout: BOOT_REGEN_WAIT_MS,
    });

    // Warm the fine-mask cache out of band (see prewarmFabMasks). '__scratch'
    // is the id the live app's own generate lands on (boxes.py::SCRATCH_BOX_ID).
    await prewarmFabMasks(request, '__scratch');
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    await expect(page.getByTestId('export-blocked-reason')).toHaveCount(0);

    // Slow the NEXT generate down so the stale window is comfortably wider
    // than the 400 ms regen debounce. Without this the whole
    // edit -> disabled -> regen -> enabled transition can complete inside one
    // assertion poll on a warm cache, which is a flake, not a pass. The route
    // also counts the POSTs, which is how we know the hinge edit is really in
    // App.tsx::specRegenKey.
    let generatePosts = 0;
    await page.route('**/boxes/generate', async (route) => {
      generatePosts += 1;
      await new Promise<void>((resolve) => {
        setTimeout(() => resolve(), 1200);
      });
      await route.continue();
    });
    await clearLog(page);

    // Hinge ROD OD, driven through the real NumberRow (commits on Enter — see
    // kit.tsx::NumberRow, which deliberately does NOT patch per keystroke).
    // 1800 um stays under the 2400 um tube OD, so the spec remains VALID: this
    // must exercise the staleness gate, not the validation gate.
    const rodOd = page.getByTestId('hinge-rod-od');
    await rodOd.fill('1800');
    await rodOd.press('Enter');
    expect(
      await page.evaluate(
        () => (window as any).__studio.store.getState().boxSpec.hinge.rod_od_um
      )
    ).toBe(1800);

    // Manifest-only edit, so the 3D scene never changed — but the bundle would
    // have. Export must refuse, with the reason on screen.
    await expect(page.getByTestId('export-fab')).toBeDisabled();
    await expect(page.getByTestId('export-blocked-reason')).toBeVisible();
    await expect(page.getByTestId('export-blocked-reason')).toContainText('Design changed');

    await expectLogEvent(page, 'box_regen_done', undefined, { timeout: REGEN_WAIT_MS });
    await page.unroute('**/boxes/generate');
    // A hinge edit that fired no generate means it fell out of the regen key.
    expect(
      generatePosts,
      'the hinge edit must POST /boxes/generate — it is part of specRegenKey'
    ).toBeGreaterThanOrEqual(1);
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    await expect(page.getByTestId('export-blocked-reason')).toHaveCount(0);

    // Snapshot what the SCREEN claims, before asking for the bundle.
    const screenCuts = new Map<string, { width: string; height: string }>();
    for (const face of FACES) {
      const cells = await page.getByTestId(`cut-${face}`).locator('td').allTextContents();
      screenCuts.set(face, { width: cells[1].trim(), height: cells[2].trim() });
    }
    const live = (await page.evaluate(() => {
      const st = (window as any).__studio.store.getState();
      return {
        id: st.boxManifest.id as string,
        contentHash: st.boxManifest.content_hash as string,
        width_um: st.boxSpec.width_um as number,
        depth_um: st.boxSpec.depth_um as number,
        height_um: st.boxSpec.height_um as number,
        rod_od_um: st.boxSpec.hinge.rod_od_um as number,
        segments: st.boxSpec.hinge.segments as number,
      };
    })) as {
      id: string;
      contentHash: string;
      width_um: number;
      depth_um: number;
      height_um: number;
      rod_od_um: number;
      segments: number;
    };

    // Blob-anchor downloads raise the same 'download' event as a server one.
    const downloadPromise = page.waitForEvent('download', { timeout: DOWNLOAD_WAIT_MS });
    const watcher = watchProgressText(page);
    const clickedAt = Date.now();
    await page.getByTestId('export-fab').click();
    // Race-free busy proof: the log entry persists, the spinner text does not.
    await expectLogEvent(page, 'export_started', (e) => e.id === live.id);
    const download = await downloadPromise;
    const exportMs = Date.now() - clickedAt;
    await watcher.stop();
    // Asserted first: the confirmation chip self-clears 6 s after the blob is
    // handed to the shelf (App.tsx), so it is the one perishable signal here.
    await expect(page.getByTestId('export-done')).toBeVisible();

    // ---- the "not hung" contract ------------------------------------------
    // What the strip actually SAID, sampled at 200 ms. Whatever it said had to
    // come from the job payload — a concrete position, never a please-wait.
    const readings = [...watcher.readings];
    for (const t of readings) {
      expect(
        /Masks: (\d+\/\d+|preparing)/.test(t) || /Starting export/.test(t),
        `export strip should report the worker's real position, saw: ${JSON.stringify(t)}`
      ).toBe(true);
    }
    if (exportMs > PROGRESS_CHANGE_FLOOR_MS) {
      // Long enough for a second poll: a strip that never moved is a build the
      // user cannot distinguish from a wedged one. (Below the floor a warm job
      // can finish inside one poll, so a single reading is honest.)
      expect(
        readings.length,
        `export took ${exportMs}ms but the progress text never changed: ` +
          `${JSON.stringify(readings)}`
      ).toBeGreaterThan(1);
    }
    // The same contract in the log, where sampling cannot miss anything:
    // export_progress fires once per PHASE CHANGE, never once per poll, so no
    // two events may carry the same job|phase|face|faces_done key.
    const progressEvents = (await readLog(page)).filter((e) => e.type === 'export_progress');
    expect(progressEvents.length, 'the export should have logged its phases').toBeGreaterThan(0);
    if (exportMs > PROGRESS_CHANGE_FLOOR_MS) {
      expect(
        progressEvents.length,
        `export took ${exportMs}ms with only one phase logged: ${JSON.stringify(progressEvents)}`
      ).toBeGreaterThan(1);
    }
    // `queued_behind` is in the key because a run that had to wait out another
    // export reports that job's phases too — the same triple from a different
    // job is not a duplicate (App.tsx::waitOutRunningExport).
    const keys = progressEvents.map((e) =>
      JSON.stringify([e.queued_behind ?? '', e.phase, e.face ?? '', e.faces_done, e.faces_total])
    );
    expect(
      new Set(keys).size,
      `export_progress must fire once per phase change, not per poll: ${JSON.stringify(keys)}`
    ).toBe(keys.length);
    await expect(page.getByTestId('regen-error')).toHaveCount(0);
    expect(await download.failure()).toBeNull();
    // App.tsx names the blob after the server's Content-Disposition.
    expect(download.suggestedFilename()).toBe(`box-${live.id}.zip`);

    const zipPath = await download.path();
    expect(zipPath, 'download.path() should resolve for an accepted download').toBeTruthy();
    const zip = await readFile(zipPath!);
    expect(zip.byteLength).toBeGreaterThan(1024);

    const doneEv = await expectLogEvent(page, 'export_done', (e) => e.id === live.id);
    expect(doneEv.content_hash).toBe(live.contentHash);
    expect(Number(doneEv.bytes)).toBe(zip.byteLength);

    // ---- bundle structure -------------------------------------------------
    const entries = readZipDirectory(zip);
    const rootFiles = [
      'CUTLIST.csv',
      'ASSEMBLY.md',
      'README.txt',
      'box.json',
      'FINE_MASKS.json',
    ];
    for (const name of rootFiles) {
      expect(entries.has(name), `bundle should contain ${name}`).toBe(true);
    }
    for (const face of FACES) {
      expect(
        entries.has(`${face}/manifest.json`),
        `bundle should contain the ${face} plate folder`
      ).toBe(true);
    }
    // The README has to keep telling the engraver which file is the mask —
    // front.svg's periods may be coarsened (api/export.py::_FINE_VS_PREVIEW_TEXT).
    expect(readZipText(zip, entries, 'README.txt')).toContain(
      'fine.gds is THE fab-grade mask'
    );
    const fineMasks = JSON.parse(readZipText(zip, entries, 'FINE_MASKS.json')) as {
      faces?: Record<string, unknown>;
    };
    expect(typeof fineMasks.faces).toBe('object');

    // ---- box.json is the design ON SCREEN, not a previous one -------------
    const boxJson = JSON.parse(readZipText(zip, entries, 'box.json')) as {
      id: string;
      content_hash: string;
      spec: { hinge: { rod_od_um: number } };
      faces: Record<string, unknown>;
    };
    expect(boxJson.id).toBe(live.id);
    expect(boxJson.content_hash).toBe(live.contentHash);
    // The whole point: the field we edited through the DOM reached the bundle.
    expect(boxJson.spec.hinge.rod_od_um).toBe(live.rod_od_um);
    expect(Object.keys(boxJson.faces).length).toBe(FACES.length);

    // ---- CUTLIST.csv == the on-screen cut list ----------------------------
    const cutRows = parseCutlist(readZipText(zip, entries, 'CUTLIST.csv'));
    expect([...cutRows.keys()].sort()).toEqual([...FACES].sort());
    for (const face of FACES) {
      const row = cutRows.get(face)!;
      const shown = screenCuts.get(face)!;
      // BuildPanel prints width_mm/height_mm at one decimal; the CSV carries
      // the same backend-rounded mm value plus its um twin.
      expect(row.width_mm.toFixed(1), `CUTLIST ${face} width`).toBe(shown.width);
      expect(row.height_mm.toFixed(1), `CUTLIST ${face} height`).toBe(shown.height);
      // um everywhere / mm only as a derived display field (CLAUDE.md units).
      expect(row.width_um).toBeCloseTo(row.width_mm * 1000, 0);
      expect(row.height_um).toBeCloseTo(row.height_mm * 1000, 0);
    }

    // ---- ASSEMBLY.md == the same numbers, in prose ------------------------
    const assemblyMd = readZipText(zip, entries, 'ASSEMBLY.md');
    expect(assemblyMd).toContain(
      `Outer dimensions: ${(live.width_um / 1000).toFixed(1)} x ` +
        `${(live.depth_um / 1000).toFixed(1)} x ${(live.height_um / 1000).toFixed(1)} mm`
    );
    // The edited rod OD, formatted the way _assembly_md writes it (1800 -> 1.80).
    expect(assemblyMd).toContain(
      `Cut the brass rod (${(live.rod_od_um / 1000).toFixed(2)} mm OD)`
    );
    expect(assemblyMd).toContain(`Thread the rod through all ${live.segments} segments`);
    // Cut table rows carry the same mm as the CSV and the panel.
    for (const face of FACES) {
      const shown = screenCuts.get(face)!;
      expect(assemblyMd, `ASSEMBLY.md cut row for ${face}`).toContain(
        `| ${face} | ${shown.width} | ${shown.height} |`
      );
    }
  });

  test('@export a failing export surfaces in the banner and the log, and downloads nothing', async ({
    page,
  }) => {
    test.setTimeout(BOOT_REGEN_WAIT_MS + 90_000);
    await page.goto('/');
    await waitForStudio(page);
    await expectLogEvent(page, 'box_regen_done', undefined, {
      timeout: BOOT_REGEN_WAIT_MS,
    });
    await expect(page.getByTestId('export-fab')).toBeEnabled();

    // A refused job start — no worker can be spawned, the box is gone, the
    // backend died. The old bare <a download> reported this only in the
    // browser's download shelf.
    await page.route('**/export/box/*/fab/start', (route) =>
      route.fulfill({ status: 500, body: 'forced export failure' })
    );
    let downloads = 0;
    page.on('download', () => {
      downloads += 1;
    });
    await clearLog(page);

    await page.getByTestId('export-fab').click();
    const failed = await expectLogEvent(page, 'export_failed', undefined, {
      timeout: 60_000,
    });
    expect(String(failed.error)).toContain('forced export failure');
    await expect(page.getByTestId('regen-error')).toBeVisible();
    await expect(page.getByTestId('regen-error')).toContainText('Fab export');
    await expect(page.getByTestId('export-done')).toHaveCount(0);
    expect(downloads, 'a failed export must not hand a partial file to the shelf').toBe(0);
    // Not wedged: the busy flag cleared and the button is clickable again.
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    await expect(page.getByTestId('export-progress')).toHaveCount(0);
    await page.unroute('**/export/box/*/fab/start');
  });

  test('@export a busy slot and a dead worker both explain themselves', async ({ page }) => {
    // Fully mocked: no real export runs here (one heavy compute at a time is a
    // HOST constraint — CLAUDE.md), so this can pin the two job-path failures
    // the content test above can never reach.
    test.setTimeout(BOOT_REGEN_WAIT_MS + 90_000);
    await page.goto('/');
    await waitForStudio(page);
    await expectLogEvent(page, 'box_regen_done', undefined, {
      timeout: BOOT_REGEN_WAIT_MS,
    });
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    let downloads = 0;
    page.on('download', () => {
      downloads += 1;
    });

    // 1. 429 — the host-wide export slot is held by another job. App.tsx tries
    //    to wait it out (polling the id named in the detail, which this backend
    //    has never heard of -> 404 -> nothing to wait on) and then retries the
    //    start exactly once; a second 429 is reported with the retry sentence.
    //    It must NEVER download that job's archive: it can predate the screen.
    await page.route('**/export/box/*/fab/start', (route) =>
      route.fulfill({
        status: 429,
        contentType: 'application/json',
        body: JSON.stringify({
          detail:
            'export job deadbeef1234 (box other-box) is still running — ' +
            'one export at a time; retry when it finishes',
        }),
      })
    );
    await clearLog(page);
    await page.getByTestId('export-fab').click();
    const busy = await expectLogEvent(page, 'export_failed', undefined, { timeout: 30_000 });
    expect(String(busy.error)).toContain('is still running');
    // The user must be told what to DO, once (App.tsx adds guidance only when
    // the backend's own sentence carries none).
    expect(String(busy.error)).toMatch(/retry|try again/i);
    await expect(page.getByTestId('regen-error')).toBeVisible();
    await expect(page.getByTestId('regen-error')).toContainText('still running');
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    await page.unroute('**/export/box/*/fab/start');

    // 2. A job that starts and then dies (worker killed / klayout crash). The
    //    poller must surface the worker's diagnosis, not hang on 'running'.
    await page.route('**/export/box/*/fab/start', (route) =>
      route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'mockjob',
          box_id: '__scratch',
          status: 'running',
          progress: {
            phase: 'starting',
            face: null,
            faces_done: 0,
            faces_total: 6,
            detail: 'worker starting',
            updated_at: 0,
          },
          poll_url: '/export/jobs/mockjob',
          download_url: '/export/jobs/mockjob/fab.zip',
        }),
      })
    );
    await page.route('**/export/jobs/mockjob', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'mockjob',
          box_id: '__scratch',
          status: 'failed',
          progress: {
            phase: 'failed',
            face: 'right',
            faces_done: 2,
            faces_total: 6,
            detail: 'klayout heal died',
            updated_at: 7,
          },
          error: 'export worker exited with code 1 and produced no archive',
          returncode: 1,
          download_url: null,
        }),
      })
    );
    await clearLog(page);
    await page.getByTestId('export-fab').click();
    const dead = await expectLogEvent(page, 'export_failed', undefined, { timeout: 30_000 });
    expect(String(dead.error)).toContain('export worker exited with code 1');
    expect(dead.job_id).toBe('mockjob');
    await expect(page.getByTestId('regen-error')).toContainText('export worker exited');
    // Whatever failed, nothing half-built reached the download shelf and the
    // button is usable again with no orphaned progress strip.
    expect(downloads, 'a failed export must not hand a partial file to the shelf').toBe(0);
    await expect(page.getByTestId('export-done')).toHaveCount(0);
    await expect(page.getByTestId('export-fab')).toBeEnabled();
    await expect(page.getByTestId('export-progress')).toHaveCount(0);
    await page.unroute('**/export/jobs/mockjob');
    await page.unroute('**/export/box/*/fab/start');
  });
});
