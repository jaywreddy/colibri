/**
 * Structured frontend event log.
 *
 * Every meaningful UI action pushes a record to `window.__log`. The buffer
 * is capped at 500 events (oldest dropped on overflow) so long sessions
 * don't grow unbounded. Playwright specs read this buffer to verify that
 * user actions produce the expected downstream work, and a Playwright
 * afterEach attaches the full buffer to any failing test for post-mortem
 * debugging.
 *
 * Event record shape: `{ t, type, ...payload }` where `t` is
 * `performance.now()` at push time.
 *
 * In dev the event is also mirrored to `console.debug` with a `[optics]`
 * tag so you can watch the log scroll by in DevTools without opening the
 * buffer inspector.
 */

export type LogEvent = {
  /** Milliseconds since page load (performance.now()). */
  t: number;
  /** Event name — one of the canonical strings from `EventType`. */
  type: string;
  /** Event-specific payload (slug, duration_ms, etc.). */
  [key: string]: unknown;
};

/**
 * Canonical event names — the single source of truth for what the app emits
 * and what specs may assert against.
 *
 * This list is REGISTRY, not enforcement: `log()` still takes a plain string
 * so a new event never fails a build. It had silently drifted (21 live events
 * missing, plus `catalog_load_failed` listed for a call site that no longer
 * exists — the boot retry emits `catalog_load_retrying` instead), which made
 * the "single source of truth" claim false and left specs asserting on names
 * nobody could look up. Two things now keep it honest:
 *   - it is a runtime array, so a test can pin it against a grep of
 *     `src/**` and against the names any spec asserts on;
 *   - `log()` warns in DEV on an unregistered name (see below), so the next
 *     new event announces itself the first time it fires.
 * Keep the ordering grouped by lifecycle — that grouping is how you find the
 * neighbouring events of whatever you are debugging.
 */
export const EVENT_TYPES = [
  // --- boot: pattern catalogue fetch (retries forever, see App.tsx) --------
  'catalog_loaded',
  'catalog_load_retrying',

  // --- box regeneration lifecycle (App.tsx::regen) -------------------------
  'box_regen_start',
  'box_regen_done',
  'box_regen_failed',
  /** A newer request superseded this one; its response was dropped. */
  'box_regen_stale',
  /** Backend-warmup / transient-failure retry was scheduled. */
  'box_regen_retry',
  /** Spec failed client-side validation — no POST was made. */
  'box_regen_skipped_invalid',

  // --- box spec persistence + presets -------------------------------------
  'box_saved',
  'box_loaded',
  'box_reset',
  'size_preset_applied',

  // (The fab-bundle export events are gone with the button: the archive is
  // built by the CLI against a named box id, not pulled through the browser.)

  // --- design edits (these are the ones that dirty the fab masks) ---------
  'face_clicked',
  'face_pattern_changed',
  /** Which prepared photograph a photo wall carries (ui/PhotoChoice.tsx). */
  'face_photo_changed',
  'foil_tape_changed',

  // --- preview-only controls (no backend regen, no fab effect) ------------
  'pattern_scale_changed',
  'lid_changed',
  'layout_changed',
  'autorotate_toggled',
  'illumination_changed',
  'light_moved',
  'laser_color_changed',

  // --- scene build + per-face texture/recipe binding (BoxScene.tsx) -------
  'box_scene_rebuilt',
  'face_texture_bound',
  /** Bounded retry after a mask PNG failed to load. */
  'face_texture_retry',
  'face_texture_failed',
  'diffraction_lut_loaded',
  'diffraction_lut_failed',
  /** Manifest asked for a recipe the two-plane renderer refuses to fake. */
  'face_recipe_unsupported',
  /** Geometry-critical recipe_data keys absent — face renders degraded. */
  'face_recipe_data_incomplete',
  /**
   * A LITERAL raster carried no chrome at all, so that pattern plane was
   * dropped instead of uploading an empty 2048² texture: the back of a
   * single-ply face, or both layers of a blank one. The glass slabs stay.
   */
  'face_layer_empty',

  // --- WebGL context lifecycle / recovery --------------------------------
  'webgl_init_failed',
  'webgl_context_lost',
  'webgl_context_restored',
  /** Lost context never came back within the restore window. */
  'webgl_restore_timeout',
  /** User took the offered reload out of an unrecoverable context loss. */
  'webgl_view_reloaded',

  // --- pattern-picker thumbnails (store.ts::loadThumbnails) --------------
  'thumbnail_loaded',
  'thumbnail_load_failed',
  'thumbnail_retry_clicked',

  // --- transport ---------------------------------------------------------
  /** Any non-2xx or network failure through api.ts::tracedFetch. */
  'fetch_error',
] as const;

/**
 * Canonical event name. Derived from `EVENT_TYPES` so the type and the runtime
 * registry can never disagree.
 */
export type EventType = (typeof EVENT_TYPES)[number];

const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set<string>(EVENT_TYPES);

/** True when `type` is registered in `EVENT_TYPES`. */
export function isKnownEventType(type: string): type is EventType {
  return KNOWN_EVENT_TYPES.has(type);
}

declare global {
  interface Window {
    /** Ring buffer of structured events. Populated by `log()`. */
    __log?: LogEvent[];
  }
}

/** Maximum events retained in the ring buffer before the oldest is dropped. */
const MAX_EVENTS = 500;

/**
 * Push a structured event to `window.__log`. Safe to call on the server
 * (no-op if `window` is undefined). In dev, also mirrors to `console.debug`
 * with an `[optics]` tag — and warns once-per-call on a name that is not in
 * `EVENT_TYPES`, which is how the registry stays complete: an unregistered
 * event is a doc bug, not a runtime one, so it is never rejected.
 *
 * `type` stays `string` deliberately. Narrowing it to `EventType` would make
 * adding an event a two-file change and tempt call sites into reusing a
 * near-miss name to avoid the edit.
 */
export function log(type: string, payload: Record<string, unknown> = {}): void {
  if (typeof window === 'undefined') return;
  const buf = (window.__log ||= []);
  buf.push({ t: performance.now(), type, ...payload });
  if (buf.length > MAX_EVENTS) buf.splice(0, buf.length - MAX_EVENTS);
  if (import.meta.env.DEV) {
    if (!isKnownEventType(type)) {
      // eslint-disable-next-line no-console
      console.warn('[optics] event type not in EVENT_TYPES (logger.ts):', type);
    }
    // eslint-disable-next-line no-console
    console.debug('[optics]', type, payload);
  }
}
