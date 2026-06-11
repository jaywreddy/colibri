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
 * Canonical event names. Not enforced at the call site (the `log` signature
 * takes `string`) but kept here as the single source of truth for what the
 * app emits and what specs can assert against.
 */
export type EventType =
  | 'catalog_loaded'
  | 'catalog_load_failed'
  | 'box_regen_start'
  | 'box_regen_done'
  | 'box_regen_failed'
  | 'box_regen_stale'
  | 'box_regen_skipped_invalid'
  | 'box_saved'
  | 'box_loaded'
  | 'box_reset'
  | 'box_scene_rebuilt'
  | 'size_preset_applied'
  | 'face_clicked'
  | 'face_apply_all'
  | 'face_seed_shuffled'
  | 'thumbnail_loaded'
  | 'thumbnail_load_failed'
  | 'autorotate_toggled'
  | 'face_pattern_changed'
  | 'face_param_changed'
  | 'face_texture_bound'
  | 'face_texture_failed'
  | 'frame_param_changed'
  | 'foil_tape_changed'
  | 'lid_changed'
  | 'layout_changed'
  | 'illumination_changed'
  | 'light_moved'
  | 'laser_color_changed'
  | 'webgl_context_lost'
  | 'webgl_context_restored'
  | 'fetch_error';

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
 * with an `[optics]` tag.
 */
export function log(type: string, payload: Record<string, unknown> = {}): void {
  if (typeof window === 'undefined') return;
  const buf = (window.__log ||= []);
  buf.push({ t: performance.now(), type, ...payload });
  if (buf.length > MAX_EVENTS) buf.splice(0, buf.length - MAX_EVENTS);
  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console.debug('[optics]', type, payload);
  }
}
