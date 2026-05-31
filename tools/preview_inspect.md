# Headless plot inspection — cheat-sheet for Claude

How to drive the optics studio frontend through the Claude_Preview MCP
(`mcp__Claude_Preview__preview_*`). Read this file when you land in the repo
and need to look at a live plot, debug a recipe, or compare two scenes —
it's faster than rediscovering the surface.

## Setup

Dev server on `:5173` is required (`just frontend`, `just backend`, or
let Playwright auto-start). The `window.__debug` surface registers itself
on every page load under `import.meta.env.DEV`. Source: `frontend/src/debug.ts`.

```
preview_start  url: http://localhost:5173/
```

## URL deep-links — preferred entry point

Land directly on the scene you want; no clicks needed.

| Param      | Example               | Notes |
|------------|-----------------------|-------|
| `pattern`  | `colibri-hologram`    | Slug from `/patterns`. Falls back to first pattern if unknown. |
| `illum`    | `ambient`/`laser`/`backlight` | Illumination mode. |
| `laser`    | `red`/`green`/`blue`  | Laser color. Only meaningful when `illum=laser`. |
| `z`        | `0.5`                 | Carpet z-slice, normalized 0..1. |
| `cam`      | `45,30`               | OrbitControls azimuth,elevation in degrees. El=0 = top-down. |
| `light`    | `120,40`              | Illumination direction az,el. |
| `param.X`  | `param.grid=512`      | Pattern parameter override. |

```
http://localhost:5173/?pattern=colibri-hologram&illum=laser&laser=red&cam=45,30
```

## Typical debug loop

```js
// 1. Land on the scene
preview_eval: await window.__debug.applyScene({ pattern: 'tairona-talbot', illumination: 'ambient', zSlice: 0.5 })

// 2. See it
preview_screenshot

// 3. Inspect uniforms
preview_eval: window.__debug.readUniforms()
//   → { uRecipe: 3, uZSlice: 0.5, uHasCarpet: true, uCarpetRows: 48, ... }

// 4. Confirm energy distribution
preview_eval: window.__debug.quadrantBrightness()
//   → { ul: ..., ur: ..., ll: ..., lr: ..., total: ... }

// 5. Tweak and re-check
preview_eval: window.__debug.setCamera(60, 20); await window.__debug.waitForRender()
preview_screenshot

// 6. Read recent events if something looks off
preview_eval: window.__debug.readLog(20)
```

## Full `window.__debug` surface

Defined in `frontend/src/debug.ts`. All async methods await render
settling internally — you don't need to add manual sleeps.

### Scene control
- `applyScene(preset)` — bulk setter, awaits render. Idempotent.
- `selectPattern(slug, timeoutMs?)` — select + wait for `texture_bound`.
- `setIllumination(mode, color?)` — `'ambient' | 'laser' | 'backlight'`.
- `setCamera(az, el)` — OrbitControls placement. El=0 is straight down.
- `setLight(az, el)` — illumination direction in degrees.
- `setZSlice(z)` — 0..1, only honored on `near_field_carpet` recipes.
- `setParam(key, value)` — patches `params` (debounced regen kicks in).

### Inspection
- `readState()` — full Zustand snapshot (functions stripped).
- `readUniforms()` — every shader uniform, JSON-safe.
- `readLog(n?)` — last N events from `window.__log`.
- `clearLog()` — reset the ring buffer.
- `quadrantBrightness()` — 64×64 grid sum per quadrant of the canvas.
- `sampleMany(pts)` — atomic render + readPixels for many points.

### Capture
- `captureCanvas()` — base64 PNG dataURL of the WebGL canvas.
- `captureSecondary(testid?)` — base64 PNG of a SecondaryView `<img>`.
  Default picks `carpet-image` or `farfield-image` if present.

### Sync
- `waitForRender({ timeoutMs? })` — pending select + secondary fetch + 2 RAFs.
- `waitForLog(type, predicate?, timeoutMs?)` — poll for an event pushed
  after the call. Tail of buffer dumped on timeout.

### Misc
- `parseSceneFromUrl(href?)` — return a `ScenePreset` for the URL.
- `lastScene` — most recent preset passed to `applyScene` (for diff'ing).

## Recipes & what to expect

| Recipe (uRecipe) | Patterns | What renders on plate | What renders on SecondaryView |
|---|---|---|---|
| 0 `iridescent_grating` | cafetero-iridescence | Bragg-tinted plate | (none) |
| 1 `stereo_lenticular` | sombrero-vueltiao-parallax, caravel-latent | View-A/View-B blend | (none) |
| 2 `moire_interactive` | wayuu, emerald-facet, compass-rose | Front+back composite, parallax | (none) |
| 3 `near_field_carpet` | tairona-talbot, muzo-emerald-zone | Z-slice of carpet atlas | Full carpet PNG |
| 4 `far_field_hologram` | colibri-hologram, meridian-speckle | Bare amplitude mask | RGB Fraunhofer reconstruction |
| 5 `stylized_amplitude` | (legacy fallback) | Generic stylized blend | (none) |

## Notes / gotchas

- The plate is rendered every RAF. After `setCamera` etc., one or two RAFs
  is enough; `waitForRender` handles that. Don't add extra `setTimeout`s.
- `captureCanvas` re-renders before `toDataURL` — safe to call without an
  explicit render first.
- Recipe 4 keeps the plate showing the bare mask. The "what you'd project"
  visualization lives in SecondaryView. Use `captureSecondary('farfield-image')`
  for it.
- WebGL context can drop in headless Chromium; `texture_bound` events are
  the cleanest sync signal. The e2e fixtures handle context loss; debug
  sessions usually don't see it.
- This file is for me, not for users. Keep the sentence-fragments style.
