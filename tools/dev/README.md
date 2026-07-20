# tools/dev

Ad-hoc developer utilities. Not part of any automated test or build.

## Headless inspection of the live app

The old `window.__debug` surface (and its cheat-sheet, `tools/preview_inspect.md`)
died with the pre-box studio UI. Current hooks, registered on every page load:

- `window.__store` — the zustand store (`frontend/src/main.tsx`); read/write
  state deterministically without fighting React-controlled inputs.
- `window.__studio` — three.js handles from `BoxScene`
  (`scene`, `camera`, `controls`, `renderer`, `faces`, `store`,
  `setLid(deg)`, `getLidDeg()`, `autoRotate`, `lidPivot`).

Render recipes (shader `uRecipe`): `0 stereo_lenticular`,
`1 moire_interactive`, `3 foliage_moire` (id 2, `phase_shift_overlay`, is
retired). Every composed box plate binds `foliage_moire`: the back gold layer
renders on a REAL inner plane at the paraxial T/n air gap, so substrate
checks manipulate the actual plane gap
(`frontend/tests/e2e/effectsHelpers.ts::scaleBackPlaneGap`) — the legacy
`uThicknessUm` uniform is dead on the foliage path and poking it proves
nothing.

For scripted screenshots use `frontend/scripts/shot.mjs` /
`moireshot.mjs`; for automated verification use `just test-visual` /
`just test-effects` (see the justfile).

## `shot_server.py`

Tiny HTTP collector for canvas captures during live Claude Code sessions.

Use when:
- You're inspecting the app via the MCP preview window, and
- `preview_screenshot` (CDP `Page.captureScreenshot`) times out because the preview tab is minimized/hidden.

Workflow:

```powershell
# Terminal 1: start the collector (listens on 127.0.0.1:9999)
python tools/dev/shot_server.py

# Terminal 2 (or Claude Code tool call): POST a canvas capture
# preview_eval:
#   const c = document.createElement('canvas');
#   c.width = img.naturalWidth; c.height = img.naturalHeight;
#   c.getContext('2d').drawImage(img, 0, 0);
#   fetch('http://127.0.0.1:9999/my-shot', {
#     method: 'POST',
#     body: c.toDataURL('image/jpeg', 0.92)
#   });

# Read the JPEG with the Read tool (LLM sees the image directly):
#   Read: %TEMP%/optics-shots/my-shot.jpg
```

Output directory defaults to `%TEMP%/optics-shots`; override with `OPTICS_SHOT_DIR`.

**For automated visual verification, use `just test-visual` instead** — that pipeline uses `captureScene()` in `frontend/tests/e2e/helpers.ts` and needs no external collector.
