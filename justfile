# justfile — Windows-friendly task runner (replaces Makefile).
# Install:       winget install Casey.Just
# List targets:  just

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set shell := ["bash", "-c"]

# Default: list recipes
default:
    @just --list

# Run backend + frontend concurrently (Ctrl+C stops both)
dev:
    pnpm dlx concurrently -k -n BE,FE -c blue,green "just backend" "just frontend"

# FastAPI on :8765 with auto-reload
backend:
    cd backend; uv run uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload

# Vite on :5173
frontend:
    cd frontend; pnpm dev

# Materialize default pattern variants to backend/data/
seed:
    cd backend; uv run python -c "from app.service import seed_defaults; seed_defaults()"

# Run as 5 SEQUENTIAL chunks, never the whole suite in one process: each
# chunk peaks ~1 GB and the 13.7 GB host has bugchecked under heavy parallel
# compute (see CLAUDE.md). Chunk order keeps the heavy files
# (plates_and_boxes, api_patterns, frames, patterns_roundtrip) paired with at
# most one other file; chunk 2 is six light/fast files.
# Chunk 5 (sim2d + showcase patterns + bitmap halftone) is synthetic/small-
# extent and runs in ~2 s.
# Backend pytest (Layer 1) — 5 sequential memory-safe chunks
test-backend flags="":
    cd backend; uv run --extra dev pytest tests/test_assembly.py tests/test_plates_and_boxes.py -q {{flags}}
    cd backend; uv run --extra dev pytest tests/test_motifs.py tests/test_rasterize.py tests/test_export_svg.py tests/test_export_gds_stub.py tests/test_theme_metadata.py tests/test_variant_hash.py -q {{flags}}
    cd backend; uv run --extra dev pytest tests/test_api_patterns.py tests/test_frames.py -q {{flags}}
    cd backend; uv run --extra dev pytest tests/test_patterns_roundtrip.py tests/test_sim_numerics.py -q {{flags}}
    cd backend; uv run --extra dev pytest tests/test_sim2d.py tests/test_pattern_types.py tests/test_showcase_patterns.py tests/test_bitmap_halftone.py -q {{flags}}

# Frontend vitest (Layer 2)
test-unit:
    cd frontend; pnpm test:unit

# Playwright E2E (Layer 3 — spins up both servers itself)
test-e2e:
    cd frontend; pnpm test:e2e

# Visual-signature capture (Layer 2 of the visual harness).
# Runs only the @visual-tagged spec; writes PNGs + meta.json per
# (slug, scene) under frontend/test-results/visual/latest/.
# No LLM calls, safe to run on every commit.
test-visual:
    cd frontend; pnpm test:e2e --grep "@visual"

# Physical-honesty effects suite: pixel-metric verification that every
# renderer effect is view-dependent, time-invariant, texture-driven, and
# obeys the substrate physics (thickness/n scaling). Dumps PNG frame
# sequences + metrics under frontend/test-results/visual/latest/effects/.
# Deterministic, no LLM. HEAVY (Playwright, spins up both servers) — never
# run alongside another compute process.
test-effects:
    cd frontend; pnpm test:e2e --grep "@effects"

# Effects capture + Claude vision grading of the frame sequences.
# Requires ANTHROPIC_API_KEY.
test-effects-verify: test-effects
    uv run --directory backend --extra dev python ../tools/visual_verifier.py --run-dir "{{justfile_directory()}}/frontend/test-results/visual/latest/effects"

# Visual-signature capture + Claude vision grading (Layer 3 of the harness).
# Requires ANTHROPIC_API_KEY in the environment. Writes
# <scene>.verdict.json files and an aggregate visualReport.md.
# NOTE: --run-dir is an absolute path because `uv run --directory backend`
# sets cwd=backend for the Python process, which would otherwise mis-resolve
# a relative frontend/... path.
test-visual-verify: test-visual
    uv run --directory backend --extra dev python ../tools/visual_verifier.py --run-dir "{{justfile_directory()}}/frontend/test-results/visual/latest"

# All three test layers, sequentially
test-all: test-backend test-unit test-e2e

# Backend runs the same safe chunk order as test-backend, with --maxfail=1
# inside each chunk.
# CI variant — fail-fast, machine-readable reporters
test-ci: (test-backend "--maxfail=1")
    cd frontend; pnpm test:unit --reporter=verbose
    cd frontend; $env:CI='1'; pnpm test:e2e --reporter=dot

# Wipe generated pattern cache (regenerate via `just seed`)
clean:
    Remove-Item -Recurse -Force backend/data -ErrorAction SilentlyContinue
