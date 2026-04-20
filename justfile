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

# Backend pytest (Layer 1)
test-backend:
    cd backend; uv run --extra dev pytest -q

# Frontend vitest (Layer 2)
test-unit:
    cd frontend; pnpm test:unit

# Playwright E2E (Layer 3 — spins up both servers itself)
test-e2e:
    cd frontend; pnpm test:e2e

# All three test layers, sequentially
test-all: test-backend test-unit test-e2e

# CI variant — fail-fast, machine-readable reporters
test-ci:
    cd backend; uv run --extra dev pytest -q --maxfail=1
    cd frontend; pnpm test:unit --reporter=verbose
    cd frontend; $env:CI='1'; pnpm test:e2e --reporter=dot

# Wipe generated pattern cache (regenerate via `just seed`)
clean:
    Remove-Item -Recurse -Force backend/data -ErrorAction SilentlyContinue
