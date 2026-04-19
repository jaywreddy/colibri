.PHONY: help dev backend frontend seed test test-backend test-frontend-unit test-e2e test-all test-all-ci clean

help:
	@echo "targets:"
	@echo "  dev              - run backend + frontend concurrently"
	@echo "  backend          - run FastAPI on :8765"
	@echo "  frontend         - run Vite on :5173"
	@echo "  seed             - materialize default patterns to backend/data"
	@echo "  test             - alias for test-backend"
	@echo "  test-backend     - pytest (Layer 1)"
	@echo "  test-frontend-unit - vitest (Layer 2)"
	@echo "  test-e2e         - playwright (Layer 3, spins up both servers)"
	@echo "  test-all         - all three layers"
	@echo "  clean            - remove generated pattern cache"

dev:
	@$(MAKE) -j2 backend frontend

backend:
	cd backend && uv run uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload

frontend:
	cd frontend && pnpm dev

seed:
	cd backend && uv run python -c "from app.service import seed_defaults; seed_defaults()"

test: test-backend

test-backend:
	cd backend && uv run --extra dev pytest -q

test-frontend-unit:
	cd frontend && pnpm test:unit

test-e2e:
	cd frontend && pnpm test:e2e

test-all: test-backend test-frontend-unit test-e2e

test-all-ci:
	cd backend && uv run --extra dev pytest -q --maxfail=1
	cd frontend && pnpm test:unit --reporter=verbose
	cd frontend && CI=1 pnpm test:e2e --reporter=dot

clean:
	rm -rf backend/data
