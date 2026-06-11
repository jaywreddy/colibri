from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import boxes as boxes_api
from .api import export as export_api
from .api import patterns as patterns_api
from .api import plates as plates_api
from .api import sim as sim_api
from .service import DATA_ROOT

log = logging.getLogger("optics")
http_log = logging.getLogger("optics.http")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No eager seeding: materializing every pattern default took minutes of
    # cold startup and nothing requires it — GET /patterns serves class
    # descriptors with zero materialization, and /patterns/generate +
    # /boxes/generate materialize variants lazily with a disk cache.
    # To pre-warm explicitly, run `just seed` (service.seed_defaults).
    log.info("Startup complete — pattern variants materialize lazily into %s", DATA_ROOT)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Optics Pattern Studio",
        description="Backend for dual-layer gold-on-quartz pattern design and preview.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Stamp every HTTP response with a short request_id and duration, and
        mirror to the ``optics.http`` logger at a level tied to the status."""
        request_id = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 — ensure 500s are logged too
            dt_ms = int((time.perf_counter() - t0) * 1000)
            http_log.exception(
                "req=%s %s %s -> 500 (%dms)",
                request_id,
                request.method,
                request.url.path,
                dt_ms,
            )
            raise
        dt_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-ID"] = request_id
        if response.status_code >= 500:
            level = logging.ERROR
        elif response.status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        http_log.log(
            level,
            "req=%s %s %s -> %d (%dms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            dt_ms,
        )
        return response

    app.include_router(patterns_api.router)
    app.include_router(sim_api.router)
    app.include_router(plates_api.router)
    app.include_router(boxes_api.router)
    app.include_router(export_api.router)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/data", StaticFiles(directory=str(DATA_ROOT)), name="data")

    @app.get("/")
    def root() -> dict:
        return {"ok": True, "name": "optics-backend", "version": "0.1.0"}

    return app


app = create_app()
