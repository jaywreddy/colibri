from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import patterns as patterns_api
from .api import sim as sim_api
from .service import DATA_ROOT, seed_defaults

log = logging.getLogger("optics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Seeding default pattern variants...")
    manifests = seed_defaults()
    log.info("Seeded %d patterns into %s", len(manifests), DATA_ROOT)
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

    app.include_router(patterns_api.router)
    app.include_router(sim_api.router)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/data", StaticFiles(directory=str(DATA_ROOT)), name="data")

    @app.get("/")
    def root() -> dict:
        return {"ok": True, "name": "optics-backend", "version": "0.1.0"}

    return app


app = create_app()
