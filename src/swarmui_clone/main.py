from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from swarmui_clone.api.routes import build_compat_router, build_router
from swarmui_clone.app_state import AppState

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, state.get_config().server.log_level, logging.INFO))
    await state.startup()
    yield
    await state.shutdown()


app = FastAPI(title="Python SwarmUI Clone", version="0.1.0", lifespan=lifespan)

if state.get_config().server.cors_allow_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[state.get_config().server.cors_allow_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)

app.include_router(build_router(state), prefix="/api")
app.include_router(build_compat_router(state))


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/api/")
