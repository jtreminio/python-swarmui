from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from swarmui_clone.api.routes import build_compat_router, build_router
from swarmui_clone.app_state import AppState
from swarmui_clone.utils.logging_utils import configure_terminal_logging

state = AppState()
LOGGER = logging.getLogger("swarmui_clone.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    force_debug = os.environ.get("SWARMUI_FORCE_DEBUG_LOGGING", "0") == "1"
    configured_level = "DEBUG" if force_debug else state.get_config().server.log_level
    configure_terminal_logging(configured_level, force=force_debug)
    LOGGER.debug(
        "Application startup (force_debug=%s, configured_level=%s)",
        force_debug,
        configured_level,
    )
    await state.startup()
    LOGGER.debug("Startup complete")
    yield
    LOGGER.debug("Application shutdown starting")
    await state.shutdown()
    LOGGER.debug("Shutdown complete")


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
