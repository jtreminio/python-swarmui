from __future__ import annotations

import logging
import os

import uvicorn

from swarmui_clone.config import app_root
from swarmui_clone.services.settings_service import SettingsService, resolve_settings_path
from swarmui_clone.utils.logging_utils import configure_terminal_logging

if __name__ == "__main__":
    settings_path = resolve_settings_path(app_root())
    config = SettingsService(settings_path).load()

    os.environ["SWARMUI_FORCE_DEBUG_LOGGING"] = "1"
    configure_terminal_logging(logging.DEBUG, force=True)
    logging.getLogger("swarmui_clone.run").debug(
        "Starting python-swarmui with forced DEBUG logging (config=%s, host=%s, port=%s)",
        settings_path,
        config.server.host,
        config.server.port,
    )
    uvicorn.run(
        "swarmui_clone.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
        log_level="debug",
        access_log=True,
    )
