from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"


def main() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    from swarmui_clone.config import app_root
    from swarmui_clone.services.settings_service import SettingsService, resolve_settings_path
    from swarmui_clone.utils.logging_utils import configure_terminal_logging

    os.environ.setdefault("SWARMUI_APP_ROOT", str(PROJECT_ROOT))
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


if __name__ == "__main__":
    main()
