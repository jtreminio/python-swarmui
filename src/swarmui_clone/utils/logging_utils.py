from __future__ import annotations

import logging
import sys
from typing import Literal

LevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def level_from_name(name: str | LevelName) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)


def configure_terminal_logging(level: int | str, force: bool = False) -> None:
    resolved = level_from_name(level) if isinstance(level, str) else level
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=force,
    )

    logging.captureWarnings(True)

    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "swarmui_clone",
    ):
        logging.getLogger(logger_name).setLevel(resolved)

    if resolved <= logging.DEBUG:
        for logger_name in (
            "httpx",
            "httpcore",
            "asyncio",
            "watchfiles",
        ):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
