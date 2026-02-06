from __future__ import annotations

from pathlib import Path

import yaml

from swarmui_clone.config import AppConfig
from swarmui_clone.utils.pathing import ensure_directory


class SettingsService:
    def __init__(self, settings_path: Path) -> None:
        self._settings_path = settings_path

    @property
    def settings_path(self) -> Path:
        return self._settings_path

    def load(self) -> AppConfig:
        if not self._settings_path.exists():
            config = AppConfig()
            self.save(config)
            return config
        raw = yaml.safe_load(self._settings_path.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(raw)

    def save(self, config: AppConfig) -> None:
        ensure_directory(self._settings_path.parent)
        payload = config.model_dump(mode="json")
        self._settings_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
