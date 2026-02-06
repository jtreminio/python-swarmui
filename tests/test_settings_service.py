from __future__ import annotations

from swarmui_clone.services.settings_service import SettingsService, resolve_settings_path


def test_resolve_settings_path_prefers_primary_config_yaml(tmp_path):
    primary = tmp_path / "config.yaml"
    legacy_dir = tmp_path / "config"
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / "settings.yaml"

    primary.write_text("version: 1\n", encoding="utf-8")
    legacy.write_text("version: 1\n", encoding="utf-8")

    assert resolve_settings_path(tmp_path) == primary


def test_resolve_settings_path_falls_back_to_legacy_settings_yaml(tmp_path):
    legacy_dir = tmp_path / "config"
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / "settings.yaml"
    legacy.write_text("version: 1\n", encoding="utf-8")

    assert resolve_settings_path(tmp_path) == legacy


def test_settings_service_creates_new_primary_config_file(tmp_path):
    primary = tmp_path / "config.yaml"
    settings = SettingsService(primary)

    cfg = settings.load()

    assert primary.exists()
    assert cfg.server.host == "0.0.0.0"
