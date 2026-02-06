from __future__ import annotations

import pytest

from swarmui_clone.config import AppConfig
from swarmui_clone.services.comfy_launcher import ComfyProcessManager


@pytest.mark.asyncio
async def test_preflight_detects_missing_module(tmp_path):
    script = tmp_path / "main.py"
    script.write_text("import this_module_should_not_exist_abc123\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.comfy.start_script = str(script)

    manager = ComfyProcessManager(lambda: cfg)
    result = await manager.preflight_check(timeout_seconds=5)

    assert result["ok"] is False
    assert "this_module_should_not_exist_abc123" in result["missing_modules"]
    assert manager.last_preflight() is not None


@pytest.mark.asyncio
async def test_preflight_passes_for_valid_script(tmp_path):
    script = tmp_path / "main.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.comfy.start_script = str(script)

    manager = ComfyProcessManager(lambda: cfg)
    result = await manager.preflight_check(timeout_seconds=5)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_preflight_import_probe_catches_module_missing_outside_help_path(tmp_path):
    script = tmp_path / "main.py"
    script.write_text(
        "import argparse\n"
        "import sys\n"
        "if '--help' not in sys.argv:\n"
        "    import not_installed_for_probe_abc123\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.comfy.start_script = str(script)

    manager = ComfyProcessManager(lambda: cfg)
    result = await manager.preflight_check(timeout_seconds=5)

    assert result["ok"] is False
    assert "not_installed_for_probe_abc123" in result["missing_modules"]
