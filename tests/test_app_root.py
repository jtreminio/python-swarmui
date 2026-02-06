from __future__ import annotations

from pathlib import Path

from swarmui_clone.config import app_root


def test_app_root_prefers_environment_override(monkeypatch, tmp_path):
    custom_root = tmp_path / "custom-root"
    custom_root.mkdir(parents=True)
    monkeypatch.setenv("SWARMUI_APP_ROOT", str(custom_root))
    assert app_root() == custom_root.resolve()


def test_app_root_uses_project_like_cwd(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    (project_root / "run.py").write_text("print('stub')\n", encoding="utf-8")
    (project_root / "config").mkdir()

    monkeypatch.delenv("SWARMUI_APP_ROOT", raising=False)
    monkeypatch.chdir(project_root)

    assert app_root() == project_root.resolve()


def test_app_root_falls_back_to_package_location(monkeypatch):
    monkeypatch.delenv("SWARMUI_APP_ROOT", raising=False)
    package_root = Path(__file__).resolve().parents[1]
    assert app_root() == package_root.resolve()
