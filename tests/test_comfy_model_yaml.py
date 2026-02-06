from __future__ import annotations

import yaml

from swarmui_clone.config import AppConfig
from swarmui_clone.services.comfy_launcher import ComfyProcessManager


def test_emit_model_paths_config(tmp_path):
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir(parents=True)
    (comfy_dir / "main.py").write_text("print('stub')\n", encoding="utf-8")

    model_root = tmp_path / "models"
    model_root.mkdir()

    cfg = AppConfig()
    cfg.paths.data_root = str(tmp_path / "data")
    cfg.paths.model_roots = [str(model_root)]
    cfg.comfy.start_script = str(comfy_dir / "main.py")

    manager = ComfyProcessManager(lambda: cfg)
    output = manager.emit_model_paths_config()

    assert output.exists()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert "swarmui_model_paths_0" in payload
    assert "checkpoints" in payload["swarmui_model_paths_0"]
    assert "swarmui_nodes" in payload
    assert "custom_nodes" in payload["swarmui_nodes"]

    clip_dir = model_root / cfg.paths.clip_folders[0]
    assert clip_dir.exists()
