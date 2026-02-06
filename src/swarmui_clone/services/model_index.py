from __future__ import annotations

from pathlib import Path

from swarmui_clone.config import resolve_path


MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".engine",
    ".gguf",
}


class ModelIndexService:
    def __init__(self, config_provider) -> None:
        self._config_provider = config_provider

    def _roots(self) -> list[Path]:
        cfg = self._config_provider()
        return [resolve_path(root) for root in cfg.paths.model_roots]

    @staticmethod
    def _is_model_file(path: Path) -> bool:
        return path.suffix.lower() in MODEL_EXTENSIONS

    def _scan_category(self, subdirs: list[str]) -> list[str]:
        found: set[str] = set()
        for root in self._roots():
            for subdir in subdirs:
                model_dir = (root / subdir).resolve()
                if not model_dir.exists():
                    continue
                for path in model_dir.rglob("*"):
                    if not path.is_file() or not self._is_model_file(path):
                        continue
                    rel = path.relative_to(model_dir).as_posix()
                    found.add(rel)
        return sorted(found)

    def scan_all(self) -> dict[str, list[str]]:
        cfg = self._config_provider()
        return {
            "Stable-Diffusion": self._scan_category(cfg.paths.sd_model_folders),
            "LoRA": self._scan_category(cfg.paths.lora_folders),
            "VAE": self._scan_category(cfg.paths.vae_folders),
            "Embedding": self._scan_category(cfg.paths.embedding_folders),
            "ControlNet": self._scan_category(cfg.paths.controlnet_folders),
            "CLIP": self._scan_category(cfg.paths.clip_folders),
            "CLIPVision": self._scan_category(cfg.paths.clip_vision_folders),
        }
