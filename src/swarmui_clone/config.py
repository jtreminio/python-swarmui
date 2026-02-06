from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    swarmui_root: str = "/Volumes/swarmui"
    model_roots: list[str] = Field(default_factory=lambda: ["/Volumes/models"])
    wildcards_root: str = "/Volumes/aiconfigs/Wildcards"
    data_root: str = "data"
    output_root: str = "data/output"
    sd_model_folders: list[str] = Field(default_factory=lambda: ["Stable-Diffusion", "checkpoints"])
    lora_folders: list[str] = Field(default_factory=lambda: ["Lora", "loras", "LyCORIS"])
    vae_folders: list[str] = Field(default_factory=lambda: ["VAE", "vae"])
    embedding_folders: list[str] = Field(default_factory=lambda: ["Embeddings", "embeddings"])
    controlnet_folders: list[str] = Field(default_factory=lambda: ["controlnet", "model_patches", "ControlNet"])
    clip_folders: list[str] = Field(default_factory=lambda: ["text_encoders", "clip", "CLIP"])
    clip_vision_folders: list[str] = Field(default_factory=lambda: ["clip_vision"])


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7801
    cors_allow_origin: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ComfyConfig(BaseModel):
    start_script: str = "/Volumes/swarmui/dlbackend/ComfyUI/main.py"
    gpu_id: str = "0"
    extra_args: str = ""
    disable_internal_args: bool = False
    preview_method: Literal["disabled", "latent2rgb", "taesd"] = "latent2rgb"
    frontend_version: Literal["latest", "none", "latest_swarm_validated", "legacy"] = (
        "latest_swarm_validated"
    )
    swarm_validated_frontend_version: str = "1.37.11"
    auto_restart: bool = True
    auto_start_on_boot: bool = True
    startup_timeout_seconds: int = 180
    backend_starting_port: int = 7820
    randomize_backend_port: bool = False


class GenerationConfig(BaseModel):
    max_concurrent_jobs: int = 1
    default_model: str = ""
    default_steps: int = 30
    default_cfg_scale: float = 7.0
    default_width: int = 1024
    default_height: int = 1024
    default_sampler: str = "euler"
    default_scheduler: str = "normal"
    output_subdir_format: str = "%Y-%m-%d"


class AppConfig(BaseModel):
    version: int = 1
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    comfy: ComfyConfig = Field(default_factory=ComfyConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return app_root() / path
