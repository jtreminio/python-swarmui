from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str
    model_architecture: Literal["auto", "checkpoint", "flux"] = "auto"
    steps: int = Field(default=30, ge=1, le=200)
    cfg_scale: float = Field(default=7.0, gt=0.0, le=50.0)
    seed: int = -1
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    sampler_name: str = "euler"
    scheduler: str = "normal"
    denoise: float = Field(default=1.0, gt=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=16)
    filename_prefix: str = "swarmui"
    flux_clip_name1: str = ""
    flux_clip_name2: str = ""
    flux_vae_name: str = ""
    flux_guidance: float | None = Field(default=None, gt=0.0, le=100.0)
    custom_workflow: dict[str, Any] | None = None


class GenerationImage(BaseModel):
    file_name: str
    relative_path: str
    absolute_path: str


class GenerationJob(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: datetime
    updated_at: datetime
    request: GenerationRequest
    expanded_prompt: str | None = None
    expanded_negative_prompt: str | None = None
    prompt_id: str | None = None
    queue_position: int | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    images: list[GenerationImage] = Field(default_factory=list)
    wildcard_tokens: list[str] = Field(default_factory=list)
    generation_seed: int | None = None

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


class BackendStatus(BaseModel):
    status: Literal["stopped", "starting", "running", "error"]
    port: int | None = None
    pid: int | None = None
    api_url: str | None = None
    web_url: str | None = None
    start_script: str
    health_ok: bool
    auto_restart: bool
    last_error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
