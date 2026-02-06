from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pytest

from swarmui_clone.api.routes import (
    _build_generation_request_from_swarm,
    _compat_describe_model,
    _compat_forward_metadata_request,
    _compat_get_model_hash,
    _compat_list_loaded_models,
    _compat_list_models,
    _compat_select_model,
)
from swarmui_clone.config import AppConfig
from swarmui_clone.schemas import GenerationJob, GenerationRequest


@dataclass
class StubModelIndex:
    values: dict[str, list[str]]

    def scan_all(self) -> dict[str, list[str]]:
        return self.values


@dataclass
class StubWildcards:
    values: list[str]

    def list_wildcards(self) -> list[str]:
        return self.values


class StubState:
    def __init__(self, cfg: AppConfig, models: dict[str, list[str]], wildcards: list[str]) -> None:
        self._cfg = cfg
        self.model_index = StubModelIndex(models)
        self.wildcards = StubWildcards(wildcards)
        self.generation = StubGeneration([])

    def get_config(self) -> AppConfig:
        return self._cfg

    def update_config(self, cfg: AppConfig) -> AppConfig:
        self._cfg = cfg
        return self._cfg


class StubGeneration:
    def __init__(self, jobs: list[GenerationJob]) -> None:
        self._jobs = jobs

    async def list_jobs(self, limit: int = 100) -> list[GenerationJob]:
        return self._jobs[:limit]


def test_build_generation_request_from_swarm_maps_aliases():
    cfg = AppConfig()
    state = StubState(
        cfg,
        {"Stable-Diffusion": ["base/model.safetensors"]},
        [],
    )

    payload = {
        "images": "3",
        "prompt": "test prompt",
        "negativeprompt": "bad",
        "model": "base/model.safetensors",
        "steps": "40",
        "cfgscale": "6.5",
        "seed": "99",
        "width": "1152",
        "height": "896",
        "sampler": "euler_ancestral",
        "scheduler": "karras",
        "batchsize": "2",
    }

    images, request = _build_generation_request_from_swarm(state, payload)

    assert images == 3
    assert request.prompt == "test prompt"
    assert request.negative_prompt == "bad"
    assert request.model == "base/model.safetensors"
    assert request.steps == 40
    assert request.cfg_scale == 6.5
    assert request.seed == 99
    assert request.width == 1152
    assert request.height == 896
    assert request.sampler_name == "euler_ancestral"
    assert request.scheduler == "karras"
    assert request.batch_size == 2


def test_build_generation_request_from_swarm_maps_flux_fields():
    cfg = AppConfig()
    state = StubState(
        cfg,
        {"Stable-Diffusion": ["flux1-dev.safetensors"]},
        [],
    )

    payload = {
        "prompt": "test prompt",
        "model": "flux1-dev.safetensors",
        "model_architecture": "flux",
        "clip_name1": "clip_l.safetensors",
        "clip_name2": "t5xxl_fp16.safetensors",
        "vae": "ae.safetensors",
        "guidance": "3.5",
    }

    _, request = _build_generation_request_from_swarm(state, payload)

    assert request.model_architecture == "flux"
    assert request.flux_clip_name1 == "clip_l.safetensors"
    assert request.flux_clip_name2 == "t5xxl_fp16.safetensors"
    assert request.flux_vae_name == "ae.safetensors"
    assert request.flux_guidance == 3.5


def test_build_generation_request_from_swarm_falls_back_to_first_model():
    cfg = AppConfig()
    state = StubState(
        cfg,
        {"Stable-Diffusion": ["fallback/model.safetensors"]},
        [],
    )

    payload = {
        "images": 1,
        "prompt": "hello",
    }

    images, request = _build_generation_request_from_swarm(state, payload)

    assert images == 1
    assert request.model == "fallback/model.safetensors"


def test_compat_list_models_filters_path_and_depth():
    cfg = AppConfig()
    state = StubState(
        cfg,
        {
            "Stable-Diffusion": [
                "root.safetensors",
                "folder/inner.safetensors",
                "folder/deep/model.safetensors",
            ]
        },
        [],
    )

    payload = {
        "subtype": "Stable-Diffusion",
        "path": "folder",
        "depth": 1,
    }
    response = _compat_list_models(state, payload)

    names = [entry["name"] for entry in response["files"]]
    assert names == ["folder/inner.safetensors"]


@pytest.mark.asyncio
async def test_compat_select_model_updates_default_model():
    cfg = AppConfig()
    state = StubState(
        cfg,
        {"Stable-Diffusion": ["base/model-a.safetensors", "base/model-b.safetensors"]},
        [],
    )

    result = await _compat_select_model(state, {"model": "base/model-b.safetensors"})

    assert result["success"] is True
    assert state.get_config().generation.default_model == "base/model-b.safetensors"


@pytest.mark.asyncio
async def test_compat_select_model_rejects_unknown_model():
    cfg = AppConfig()
    state = StubState(cfg, {"Stable-Diffusion": ["base/model-a.safetensors"]}, [])

    result = await _compat_select_model(state, {"model": "base/missing.safetensors"})

    assert result == {"error": "Model not found."}


@pytest.mark.asyncio
async def test_compat_list_loaded_models_uses_default_and_running_jobs():
    cfg = AppConfig()
    cfg.generation.default_model = "base/default.safetensors"
    state = StubState(cfg, {"Stable-Diffusion": []}, [])
    state.generation = StubGeneration(
        [
            GenerationJob(
                id="job-running",
                status="running",
                created_at=GenerationJob.now(),
                updated_at=GenerationJob.now(),
                request=GenerationRequest(prompt="p", model="base/running.safetensors"),
            ),
            GenerationJob(
                id="job-queued",
                status="queued",
                created_at=GenerationJob.now(),
                updated_at=GenerationJob.now(),
                request=GenerationRequest(prompt="p", model="base/queued.safetensors"),
            ),
            GenerationJob(
                id="job-succeeded",
                status="succeeded",
                created_at=GenerationJob.now(),
                updated_at=GenerationJob.now(),
                request=GenerationRequest(prompt="p", model="base/succeeded.safetensors"),
            ),
        ]
    )

    result = await _compat_list_loaded_models(state)
    names = [entry["name"] for entry in result["models"]]

    assert names == [
        "base/default.safetensors",
        "base/running.safetensors",
        "base/succeeded.safetensors",
    ]


@pytest.mark.asyncio
async def test_compat_describe_model_finds_model_and_returns_metadata(tmp_path):
    cfg = AppConfig()
    model_root = tmp_path / "models"
    model_dir = model_root / "Stable-Diffusion"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "demo-model.safetensors"
    model_file.write_bytes(b"weights")

    cfg.paths.model_roots = [str(model_root)]
    state = StubState(cfg, {"Stable-Diffusion": ["demo-model.safetensors"]}, [])

    result = await _compat_describe_model(
        state,
        {"modelName": "demo-model", "subtype": "Stable-Diffusion"},
    )

    assert "model" in result
    assert result["model"]["name"] == "demo-model.safetensors"
    assert result["model"]["local"] is True
    assert result["model"]["is_supported_model_format"] is True


@pytest.mark.asyncio
async def test_compat_get_model_hash_matches_sha256(tmp_path):
    cfg = AppConfig()
    model_root = tmp_path / "models"
    model_dir = model_root / "Stable-Diffusion"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "hash-me.safetensors"
    model_file.write_bytes(b"abc123")

    cfg.paths.model_roots = [str(model_root)]
    state = StubState(cfg, {"Stable-Diffusion": ["hash-me.safetensors"]}, [])

    result = await _compat_get_model_hash(
        state,
        {"modelName": "hash-me", "subtype": "Stable-Diffusion"},
    )

    expected = hashlib.sha256(b"abc123").hexdigest()
    assert result == {"hash": expected}


@pytest.mark.asyncio
async def test_compat_forward_metadata_request_rejects_non_civitai():
    cfg = AppConfig()
    state = StubState(cfg, {"Stable-Diffusion": []}, [])

    result = await _compat_forward_metadata_request(
        state,
        {"url": "https://example.com/api"},
    )

    assert result == {"error": "Invalid URL."}
