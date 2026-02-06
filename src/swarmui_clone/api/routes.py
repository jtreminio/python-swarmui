from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from swarmui_clone.app_state import AppState
from swarmui_clone.config import AppConfig, app_root, resolve_path
from swarmui_clone.schemas import GenerationRequest
from swarmui_clone.services.model_index import MODEL_EXTENSIONS
from swarmui_clone.utils.pathing import safe_relative_path

LOGGER = logging.getLogger("swarmui_clone.api.compat")


def _resolve_template_file(template_name: str) -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "templates" / template_name,
        app_root() / "src" / "swarmui_clone" / "templates" / template_name,
        app_root() / "swarmui_clone" / "templates" / template_name,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    return candidates[0]

def _resolve_output_file(state: AppState, image_path: str) -> Path:
    output_root = resolve_path(state.get_config().paths.output_root)
    return safe_relative_path(output_root, image_path)


def _output_image_response(state: AppState, image_path: str) -> FileResponse:
    try:
        target = _resolve_output_file(state, image_path)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path=str(target))


async def _proxy_to_comfy(state: AppState, full_path: str, request: Request) -> Response:
    backend = state.comfy_manager.snapshot()
    if backend.status != "running" or not backend.api_url:
        raise HTTPException(status_code=503, detail="ComfyUI backend is not running")

    query = request.url.query
    target_url = f"{backend.api_url}/{full_path}"
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()
    passthrough_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        response = await client.request(
            request.method,
            target_url,
            content=body,
            headers=passthrough_headers,
        )

    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type"),
    )


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    value_text = str(value).strip()
    if not value_text:
        return None
    try:
        return float(value_text)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value_text = str(value).strip().lower()
    if value_text in {"1", "true", "yes", "y", "on"}:
        return True
    if value_text in {"0", "false", "no", "n", "off"}:
        return False
    return default


async def _extract_payload(request: Request) -> dict[str, Any]:
    payload: dict[str, Any] = dict(request.query_params.items())

    if request.method not in {"GET", "HEAD"}:
        content_type = request.headers.get("content-type", "").lower()
        body_data: dict[str, Any] = {}
        if "application/json" in content_type:
            body = await request.json()
            if isinstance(body, dict):
                body_data = body
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            body_data = {key: value for key, value in form.items()}
        else:
            raw = await request.body()
            if raw:
                try:
                    maybe_json = json.loads(raw.decode("utf-8"))
                    if isinstance(maybe_json, dict):
                        body_data = maybe_json
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body_data = {}
        payload.update(body_data)

    return _merge_raw_input(payload)


def _merge_raw_input(payload: dict[str, Any]) -> dict[str, Any]:
    raw_input = payload.get("rawInput")
    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                raw_input = parsed
        except json.JSONDecodeError:
            raw_input = None
    if not isinstance(raw_input, dict):
        return payload
    merged = dict(raw_input)
    for key, value in payload.items():
        if key != "rawInput":
            merged[key] = value
    return merged


async def _extract_ws_payload(websocket: WebSocket) -> dict[str, Any]:
    if websocket.query_params:
        query_payload = dict(websocket.query_params.items())
        return _merge_raw_input(query_payload)
    message = await websocket.receive_json()
    if not isinstance(message, dict):
        raise ValueError("Websocket payload must be a JSON object.")
    return _merge_raw_input(message)


def _swarm_error(message: str) -> dict[str, Any]:
    return {"error": message}


async def _safe_close_websocket(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:
        pass


def _normalize_model_name(name: str) -> str:
    clean = name.replace("\\", "/")
    while "//" in clean:
        clean = clean.replace("//", "/")
    return clean.lstrip("/").strip()


def _subtype_folders(cfg: AppConfig, subtype: str) -> list[str] | None:
    mapping: dict[str, list[str]] = {
        "Stable-Diffusion": cfg.paths.sd_model_folders,
        "LoRA": cfg.paths.lora_folders,
        "VAE": cfg.paths.vae_folders,
        "Embedding": cfg.paths.embedding_folders,
        "ControlNet": cfg.paths.controlnet_folders,
        "CLIP": cfg.paths.clip_folders,
        "CLIPVision": cfg.paths.clip_vision_folders,
    }
    return mapping.get(subtype)


def _resolve_model_identity(state: AppState, model_name: str, subtype: str) -> tuple[str | None, Path | None]:
    cfg = state.get_config()
    all_models = state.model_index.scan_all()
    if subtype not in all_models:
        return None, None

    normalized = _normalize_model_name(model_name)
    candidates = [normalized]
    if normalized and not normalized.lower().endswith(".safetensors"):
        candidates.insert(0, f"{normalized}.safetensors")

    selected_name: str | None = None
    for candidate in candidates:
        if candidate in all_models[subtype]:
            selected_name = candidate
            break
    if selected_name is None:
        return None, None

    folders = _subtype_folders(cfg, subtype) or []
    for root_text in cfg.paths.model_roots:
        root = resolve_path(root_text)
        for folder in folders:
            candidate_path = (root / folder / selected_name).resolve()
            if candidate_path.exists() and candidate_path.is_file():
                return selected_name, candidate_path

    if Path(selected_name).is_absolute():
        absolute = Path(selected_name).resolve()
        if absolute.exists() and absolute.is_file():
            return selected_name, absolute

    return selected_name, None


def _model_net_object(state: AppState, model_name: str, subtype: str, model_path: Path | None) -> dict[str, Any]:
    cfg = state.get_config()
    loaded = cfg.generation.default_model.strip() == model_name
    title = Path(model_name).name
    is_supported = Path(model_name).suffix.lower() in MODEL_EXTENSIONS

    date_text = ""
    if model_path and model_path.exists():
        date_text = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()

    return {
        "name": model_name,
        "title": title,
        "author": "",
        "description": "",
        "preview_image": None,
        "loaded": loaded,
        "architecture": subtype,
        "class": subtype,
        "compat_class": subtype,
        "standard_width": cfg.generation.default_width,
        "standard_height": cfg.generation.default_height,
        "license": "",
        "date": date_text,
        "usage_hint": "",
        "trigger_phrase": "",
        "merged_from": "",
        "tags": [],
        "is_supported_model_format": is_supported,
        "is_negative_embedding": False,
        "local": model_path is not None,
    }


def _hash_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _describe_wildcard(state: AppState, model_name: str) -> dict[str, Any] | None:
    normalized = _normalize_model_name(model_name)
    wildcard_names = state.wildcards.list_wildcards()
    if normalized not in wildcard_names:
        return None

    wildcard_root = resolve_path(state.get_config().paths.wildcards_root)
    wildcard_file = (wildcard_root / f"{normalized}.txt").resolve()
    description = ""
    if wildcard_file.exists() and wildcard_file.is_file():
        try:
            lines = wildcard_file.read_text(encoding="utf-8").splitlines()
            non_empty = [line.strip() for line in lines if line.strip()]
            if non_empty:
                description = non_empty[0][:200]
        except Exception:
            description = ""

    return {
        "name": normalized,
        "title": Path(normalized).name,
        "description": description,
        "local": True,
        "type": "wildcard",
    }


async def _compat_describe_model(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    subtype = str(payload.get("subtype", "Stable-Diffusion"))
    model_name = str(payload.get("modelName", payload.get("model", ""))).strip()
    if not model_name:
        return _swarm_error("Model not found.")

    if subtype == "Wildcards":
        wildcard = _describe_wildcard(state, model_name)
        if wildcard is None:
            LOGGER.debug("DescribeModel wildcard not found: %s", model_name)
            return _swarm_error("Model not found.")
        return wildcard

    if subtype not in state.model_index.scan_all():
        return _swarm_error("Invalid sub-type.")

    if subtype == "Stable-Diffusion" and model_name.lower() == "(none)":
        none_model = _model_net_object(state, "(None)", subtype, None)
        none_model["description"] = "No model selected."
        return {"model": none_model}

    resolved_name, resolved_path = _resolve_model_identity(state, model_name, subtype)
    if not resolved_name:
        LOGGER.debug("DescribeModel not found: subtype=%s model=%s", subtype, model_name)
        return _swarm_error("Model not found.")

    return {"model": _model_net_object(state, resolved_name, subtype, resolved_path)}


async def _compat_get_model_hash(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    subtype = str(payload.get("subtype", "Stable-Diffusion"))
    if subtype not in state.model_index.scan_all():
        return _swarm_error("Invalid sub-type.")

    model_name = str(payload.get("modelName", payload.get("model", ""))).strip()
    resolved_name, resolved_path = _resolve_model_identity(state, model_name, subtype)
    if not resolved_name or not resolved_path:
        return _swarm_error("Model not found.")

    digest = _hash_file_sha256(resolved_path)
    return {"hash": digest}


async def _compat_forward_metadata_request(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    _ = state
    url = str(payload.get("url", "")).strip()
    if not url.startswith("https://civitai.com/"):
        return _swarm_error("Invalid URL.")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.text
    except Exception as ex:
        return _swarm_error(f"{type(ex).__name__}: {ex}")

    try:
        return {"response": json.loads(body)}
    except Exception as ex:
        return _swarm_error(f"{type(ex).__name__}: {ex}")


async def _swarm_get_current_status(state: AppState) -> dict[str, Any]:
    jobs = await state.generation.list_jobs(limit=1000)
    waiting_gens = len([job for job in jobs if job.status == "queued"])
    live_gens = len([job for job in jobs if job.status == "running"])

    backend = state.comfy_manager.snapshot()
    backend_state = backend.status
    if backend_state == "running":
        backend_status = {"status": "running", "class": "", "message": "", "any_loading": False}
    elif backend_state == "starting":
        backend_status = {
            "status": "loading",
            "class": "soft",
            "message": "ComfyUI backend is starting...",
            "any_loading": True,
        }
    elif backend_state == "stopped":
        backend_status = {
            "status": "disabled",
            "class": "warn",
            "message": "ComfyUI backend is stopped.",
            "any_loading": False,
        }
    else:
        backend_status = {
            "status": "errored",
            "class": "error",
            "message": backend.last_error or "ComfyUI backend failed.",
            "any_loading": False,
        }

    return {
        "status": {
            "waiting_gens": waiting_gens,
            "loading_models": 0,
            "waiting_backends": waiting_gens if backend.status != "running" else 0,
            "live_gens": live_gens,
        },
        "backend_status": backend_status,
        "supported_features": [
            "comfyui",
            "txt2img",
            "wildcards",
            "model_list",
        ],
    }


def _compat_generation_params(state: AppState) -> dict[str, Any]:
    cfg = state.get_config()
    models = state.model_index.scan_all()
    return {
        "list": [
            {
                "name": "Prompt",
                "id": "prompt",
                "type": "text",
                "view_type": "big",
                "description": "Main positive prompt",
                "default": "",
                "group": "general",
            },
            {
                "name": "Negative Prompt",
                "id": "negative_prompt",
                "type": "text",
                "view_type": "big",
                "description": "Negative prompt",
                "default": "",
                "group": "general",
            },
            {
                "name": "Model",
                "id": "model",
                "type": "model",
                "description": "Stable Diffusion checkpoint",
                "values": models.get("Stable-Diffusion", []),
                "group": "general",
            },
            {
                "name": "Steps",
                "id": "steps",
                "type": "integer",
                "view_type": "small",
                "default": cfg.generation.default_steps,
                "min": 1,
                "max": 200,
                "group": "sampling",
            },
            {
                "name": "CFG Scale",
                "id": "cfg_scale",
                "type": "decimal",
                "view_type": "small",
                "default": cfg.generation.default_cfg_scale,
                "min": 1,
                "max": 50,
                "group": "sampling",
            },
            {
                "name": "Width",
                "id": "width",
                "type": "integer",
                "view_type": "small",
                "default": cfg.generation.default_width,
                "group": "image",
            },
            {
                "name": "Height",
                "id": "height",
                "type": "integer",
                "view_type": "small",
                "default": cfg.generation.default_height,
                "group": "image",
            },
            {
                "name": "Seed",
                "id": "seed",
                "type": "integer",
                "view_type": "seed",
                "default": -1,
                "group": "sampling",
            },
        ],
        "groups": [
            {
                "name": "General",
                "id": "general",
                "description": "Core generation fields",
                "open": True,
            },
            {
                "name": "Sampling",
                "id": "sampling",
                "description": "Sampling controls",
                "open": True,
            },
            {
                "name": "Image",
                "id": "image",
                "description": "Output image sizing",
                "open": True,
            },
        ],
        "models": models,
        "model_compat_classes": {},
        "model_classes": {},
        "wildcards": state.wildcards.list_wildcards(),
        "param_edits": None,
    }


def _collect_path_folders(relative_path: str, depth: int) -> set[str]:
    folders: set[str] = set()
    parts = [part for part in relative_path.split("/") if part]
    for index in range(1, min(len(parts), depth) + 1):
        folder = "/".join(parts[:index])
        if folder:
            folders.add(folder)
    return folders


def _compat_list_models(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    subtype = str(payload.get("subtype", "Stable-Diffusion"))
    depth = max(1, min(_coerce_int(payload.get("depth"), 3), 20))
    path_prefix = str(payload.get("path", "")).replace("\\", "/").strip("/")
    sort_by = str(payload.get("sortBy", "Name")).lower()
    sort_reverse = _coerce_bool(payload.get("sortReverse"), False)

    if subtype == "Wildcards":
        items = state.wildcards.list_wildcards()
    else:
        all_models = state.model_index.scan_all()
        if subtype not in all_models:
            return {"error": "Invalid sub-type."}
        items = all_models[subtype]

    prefix = f"{path_prefix}/" if path_prefix else ""
    folders: set[str] = set()
    files: list[dict[str, Any]] = []

    for name in items:
        normalized = name.replace("\\", "/")
        if prefix and not normalized.startswith(prefix):
            continue
        relative = normalized[len(prefix) :] if prefix else normalized
        slash_count = relative.count("/")
        folders.update(_collect_path_folders(relative.rsplit("/", 1)[0], depth))
        if slash_count >= depth:
            continue
        files.append(
            {
                "name": normalized,
                "title": Path(normalized).name,
                "local": True,
            }
        )

    if sort_by == "datecreated" or sort_by == "datemodified":
        files.sort(key=lambda item: item["name"])
    else:
        files.sort(key=lambda item: item["name"])
    if sort_reverse:
        files.reverse()

    return {
        "folders": sorted(folders),
        "files": files,
    }


def _compat_list_images(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    depth = max(1, min(_coerce_int(payload.get("depth"), 3), 20))
    path_prefix = str(payload.get("path", "")).replace("\\", "/").strip("/")
    sort_by = str(payload.get("sortBy", "Name")).lower()
    sort_reverse = _coerce_bool(payload.get("sortReverse"), False)

    images = state.generation.list_output_images(limit=1000)
    prefix = f"{path_prefix}/" if path_prefix else ""

    enriched: list[tuple[str, float]] = []
    folders: set[str] = set()
    output_root = resolve_path(state.get_config().paths.output_root)

    for image in images:
        rel_path = image.relative_path.replace("\\", "/")
        if prefix and not rel_path.startswith(prefix):
            continue
        relative = rel_path[len(prefix) :] if prefix else rel_path
        slash_count = relative.count("/")
        folders.update(_collect_path_folders(relative.rsplit("/", 1)[0], depth))
        if slash_count >= depth:
            continue
        file_path = output_root / rel_path
        mtime = file_path.stat().st_mtime if file_path.exists() else 0.0
        enriched.append((rel_path, mtime))

    if sort_by == "date":
        enriched.sort(key=lambda item: item[1], reverse=True)
    else:
        enriched.sort(key=lambda item: item[0])

    if sort_reverse:
        enriched.reverse()

    files = [{"src": item[0], "metadata": ""} for item in enriched]
    return {
        "folders": sorted(folders),
        "files": files,
    }


def _build_generation_request_from_swarm(
    state: AppState,
    payload: dict[str, Any],
) -> tuple[int, GenerationRequest]:
    cfg = state.get_config()

    images_count = max(1, _coerce_int(payload.get("images"), 1))
    batch_size = max(1, _coerce_int(payload.get("batchsize", payload.get("batch_size")), 1))

    prompt = str(payload.get("prompt", "")).strip()
    negative_prompt = str(
        payload.get("negative_prompt", payload.get("negativeprompt", ""))
    ).strip()

    model = str(payload.get("model", payload.get("sd_model", cfg.generation.default_model))).strip()
    if not model:
        discovered = state.model_index.scan_all().get("Stable-Diffusion", [])
        model = discovered[0] if discovered else ""
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No model supplied and no local Stable-Diffusion models were discovered.",
        )
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    model_architecture = str(payload.get("model_architecture", payload.get("modelArchitecture", "auto")))
    model_architecture = model_architecture.strip().lower()
    if model_architecture not in {"auto", "checkpoint", "flux"}:
        model_architecture = "auto"

    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        model_architecture=model_architecture,
        steps=_coerce_int(payload.get("steps"), cfg.generation.default_steps),
        cfg_scale=_coerce_float(
            payload.get("cfg_scale", payload.get("cfgscale")),
            cfg.generation.default_cfg_scale,
        ),
        seed=_coerce_int(payload.get("seed"), -1),
        width=_coerce_int(payload.get("width"), cfg.generation.default_width),
        height=_coerce_int(payload.get("height"), cfg.generation.default_height),
        sampler_name=str(payload.get("sampler_name", payload.get("sampler", cfg.generation.default_sampler))),
        scheduler=str(payload.get("scheduler", cfg.generation.default_scheduler)),
        denoise=_coerce_float(payload.get("denoise"), 1.0),
        batch_size=batch_size,
        filename_prefix=str(payload.get("filename_prefix", "swarmui")),
        flux_clip_name1=str(payload.get("flux_clip_name1", payload.get("clip_name1", ""))).strip(),
        flux_clip_name2=str(payload.get("flux_clip_name2", payload.get("clip_name2", ""))).strip(),
        flux_vae_name=str(payload.get("flux_vae_name", payload.get("vae_name", payload.get("vae", "")))).strip(),
        flux_guidance=_coerce_optional_float(
            payload.get("flux_guidance", payload.get("guidance"))
        ),
    )
    return images_count, request


async def _compat_generate_text_to_image(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    images_count, request = _build_generation_request_from_swarm(state, payload)
    timeout_seconds = max(1.0, _coerce_float(payload.get("timeout_seconds"), 3600.0))

    results: list[str] = []
    for index in range(images_count):
        request_for_run = request
        if request.seed >= 0:
            request_for_run = request.model_copy(update={"seed": request.seed + index})

        job = await state.generation.submit(request_for_run)
        completed = await state.generation.wait_for_completion(
            job.id,
            timeout_seconds=timeout_seconds,
        )

        if completed.status != "succeeded":
            return {
                "error": completed.error or completed.message or "Generation failed.",
                "job_id": completed.id,
            }
        results.extend([f"View/{image.relative_path}" for image in completed.images])

    return {"images": results}


async def _compat_list_loaded_models(state: AppState) -> dict[str, Any]:
    jobs = await state.generation.list_jobs(limit=2000)
    models: list[str] = []
    seen: set[str] = set()

    default_model = state.get_config().generation.default_model.strip()
    if default_model:
        models.append(default_model)
        seen.add(default_model)

    for job in jobs:
        model = job.request.model.strip()
        if not model or model in seen:
            continue
        if job.status in {"running", "succeeded"}:
            seen.add(model)
            models.append(model)

    return {
        "models": [
            {
                "name": model,
                "title": Path(model).name,
                "local": True,
            }
            for model in models
        ]
    }


async def _compat_select_model(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", "")).strip()
    if not model:
        return _swarm_error("Invalid empty model name.")

    available_models = state.model_index.scan_all().get("Stable-Diffusion", [])
    if model not in available_models:
        return _swarm_error("Model not found.")

    cfg = state.get_config().model_copy(deep=True)
    cfg.generation.default_model = model
    state.update_config(cfg)
    return {"success": True, "model": model}


async def _compat_generate_text_to_image_ws(
    state: AppState,
    websocket: WebSocket,
    payload: dict[str, Any],
) -> None:
    images_count, request = _build_generation_request_from_swarm(state, payload)
    timeout_seconds = max(1.0, _coerce_float(payload.get("timeout_seconds"), 3600.0))
    batch_index = 0

    await websocket.send_json(await _swarm_get_current_status(state))

    for index in range(images_count):
        request_for_run = request
        if request.seed >= 0:
            request_for_run = request.model_copy(update={"seed": request.seed + index})

        job = await state.generation.submit(request_for_run)
        prior_progress = -1.0
        started = asyncio.get_running_loop().time()
        while True:
            current = await state.generation.get_job(job.id)
            if not current:
                await websocket.send_json(_swarm_error(f"Generation job '{job.id}' disappeared."))
                return

            progress = max(0.0, min(1.0, float(current.progress)))
            if progress != prior_progress:
                prior_progress = progress
                await websocket.send_json(
                    {
                        "gen_progress": {
                            "batch_index": str(batch_index),
                            "overall_percent": progress,
                            "current_percent": progress,
                        }
                    }
                )

            if current.status in {"succeeded", "failed", "cancelled"}:
                break
            if asyncio.get_running_loop().time() - started > timeout_seconds:
                await state.generation.cancel(job.id)
                await websocket.send_json(_swarm_error("Generation timed out."))
                return
            await asyncio.sleep(0.5)

        if current.status != "succeeded":
            await websocket.send_json(
                _swarm_error(current.error or current.message or "Generation failed.")
            )
            return

        for image in current.images:
            await websocket.send_json(
                {
                    "image": {
                        "image": f"View/{image.relative_path}",
                        "batch_index": str(batch_index),
                        "metadata": "",
                    }
                }
            )
            batch_index += 1

    await websocket.send_json(await _swarm_get_current_status(state))
    await websocket.send_json({"socket_intention": "close"})


async def _compat_interrupt_all(state: AppState) -> dict[str, Any]:
    jobs = await state.generation.list_jobs(limit=2000)
    cancelled = 0
    for job in jobs:
        if job.status in {"queued", "running"}:
            result = await state.generation.cancel(job.id)
            if result:
                cancelled += 1
    return {"success": True, "cancelled": cancelled}


def build_router(state: AppState) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/status")
    async def status() -> dict[str, Any]:
        jobs = await state.generation.list_jobs(limit=200)
        counts = {
            "queued": len([job for job in jobs if job.status == "queued"]),
            "running": len([job for job in jobs if job.status == "running"]),
            "succeeded": len([job for job in jobs if job.status == "succeeded"]),
            "failed": len([job for job in jobs if job.status == "failed"]),
            "cancelled": len([job for job in jobs if job.status == "cancelled"]),
        }
        return {
            "backend": state.comfy_manager.snapshot(),
            "job_counts": counts,
            "models": state.model_index.scan_all(),
            "wildcards": state.wildcards.list_wildcards(),
        }

    @router.get("/settings")
    async def get_settings() -> AppConfig:
        return state.get_config()

    @router.put("/settings")
    async def update_settings(config: AppConfig) -> AppConfig:
        return state.update_config(config)

    @router.get("/backend")
    async def backend_status() -> dict[str, Any]:
        return state.comfy_manager.snapshot().model_dump(mode="json")

    @router.get("/backend/preflight")
    async def backend_preflight(refresh: bool = True) -> dict[str, Any]:
        cfg = state.get_config()
        if refresh or state.comfy_manager.last_preflight() is None:
            return await state.comfy_manager.preflight_check(
                timeout_seconds=cfg.comfy.preflight_timeout_seconds
            )
        cached = state.comfy_manager.last_preflight()
        return cached if cached is not None else {}

    @router.post("/backend/start")
    async def backend_start() -> dict[str, Any]:
        try:
            snapshot = await state.comfy_manager.start()
            return snapshot.model_dump(mode="json")
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex)) from ex

    @router.post("/backend/stop")
    async def backend_stop() -> dict[str, Any]:
        snapshot = await state.comfy_manager.stop()
        return snapshot.model_dump(mode="json")

    @router.post("/backend/restart")
    async def backend_restart() -> dict[str, Any]:
        try:
            snapshot = await state.comfy_manager.restart()
            return snapshot.model_dump(mode="json")
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex)) from ex

    @router.get("/models")
    async def models() -> dict[str, list[str]]:
        return state.model_index.scan_all()

    @router.get("/wildcards")
    async def wildcards() -> dict[str, list[str]]:
        return {"wildcards": state.wildcards.list_wildcards()}

    @router.get("/output")
    async def output_images() -> dict[str, Any]:
        return {
            "images": [
                image.model_dump(mode="json") for image in state.generation.list_output_images()
            ]
        }

    @router.post("/generate")
    async def generate(request: GenerationRequest) -> dict[str, Any]:
        job = await state.generation.submit(request)
        return {"job": job.model_dump(mode="json")}

    @router.get("/jobs")
    async def list_jobs(limit: int = 100) -> dict[str, Any]:
        jobs = await state.generation.list_jobs(limit=limit)
        return {"jobs": [job.model_dump(mode="json") for job in jobs]}

    @router.get("/jobs/{job_id}")
    async def job_by_id(job_id: str) -> dict[str, Any]:
        job = await state.generation.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job": job.model_dump(mode="json")}

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = await state.generation.cancel(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job": job.model_dump(mode="json")}

    @router.get("/image/{image_path:path}")
    async def view_image(image_path: str) -> FileResponse:
        return _output_image_response(state, image_path)

    @router.api_route(
        "/comfy/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def comfy_proxy(full_path: str, request: Request) -> Response:
        return await _proxy_to_comfy(state, full_path, request)

    @router.get("/", response_class=HTMLResponse)
    async def index() -> str:
        index_path = _resolve_template_file("index.html")
        return index_path.read_text(encoding="utf-8")

    return router


def build_compat_router(state: AppState) -> APIRouter:
    router = APIRouter()

    @router.get("/API")
    async def api_index() -> dict[str, Any]:
        return {
            "name": "Python SwarmUI compatibility API",
            "calls": [
                "GetCurrentStatus",
                "ListT2IParams",
                "GenerateText2Image",
                "GenerateText2ImageWS",
                "ListImages",
                "ListModels",
                "DescribeModel",
                "ListLoadedModels",
                "SelectModel",
                "SelectModelWS",
                "GetModelHash",
                "ForwardMetadataRequest",
                "TriggerRefresh",
                "InterruptAll",
            ],
        }

    @router.api_route("/API/{call_name}", methods=["GET", "POST"])
    async def swarm_api_dispatch(call_name: str, request: Request) -> dict[str, Any]:
        payload = await _extract_payload(request)
        call = call_name.lower()
        LOGGER.debug(
            "Compatibility API call '%s' via %s with keys=%s",
            call_name,
            request.method,
            sorted(payload.keys()),
        )

        if call == "getcurrentstatus":
            return await _swarm_get_current_status(state)
        if call == "listt2iparams":
            return _compat_generation_params(state)
        if call == "generatetext2image":
            return await _compat_generate_text_to_image(state, payload)
        if call == "listimages":
            return _compat_list_images(state, payload)
        if call == "listmodels":
            return _compat_list_models(state, payload)
        if call == "describemodel":
            return await _compat_describe_model(state, payload)
        if call == "listloadedmodels":
            return await _compat_list_loaded_models(state)
        if call == "selectmodel":
            return await _compat_select_model(state, payload)
        if call == "selectmodelws":
            return await _compat_select_model(state, payload)
        if call == "getmodelhash":
            return await _compat_get_model_hash(state, payload)
        if call == "forwardmetadatarequest":
            return await _compat_forward_metadata_request(state, payload)
        if call == "triggerrefresh":
            return {
                "success": True,
                "models": state.model_index.scan_all(),
                "wildcards": state.wildcards.list_wildcards(),
            }
        if call == "interruptall":
            return await _compat_interrupt_all(state)
        if call == "generatetext2imagews":
            return await _compat_generate_text_to_image(state, payload)
        return {"error": f"Unknown API call '{call_name}'."}

    @router.websocket("/API/GenerateText2ImageWS")
    async def generate_text_to_image_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            payload = await _extract_ws_payload(websocket)
            LOGGER.debug("Compatibility WS GenerateText2ImageWS payload keys=%s", sorted(payload.keys()))
            await _compat_generate_text_to_image_ws(state, websocket, payload)
        except WebSocketDisconnect:
            return
        except HTTPException as ex:
            await websocket.send_json(_swarm_error(str(ex.detail)))
        except ValueError as ex:
            await websocket.send_json(_swarm_error(str(ex)))
        except Exception as ex:
            await websocket.send_json(_swarm_error(str(ex)))
        finally:
            await _safe_close_websocket(websocket)

    @router.websocket("/API/SelectModelWS")
    async def select_model_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            payload = await _extract_ws_payload(websocket)
            LOGGER.debug("Compatibility WS SelectModelWS payload keys=%s", sorted(payload.keys()))
            result = await _compat_select_model(state, payload)
            await websocket.send_json(result)
            await websocket.send_json(await _swarm_get_current_status(state))
        except WebSocketDisconnect:
            return
        except ValueError as ex:
            await websocket.send_json(_swarm_error(str(ex)))
        except Exception as ex:
            await websocket.send_json(_swarm_error(str(ex)))
        finally:
            await _safe_close_websocket(websocket)

    @router.get("/View/{image_path:path}")
    async def view_output_compat(image_path: str) -> FileResponse:
        return _output_image_response(state, image_path)

    @router.get("/Output/{image_path:path}")
    async def output_compat(image_path: str) -> FileResponse:
        return _output_image_response(state, image_path)

    @router.get("/Text2Image")
    async def text2image_compat() -> RedirectResponse:
        return RedirectResponse(url="/api/")

    @router.api_route(
        "/ComfyBackendDirect/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def comfy_backend_direct(full_path: str, request: Request) -> Response:
        return await _proxy_to_comfy(state, full_path, request)

    return router
