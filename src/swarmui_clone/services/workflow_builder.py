from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from swarmui_clone.schemas import GenerationRequest


class WorkflowBuildError(ValueError):
    pass


class WorkflowBuilder:
    MODEL_PARENT_PROFILES: dict[str, dict[str, Any]] = {
        "z-image": {
            "clip_name": "qwen_3_4b.safetensors",
            "vae_name": "Flux/ae.safetensors",
            "clip_loader_type": "stable_diffusion",
            "latent_node_type": "EmptySD3LatentImage",
            "size_step": 16,
        },
        "flux": {
            "clip_name": "qwen_3_4b.safetensors",
            "vae_name": "Flux/flux2-vae.safetensors",
            "clip_loader_type": "flux2",
            "latent_node_type": "EmptyFlux2LatentImage",
            "size_step": 16,
        },
    }

    @staticmethod
    def _align_size(value: int, step: int = 8) -> int:
        return max(step, int(value / step) * step)

    @staticmethod
    def _normalize_name(value: str) -> str:
        return Path(value).name.lower()

    @staticmethod
    def _normalized_parent_name(value: str) -> str:
        normalized = Path(value.replace("\\", "/"))
        parent = normalized.parent.name.strip().lower()
        return parent

    @staticmethod
    def _has_any_token(value: str, tokens: Iterable[str]) -> bool:
        return any(token in value for token in tokens)

    @staticmethod
    def _pick_candidate(
        candidates: list[str],
        include_tokens: tuple[str, ...],
        avoid_tokens: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
    ) -> str | None:
        excluded = {item.lower() for item in exclude if item}
        pool = [(item, Path(item).name.lower()) for item in candidates if item.lower() not in excluded]

        for token in include_tokens:
            for original, lowered in pool:
                if token in lowered and not any(avoid in lowered for avoid in avoid_tokens):
                    return original

        for original, lowered in pool:
            if avoid_tokens and any(avoid in lowered for avoid in avoid_tokens):
                continue
            return original

        return None

    def _is_flux_mode(self, request: GenerationRequest) -> bool:
        architecture = request.model_architecture.strip().lower()
        if architecture == "flux":
            return True
        if architecture == "checkpoint":
            return False

        if request.flux_clip_name1.strip() or request.flux_clip_name2.strip() or request.flux_vae_name.strip():
            return True

        model_name = self._normalize_name(request.model)
        return self._has_any_token(model_name, ("flux", "schnell"))

    def _profile_from_model_parent(self, request: GenerationRequest) -> dict[str, Any] | None:
        parent = self._normalized_parent_name(request.model)
        return self.MODEL_PARENT_PROFILES.get(parent)

    def _resolve_flux_components(
        self,
        request: GenerationRequest,
        available_models: dict[str, list[str]],
    ) -> tuple[str, str, str, float]:
        clip_models = available_models.get("CLIP", [])
        vae_models = available_models.get("VAE", [])

        clip_name1 = request.flux_clip_name1.strip()
        clip_name2 = request.flux_clip_name2.strip()
        vae_name = request.flux_vae_name.strip()

        if not clip_name1:
            clip_name1 = self._pick_candidate(
                clip_models,
                include_tokens=("clip_l", "clip-l", "clipl", "vit-l", "clip"),
                avoid_tokens=("t5",),
            ) or ""
        if not clip_name2:
            clip_name2 = self._pick_candidate(
                clip_models,
                include_tokens=("t5xxl", "t5_xxl", "t5xx", "umt5", "t5"),
                exclude=(clip_name1,),
            ) or ""
        if not vae_name:
            vae_name = self._pick_candidate(
                vae_models,
                include_tokens=("ae.safetensors", "flux_vae", "fluxvae", "ae", "vae"),
            ) or ""

        if not clip_name1 or not clip_name2:
            raise WorkflowBuildError(
                "FLUX workflow requires two text encoders. Set flux_clip_name1/flux_clip_name2 "
                "or add CLIP models (for example clip_l + t5xxl) under your text_encoders folders."
            )
        if not vae_name:
            raise WorkflowBuildError(
                "FLUX workflow requires a VAE. Set flux_vae_name or add a VAE model "
                "(for example ae.safetensors) under your VAE folders."
            )

        guidance = request.flux_guidance if request.flux_guidance is not None else request.cfg_scale
        return clip_name1, clip_name2, vae_name, max(0.1, float(guidance))

    def _build_parent_profile_workflow(
        self,
        request: GenerationRequest,
        seed: int,
        prompt: str,
        negative_prompt: str,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        size_step = int(profile.get("size_step", 16))
        width = self._align_size(request.width, step=size_step)
        height = self._align_size(request.height, step=size_step)
        clip_name = request.flux_clip_name1.strip() or str(profile["clip_name"])
        vae_name = request.flux_vae_name.strip() or str(profile["vae_name"])
        latent_node_type = str(profile["latent_node_type"])
        clip_loader_type = str(profile["clip_loader_type"])

        workflow: dict[str, Any] = {
            "3": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": request.model},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": request.steps,
                    "cfg": request.cfg_scale,
                    "sampler_name": request.sampler_name,
                    "scheduler": request.scheduler,
                    "denoise": request.denoise,
                    "model": ["3", 0],
                    "positive": ["11", 0],
                    "negative": ["12", 0],
                    "latent_image": ["5", 0],
                },
            },
            "5": {
                "class_type": latent_node_type,
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": request.batch_size,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["13", 0],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": request.filename_prefix,
                    "images": ["8", 0],
                },
            },
            "10": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": clip_name,
                    "type": clip_loader_type,
                },
            },
            "11": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["10", 0],
                },
            },
            "13": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_name,
                },
            },
        }

        if negative_prompt.strip():
            workflow["12"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["10", 0],
                },
            }
        else:
            workflow["12"] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["11", 0],
                },
            }

        return workflow

    def _build_checkpoint_workflow(
        self,
        request: GenerationRequest,
        seed: int,
        prompt: str,
        negative_prompt: str,
    ) -> dict[str, Any]:
        width = self._align_size(request.width, step=8)
        height = self._align_size(request.height, step=8)
        return {
            "3": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": request.model},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": request.steps,
                    "cfg": request.cfg_scale,
                    "sampler_name": request.sampler_name,
                    "scheduler": request.scheduler,
                    "denoise": request.denoise,
                    "model": ["3", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": request.batch_size,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["3", 1],
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["3", 1],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["3", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": request.filename_prefix,
                    "images": ["8", 0],
                },
            },
        }

    def _build_flux_workflow(
        self,
        request: GenerationRequest,
        seed: int,
        prompt: str,
        negative_prompt: str,
        available_models: dict[str, list[str]],
    ) -> dict[str, Any]:
        width = self._align_size(request.width, step=16)
        height = self._align_size(request.height, step=16)
        clip_name1, clip_name2, vae_name, guidance = self._resolve_flux_components(
            request,
            available_models,
        )

        positive_node = "11"
        negative_node = "12"
        workflow: dict[str, Any] = {
            "3": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": request.model},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": request.steps,
                    "cfg": 1.0,
                    "sampler_name": request.sampler_name,
                    "scheduler": request.scheduler,
                    "denoise": request.denoise,
                    "model": ["3", 0],
                    "positive": [positive_node, 0],
                    "negative": [negative_node, 0],
                    "latent_image": ["5", 0],
                },
            },
            "5": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": request.batch_size,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["13", 0],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": request.filename_prefix,
                    "images": ["8", 0],
                },
            },
            "10": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": clip_name1,
                    "clip_name2": clip_name2,
                    "type": "flux",
                },
            },
            "11": {
                "class_type": "CLIPTextEncodeFlux",
                "inputs": {
                    "clip": ["10", 0],
                    "clip_l": prompt,
                    "t5xxl": prompt,
                    "guidance": guidance,
                },
            },
            "13": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_name,
                },
            },
        }

        if negative_prompt.strip():
            workflow[negative_node] = {
                "class_type": "CLIPTextEncodeFlux",
                "inputs": {
                    "clip": ["10", 0],
                    "clip_l": negative_prompt,
                    "t5xxl": negative_prompt,
                    "guidance": guidance,
                },
            }
        else:
            workflow[negative_node] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": [positive_node, 0],
                },
            }

        return workflow

    def build(
        self,
        request: GenerationRequest,
        seed: int,
        prompt: str,
        negative_prompt: str,
        available_models: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        if request.custom_workflow:
            return request.custom_workflow

        parent_profile = self._profile_from_model_parent(request)
        if parent_profile is not None:
            return self._build_parent_profile_workflow(
                request=request,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                profile=parent_profile,
            )

        if self._is_flux_mode(request):
            return self._build_flux_workflow(
                request=request,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                available_models=available_models or {},
            )

        return self._build_checkpoint_workflow(
            request=request,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
