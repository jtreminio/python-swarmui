from __future__ import annotations

from typing import Any

from swarmui_clone.schemas import GenerationRequest


class WorkflowBuilder:
    @staticmethod
    def _align_size(value: int) -> int:
        return max(64, int(value / 8) * 8)

    def build(self, request: GenerationRequest, seed: int, prompt: str, negative_prompt: str) -> dict[str, Any]:
        if request.custom_workflow:
            return request.custom_workflow

        width = self._align_size(request.width)
        height = self._align_size(request.height)

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
