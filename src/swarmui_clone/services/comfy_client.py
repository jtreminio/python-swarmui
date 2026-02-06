from __future__ import annotations

from typing import Any

import httpx

from swarmui_clone.services.comfy_launcher import ComfyProcessManager


class ComfyClient:
    def __init__(self, manager: ComfyProcessManager) -> None:
        self._manager = manager

    @property
    def base_url(self) -> str:
        api_url = self._manager.api_url
        if not api_url:
            raise RuntimeError("ComfyUI backend is not running")
        return api_url.rstrip("/")

    async def get_object_info(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/object_info")
            response.raise_for_status()
            return response.json()

    async def get_system_stats(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()

    async def queue_prompt(self, workflow: dict[str, Any], client_id: str) -> dict[str, Any]:
        payload = {"prompt": workflow, "client_id": client_id}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.base_url}/prompt", json=payload)
            response.raise_for_status()
            return response.json()

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()

    async def cancel_prompt(self, prompt_id: str) -> None:
        payload = {"delete": [prompt_id]}
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{self.base_url}/queue", json=payload)
            await client.post(f"{self.base_url}/history", json=payload)

    async def interrupt_prompt(self, prompt_id: str) -> None:
        payload = {"prompt_id": prompt_id}
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{self.base_url}/interrupt", json=payload)
