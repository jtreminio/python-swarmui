from __future__ import annotations

import logging
from pathlib import Path

from swarmui_clone.config import AppConfig, app_root, resolve_path
from swarmui_clone.services.comfy_client import ComfyClient
from swarmui_clone.services.comfy_launcher import ComfyProcessManager
from swarmui_clone.services.generation import GenerationService
from swarmui_clone.services.model_index import ModelIndexService
from swarmui_clone.services.settings_service import SettingsService, resolve_settings_path
from swarmui_clone.services.wildcards import WildcardService
from swarmui_clone.services.workflow_builder import WorkflowBuilder
from swarmui_clone.utils.pathing import ensure_directory

LOGGER = logging.getLogger("swarmui_clone.state")


class AppState:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or app_root()
        self.settings_service = SettingsService(resolve_settings_path(self.root))
        self.config: AppConfig = self.settings_service.load()

        self.comfy_manager = ComfyProcessManager(config_provider=self.get_config)
        self.comfy_client = ComfyClient(self.comfy_manager)
        self.model_index = ModelIndexService(self.get_config)
        self.wildcards = WildcardService(self.get_config)
        self.workflow_builder = WorkflowBuilder()
        self.generation = GenerationService(
            config_provider=self.get_config,
            comfy_manager=self.comfy_manager,
            comfy_client=self.comfy_client,
            wildcard_service=self.wildcards,
            workflow_builder=self.workflow_builder,
        )

    def get_config(self) -> AppConfig:
        return self.config

    def update_config(self, config: AppConfig) -> AppConfig:
        self.config = config
        self.settings_service.save(config)
        return self.config

    async def startup(self) -> None:
        ensure_directory(resolve_path(self.config.paths.data_root))
        ensure_directory(resolve_path(self.config.paths.output_root))
        await self.generation.start()
        if self.config.comfy.auto_start_on_boot:
            if self.config.comfy.enable_preflight_checks:
                preflight = await self.comfy_manager.preflight_check(
                    timeout_seconds=self.config.comfy.preflight_timeout_seconds
                )
                if not preflight["ok"]:
                    LOGGER.warning(
                        "ComfyUI preflight failed on startup: %s",
                        "; ".join(preflight.get("errors", [])) or "unknown error",
                    )
                    for line in preflight.get("output_tail", []):
                        LOGGER.warning("[PRECHECK] %s", line)
                    for line in preflight.get("import_probe_output_tail", []):
                        LOGGER.warning("[PRECHECK-IMPORT] %s", line)
                    if self.config.comfy.skip_auto_start_on_preflight_error:
                        LOGGER.warning(
                            "Skipping backend auto-start due to preflight failure "
                            "(skip_auto_start_on_preflight_error=true)."
                        )
                        return
                    raise RuntimeError(
                        "ComfyUI preflight failed and skip_auto_start_on_preflight_error=false."
                    )
                await self.comfy_manager.start(run_preflight=False)
                return
            await self.comfy_manager.start()

    async def shutdown(self) -> None:
        await self.generation.shutdown()
        await self.comfy_manager.stop()
