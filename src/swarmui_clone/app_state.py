from __future__ import annotations

from pathlib import Path

from swarmui_clone.config import AppConfig, app_root, resolve_path
from swarmui_clone.services.comfy_client import ComfyClient
from swarmui_clone.services.comfy_launcher import ComfyProcessManager
from swarmui_clone.services.generation import GenerationService
from swarmui_clone.services.model_index import ModelIndexService
from swarmui_clone.services.settings_service import SettingsService
from swarmui_clone.services.wildcards import WildcardService
from swarmui_clone.services.workflow_builder import WorkflowBuilder
from swarmui_clone.utils.pathing import ensure_directory


class AppState:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or app_root()
        self.settings_service = SettingsService(self.root / "config" / "settings.yaml")
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
            await self.comfy_manager.start()

    async def shutdown(self) -> None:
        await self.generation.shutdown()
        await self.comfy_manager.stop()
