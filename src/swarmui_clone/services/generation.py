from __future__ import annotations

import asyncio
import copy
import random
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from swarmui_clone.config import AppConfig, resolve_path
from swarmui_clone.schemas import GenerationImage, GenerationJob, GenerationRequest
from swarmui_clone.services.comfy_client import ComfyClient
from swarmui_clone.services.comfy_launcher import ComfyProcessManager
from swarmui_clone.services.wildcards import WildcardService
from swarmui_clone.services.workflow_builder import WorkflowBuilder
from swarmui_clone.utils.pathing import ensure_directory


class GenerationService:
    def __init__(
        self,
        config_provider,
        comfy_manager: ComfyProcessManager,
        comfy_client: ComfyClient,
        wildcard_service: WildcardService,
        workflow_builder: WorkflowBuilder,
    ) -> None:
        self._config_provider = config_provider
        self._comfy_manager = comfy_manager
        self._comfy_client = comfy_client
        self._wildcard_service = wildcard_service
        self._workflow_builder = workflow_builder

        self._jobs: dict[str, GenerationJob] = {}
        self._jobs_lock = asyncio.Lock()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        cfg = self._config_provider()
        worker_count = max(1, cfg.generation.max_concurrent_jobs)
        if self._workers:
            return
        for index in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(index)))

    async def shutdown(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def submit(self, request: GenerationRequest) -> GenerationJob:
        job_id = str(uuid.uuid4())
        now = GenerationJob.now()
        seed = request.seed if request.seed >= 0 else random.randint(0, 2_147_483_647)

        positive = self._wildcard_service.expand(request.prompt, seed)
        negative = self._wildcard_service.expand(request.negative_prompt, seed + 1)

        job = GenerationJob(
            id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=request,
            expanded_prompt=positive.text,
            expanded_negative_prompt=negative.text,
            wildcard_tokens=[*positive.tokens, *negative.tokens],
            generation_seed=seed,
            queue_position=self._queue.qsize() + 1,
            message="Queued",
        )
        async with self._jobs_lock:
            self._jobs[job_id] = job
        await self._queue.put(job_id)
        return copy.deepcopy(job)

    async def cancel(self, job_id: str) -> GenerationJob | None:
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status in {"succeeded", "failed", "cancelled"}:
                return copy.deepcopy(job)
            job.status = "cancelled"
            job.message = "Cancelled by user"
            job.updated_at = GenerationJob.now()
            prompt_id = job.prompt_id

        if prompt_id:
            try:
                await self._comfy_client.cancel_prompt(prompt_id)
                await self._comfy_client.interrupt_prompt(prompt_id)
            except Exception:
                pass

        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> GenerationJob | None:
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    async def wait_for_completion(
        self,
        job_id: str,
        timeout_seconds: float = 3600.0,
        poll_seconds: float = 0.5,
    ) -> GenerationJob:
        end_time = time.monotonic() + timeout_seconds
        while True:
            job = await self.get_job(job_id)
            if not job:
                raise RuntimeError(f"Job '{job_id}' was not found")
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            if time.monotonic() >= end_time:
                raise TimeoutError(f"Timed out waiting for job '{job_id}'")
            await asyncio.sleep(poll_seconds)

    async def list_jobs(self, limit: int = 100) -> list[GenerationJob]:
        async with self._jobs_lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [copy.deepcopy(job) for job in jobs[:limit]]

    def _resolve_comfy_output(self, image_type: str, subfolder: str, filename: str) -> Path:
        cfg: AppConfig = self._config_provider()
        comfy_root = resolve_path(cfg.comfy.start_script).parent
        image_root = comfy_root / image_type
        return (image_root / subfolder / filename).resolve()

    def _copy_output_image(self, source_path: Path) -> GenerationImage:
        cfg: AppConfig = self._config_provider()
        output_root = ensure_directory(resolve_path(cfg.paths.output_root))
        subdir_name = datetime.now().strftime(cfg.generation.output_subdir_format)
        target_dir = ensure_directory(output_root / subdir_name)

        destination = target_dir / source_path.name
        counter = 1
        while destination.exists():
            destination = target_dir / f"{source_path.stem}-{counter}{source_path.suffix}"
            counter += 1

        shutil.copy2(source_path, destination)

        return GenerationImage(
            file_name=destination.name,
            relative_path=destination.relative_to(output_root).as_posix(),
            absolute_path=str(destination),
        )

    async def _update_job(self, job_id: str, **kwargs) -> None:
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)
            job.updated_at = GenerationJob.now()

    async def _extract_images(self, job_id: str, history_payload: dict) -> list[GenerationImage]:
        job = await self.get_job(job_id)
        if not job or not job.prompt_id:
            return []

        prompt_data = history_payload.get(job.prompt_id) or {}
        outputs = prompt_data.get("outputs") or {}
        images: list[GenerationImage] = []

        for node_data in outputs.values():
            for image_info in node_data.get("images", []):
                filename = image_info.get("filename")
                if not filename:
                    continue
                subfolder = image_info.get("subfolder", "")
                image_type = image_info.get("type", "output")
                source = self._resolve_comfy_output(image_type, subfolder, filename)
                if not source.exists():
                    continue
                images.append(self._copy_output_image(source))

        return images

    async def _run_generation(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if not job:
            return
        if job.status == "cancelled":
            return

        await self._update_job(job_id, status="running", message="Starting backend", progress=0.02)
        await self._comfy_manager.ensure_running()

        job = await self.get_job(job_id)
        if not job or job.status == "cancelled":
            return

        workflow = self._workflow_builder.build(
            request=job.request,
            seed=job.generation_seed or 0,
            prompt=job.expanded_prompt or job.request.prompt,
            negative_prompt=job.expanded_negative_prompt or job.request.negative_prompt,
        )

        client_id = str(uuid.uuid4())
        await self._update_job(job_id, message="Submitting prompt", progress=0.08)
        prompt_data = await self._comfy_client.queue_prompt(workflow=workflow, client_id=client_id)

        if "error" in prompt_data:
            raise RuntimeError(str(prompt_data["error"]))

        prompt_id = prompt_data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return a prompt_id")

        await self._update_job(job_id, prompt_id=prompt_id, message="Waiting for result", progress=0.12)

        attempts = 0
        max_attempts = 60 * 60
        while True:
            attempts += 1
            job = await self.get_job(job_id)
            if not job or job.status == "cancelled":
                if job and job.prompt_id:
                    await self._comfy_client.cancel_prompt(job.prompt_id)
                    await self._comfy_client.interrupt_prompt(job.prompt_id)
                return

            history = await self._comfy_client.get_history(prompt_id)
            images = await self._extract_images(job_id, history)
            if images:
                await self._update_job(
                    job_id,
                    status="succeeded",
                    message=f"Completed with {len(images)} image(s)",
                    images=images,
                    progress=1.0,
                )
                return

            if attempts >= max_attempts:
                raise TimeoutError("Timed out waiting for ComfyUI generation result")

            progress = min(0.12 + (attempts * 0.01), 0.95)
            await self._update_job(job_id, progress=progress, message="Processing")
            await asyncio.sleep(1)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = await self.get_job(job_id)
                if not job or job.status == "cancelled":
                    continue
                await self._run_generation(job_id)
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                await self._update_job(
                    job_id,
                    status="failed",
                    error=str(ex),
                    message="Generation failed",
                    progress=1.0,
                )
            finally:
                self._queue.task_done()

    def list_output_images(self, limit: int = 200) -> list[GenerationImage]:
        cfg = self._config_provider()
        output_root = resolve_path(cfg.paths.output_root)
        if not output_root.exists():
            return []

        files = [
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        files.sort(key=lambda item: item.stat().st_mtime, reverse=True)

        out: list[GenerationImage] = []
        for path in files[:limit]:
            out.append(
                GenerationImage(
                    file_name=path.name,
                    relative_path=path.relative_to(output_root).as_posix(),
                    absolute_path=str(path),
                )
            )
        return out
