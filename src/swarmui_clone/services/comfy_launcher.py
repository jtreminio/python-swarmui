from __future__ import annotations

import ast
import asyncio
import copy
import json
import logging
import os
import random
import re
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from swarmui_clone.config import AppConfig, resolve_path
from swarmui_clone.schemas import BackendStatus
from swarmui_clone.utils.pathing import ensure_directory

LOGGER = logging.getLogger("swarmui_clone.comfy")

FORWARDED_COMFY_FOLDERS = [
    "unet",
    "diffusion_models",
    "gligen",
    "ipadapter",
    "yolov8",
    "tensorrt",
    "clipseg",
    "style_models",
    "latent_upscale_models",
]

UPSCALE_FOLDERS = ["ESRGAN", "RealESRGAN", "SwinIR", "upscale-models", "upscale_models"]
MODULE_NOT_FOUND_PATTERN = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
IMPORT_ERROR_PATTERN = re.compile(r"ImportError: cannot import name ['\"]([^'\"]+)['\"]")


class ComfyProcessManager:
    def __init__(self, config_provider: Callable[[], AppConfig]) -> None:
        self._config_provider = config_provider
        self._status = "stopped"
        self._port: int | None = None
        self._process: subprocess.Popen[str] | None = None
        self._last_error: str | None = None
        self._expected_shutdown = False
        self._log_tail: deque[str] = deque(maxlen=500)
        self._log_lock = threading.Lock()
        self._lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._last_preflight: dict[str, Any] | None = None

    @property
    def api_url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}"

    @property
    def web_url(self) -> str | None:
        return self.api_url

    def _append_log(self, line: str) -> None:
        with self._log_lock:
            self._log_tail.append(line)
        LOGGER.debug("%s", line)

    @staticmethod
    def _tail_lines(text: str, limit: int = 60) -> list[str]:
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-limit:]

    @staticmethod
    def _discover_import_targets(start_script: Path) -> list[str]:
        try:
            source = start_script.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(start_script))
        except Exception:
            return []

        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        modules.add(alias.name.strip())
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue
                if node.module:
                    modules.add(node.module.strip())

        modules.discard("__future__")
        return sorted(name for name in modules if name)

    async def _run_import_probe(
        self,
        python_executable: str,
        workdir: Path,
        env: dict[str, str],
        modules: list[str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True,
            "command": [],
            "missing_modules": [],
            "import_errors": [],
            "errors": [],
            "output_tail": [],
            "timed_out": False,
            "return_code": None,
        }
        if not modules:
            return result

        script = (
            "import importlib.util\n"
            "import json\n"
            "modules = json.loads('''" + json.dumps(modules) + "''')\n"
            "missing = []\n"
            "import_errors = []\n"
            "for module_name in modules:\n"
            "    try:\n"
            "        found = importlib.util.find_spec(module_name)\n"
            "        if found is None:\n"
            "            missing.append(module_name)\n"
            "    except ModuleNotFoundError:\n"
            "        missing.append(module_name)\n"
            "    except Exception as ex:\n"
            "        import_errors.append(f\"{module_name}: {ex}\")\n"
            "print(json.dumps({\n"
            "    'missing_modules': sorted(set(missing)),\n"
            "    'import_errors': sorted(set(import_errors)),\n"
            "}))\n"
        )
        cmd = [python_executable, "-s", "-c", script]
        result["command"] = cmd

        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    cwd=str(workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            result["ok"] = False
            result["timed_out"] = True
            result["errors"].append(f"Import probe timed out after {timeout_seconds}s.")
            return result
        except Exception as ex:
            result["ok"] = False
            result["errors"].append(f"Import probe failed to run: {type(ex).__name__}: {ex}")
            return result

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = f"{stdout}\n{stderr}".strip()
        result["return_code"] = completed.returncode
        result["output_tail"] = self._tail_lines(combined)

        parsed: dict[str, Any] = {}
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
                break
            except Exception:
                continue

        result["missing_modules"] = sorted(set(parsed.get("missing_modules", [])))
        result["import_errors"] = sorted(set(parsed.get("import_errors", [])))

        if result["missing_modules"]:
            result["errors"].append(
                "Import probe missing python modules: " + ", ".join(result["missing_modules"])
            )
        if result["import_errors"]:
            result["errors"].append(
                "Import probe import errors: " + ", ".join(result["import_errors"])
            )
        if completed.returncode != 0 and not result["errors"]:
            result["errors"].append(f"Import probe command failed with return code {completed.returncode}.")

        result["ok"] = not result["errors"]
        return result

    async def preflight_check(self, timeout_seconds: int = 20) -> dict[str, Any]:
        cfg = self._config_provider()
        start_script = resolve_path(cfg.comfy.start_script)
        result: dict[str, Any] = {
            "ok": False,
            "checked_at": time.time(),
            "start_script": str(start_script),
            "python_executable": None,
            "command": [],
            "return_code": None,
            "timed_out": False,
            "missing_modules": [],
            "import_errors": [],
            "import_probe_modules": [],
            "import_probe_command": [],
            "import_probe_output_tail": [],
            "warnings": [],
            "errors": [],
            "output_tail": [],
        }

        if not start_script.exists():
            result["errors"].append(f"ComfyUI start script does not exist: {start_script}")
            self._last_preflight = result
            return result
        if start_script.suffix.lower() not in {".py", ".sh", ".bat"}:
            result["errors"].append(
                f"ComfyUI start script must end with .py, .sh, or .bat: {start_script.name}"
            )
            self._last_preflight = result
            return result

        env = os.environ.copy()
        self._clean_python_environment(env)
        if start_script.suffix.lower() == ".py":
            python_exe, script_arg, workdir = self._resolve_python_launch(start_script)
            result["python_executable"] = python_exe
            cmd = [python_exe, "-s", script_arg, "--help"]
        else:
            workdir = start_script.parent
            cmd = [str(start_script), "--help"]

        result["command"] = cmd
        LOGGER.debug("Running Comfy preflight: cwd=%s cmd=%s", workdir, " ".join(cmd))

        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    cwd=str(workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            result["timed_out"] = True
            result["errors"].append(
                f"Preflight command timed out after {timeout_seconds}s."
            )
            self._last_preflight = result
            return result
        except Exception as ex:
            result["errors"].append(f"Failed to run preflight command: {type(ex).__name__}: {ex}")
            self._last_preflight = result
            return result

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = f"{stdout}\n{stderr}".strip()
        result["return_code"] = completed.returncode
        result["output_tail"] = self._tail_lines(combined)
        result["missing_modules"] = sorted(set(MODULE_NOT_FOUND_PATTERN.findall(combined)))
        result["import_errors"] = sorted(set(IMPORT_ERROR_PATTERN.findall(combined)))

        if result["missing_modules"]:
            result["errors"].append(
                "Missing python modules: " + ", ".join(result["missing_modules"])
            )
        if result["import_errors"]:
            result["errors"].append(
                "Import errors detected: " + ", ".join(result["import_errors"])
            )

        if completed.returncode != 0 and not result["errors"]:
            if "Traceback (most recent call last)" in combined:
                result["errors"].append(
                    f"Preflight command failed with return code {completed.returncode}."
                )
            else:
                result["warnings"].append(
                    f"Preflight command returned {completed.returncode} without import errors."
                )

        if start_script.suffix.lower() == ".py":
            import_probe_modules = self._discover_import_targets(start_script)
            result["import_probe_modules"] = import_probe_modules
            import_probe = await self._run_import_probe(
                python_executable=python_exe,
                workdir=workdir,
                env=env,
                modules=import_probe_modules,
                timeout_seconds=timeout_seconds,
            )
            result["import_probe_command"] = import_probe["command"]
            result["import_probe_output_tail"] = import_probe["output_tail"]
            if import_probe["timed_out"]:
                result["errors"].append(f"Import probe timed out after {timeout_seconds}s.")
            if import_probe["missing_modules"]:
                result["missing_modules"] = sorted(
                    set(result["missing_modules"]).union(import_probe["missing_modules"])
                )
            if import_probe["import_errors"]:
                probe_import_errors = []
                for value in import_probe["import_errors"]:
                    probe_import_errors.append(value)
                    if ":" in value:
                        probe_import_errors.append(value.split(":", 1)[0].strip())
                result["import_errors"] = sorted(
                    set(result["import_errors"]).union(probe_import_errors)
                )
            result["errors"].extend(
                message
                for message in import_probe["errors"]
                if message not in result["errors"]
            )

        result["ok"] = not result["errors"]
        self._last_preflight = result
        return result

    def last_preflight(self) -> dict[str, Any] | None:
        if self._last_preflight is None:
            return None
        return copy.deepcopy(self._last_preflight)

    def _read_stream(self, stream: object, label: str) -> None:
        if stream is None:
            return
        for raw_line in iter(stream.readline, ""):
            line = raw_line.rstrip()
            if line:
                self._append_log(f"[{label}] {line}")
        try:
            stream.close()
        except Exception:
            pass

    @staticmethod
    def _is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _allocate_port(self, cfg: AppConfig) -> int:
        start = cfg.comfy.backend_starting_port
        if cfg.comfy.randomize_backend_port:
            start += random.randint(1, 200)
        port = start
        while not self._is_port_available(port):
            port += 1
        return port

    def _ensure_model_dirs(self, cfg: AppConfig) -> None:
        if not cfg.paths.model_roots:
            return
        root = resolve_path(cfg.paths.model_roots[0])
        ensure_directory(root / cfg.paths.clip_vision_folders[0])
        ensure_directory(root / cfg.paths.clip_folders[0])
        ensure_directory(root / "upscale_models")

    @staticmethod
    def _build_lines(root: Path, folders: list[str]) -> str:
        values: list[str] = []
        seen: set[str] = set()
        for folder in folders:
            candidate = Path(folder)
            full = (candidate if candidate.is_absolute() else root / candidate).resolve()
            as_text = str(full)
            if as_text not in seen:
                seen.add(as_text)
                values.append(as_text)
        return "\n".join(values)

    def emit_model_paths_config(self) -> Path:
        cfg = self._config_provider()
        self._ensure_model_dirs(cfg)

        data_root = ensure_directory(resolve_path(cfg.paths.data_root))
        payload: dict[str, dict[str, str]] = {}

        for index, root_text in enumerate(cfg.paths.model_roots):
            root = resolve_path(root_text)
            key = f"swarmui_model_paths_{index}"
            entry: dict[str, str] = {
                "checkpoints": self._build_lines(root, cfg.paths.sd_model_folders),
                "vae": self._build_lines(root, cfg.paths.vae_folders),
                "loras": self._build_lines(root, cfg.paths.lora_folders),
                "upscale_models": self._build_lines(root, UPSCALE_FOLDERS),
                "embeddings": self._build_lines(root, cfg.paths.embedding_folders),
                "hypernetworks": self._build_lines(root, ["hypernetworks"]),
                "controlnet": self._build_lines(root, cfg.paths.controlnet_folders),
                "model_patches": self._build_lines(root, cfg.paths.controlnet_folders),
                "clip": self._build_lines(root, cfg.paths.clip_folders),
                "clip_vision": self._build_lines(root, cfg.paths.clip_vision_folders),
            }
            for folder in FORWARDED_COMFY_FOLDERS:
                entry[folder] = self._build_lines(root, [folder])
            payload[key] = entry

        comfy_dir = resolve_path(cfg.comfy.start_script).parent
        custom_node_paths = [str((comfy_dir / "custom_nodes").resolve())]
        payload["swarmui_nodes"] = {"custom_nodes": "\n".join(custom_node_paths)}

        out_path = data_root / "comfy-auto-model.yaml"
        out_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return out_path

    @staticmethod
    def _clean_python_environment(env: dict[str, str]) -> None:
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)

    def _resolve_python_launch(self, start_script: Path) -> tuple[str, str, Path]:
        script_dir = start_script.parent
        if os.name == "nt":
            venv_python = script_dir / "venv" / "Scripts" / "python.exe"
            embedded_python = script_dir.parent / "python_embeded" / "python.exe"
            if venv_python.exists():
                return str(venv_python), str(start_script), script_dir
            if embedded_python.exists():
                workdir = script_dir.parent
                relative_script = os.path.relpath(start_script, workdir)
                return str(embedded_python), relative_script, workdir
            return "python", str(start_script), script_dir

        venv_python = script_dir / "venv" / "bin" / "python3"
        if venv_python.exists():
            return str(venv_python), str(start_script), script_dir
        return "python3", str(start_script), script_dir

    def _internal_args(self, cfg: AppConfig, model_yaml_path: Path) -> list[str]:
        if cfg.comfy.disable_internal_args:
            return []

        args: list[str] = ["--extra-model-paths-config", str(model_yaml_path)]

        if cfg.comfy.preview_method == "latent2rgb":
            args.extend(["--preview-method", "latent2rgb"])
        elif cfg.comfy.preview_method == "taesd":
            args.extend(["--preview-method", "taesd"])

        if cfg.comfy.frontend_version == "latest":
            args.extend(["--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"])
        elif cfg.comfy.frontend_version == "latest_swarm_validated":
            args.extend(
                [
                    "--front-end-version",
                    f"Comfy-Org/ComfyUI_frontend@v{cfg.comfy.swarm_validated_frontend_version}",
                ]
            )
        elif cfg.comfy.frontend_version == "legacy":
            args.extend(["--front-end-version", "Comfy-Org/ComfyUI_legacy_frontend@latest"])

        return args

    def _build_command(self, cfg: AppConfig, model_yaml_path: Path) -> tuple[list[str], Path, dict[str, str], int]:
        start_script = resolve_path(cfg.comfy.start_script)
        if not start_script.exists():
            raise FileNotFoundError(f"ComfyUI start script was not found: {start_script}")
        if start_script.suffix.lower() not in {".py", ".sh", ".bat"}:
            raise ValueError(
                f"ComfyUI start script must end in .py, .sh, or .bat, got: {start_script.name}"
            )

        port = self._allocate_port(cfg)
        env = os.environ.copy()
        self._clean_python_environment(env)
        env["CUDA_VISIBLE_DEVICES"] = cfg.comfy.gpu_id
        env["HIP_VISIBLE_DEVICES"] = cfg.comfy.gpu_id
        env["ROCR_VISIBLE_DEVICES"] = cfg.comfy.gpu_id

        internal_args = self._internal_args(cfg, model_yaml_path)
        user_args = shlex.split(cfg.comfy.extra_args.strip()) if cfg.comfy.extra_args.strip() else []

        if start_script.suffix.lower() == ".py":
            python_exe, script_arg, workdir = self._resolve_python_launch(start_script)
            cmd = [python_exe, "-s", script_arg, "--port", str(port), *internal_args, *user_args]
            return cmd, workdir, env, port

        cmd = [str(start_script), "--port", str(port), *internal_args, *user_args]
        return cmd, start_script.parent, env, port

    async def _health_check(self) -> bool:
        if not self.api_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                response = await client.get(f"{self.api_url}/object_info")
            return response.status_code == 200
        except Exception:
            return False

    async def _wait_until_ready(self, timeout_seconds: int) -> None:
        end_time = time.monotonic() + timeout_seconds
        while time.monotonic() < end_time:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI process exited unexpectedly with code {self._process.returncode}"
                )
            if await self._health_check():
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"ComfyUI did not become healthy within {timeout_seconds}s")

    async def _monitor_process_exit(self, process: subprocess.Popen[str]) -> None:
        return_code = await asyncio.to_thread(process.wait)
        should_restart = False
        cfg = self._config_provider()

        async with self._lock:
            if self._process is not process:
                return
            if self._expected_shutdown:
                self._append_log("[MANAGER] ComfyUI stopped")
                self._status = "stopped"
                self._process = None
                self._port = None
                return
            self._status = "error"
            self._last_error = f"ComfyUI exited unexpectedly with code {return_code}"
            self._append_log(f"[MANAGER] {self._last_error}")
            self._process = None
            self._port = None
            should_restart = cfg.comfy.auto_restart

        if should_restart:
            self._append_log("[MANAGER] Auto-restart enabled, restarting ComfyUI in 2s")
            await asyncio.sleep(2)
            try:
                await self.start()
            except Exception as ex:
                self._append_log(f"[MANAGER] Auto-restart failed: {ex}")

    async def start(self, run_preflight: bool = True) -> BackendStatus:
        async with self._lock:
            if self._process and self._process.poll() is None:
                return self.snapshot()

            cfg = self._config_provider()
            self._expected_shutdown = False
            self._status = "starting"
            self._last_error = None

            if run_preflight and cfg.comfy.enable_preflight_checks:
                preflight = await self.preflight_check(timeout_seconds=cfg.comfy.preflight_timeout_seconds)
                if not preflight["ok"]:
                    error_text = (
                        preflight["errors"][0]
                        if preflight.get("errors")
                        else "ComfyUI preflight failed."
                    )
                    self._status = "error"
                    self._last_error = f"Preflight failed: {error_text}"
                    self._append_log(f"[MANAGER] {self._last_error}")
                    raise RuntimeError(self._last_error)

            model_yaml = self.emit_model_paths_config()
            cmd, workdir, env, port = self._build_command(cfg, model_yaml)
            self._append_log(f"[MANAGER] Launching ComfyUI: {' '.join(cmd)}")
            LOGGER.debug("Launching ComfyUI from %s on port %s", workdir, port)

            process = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._process = process
            self._port = port

            threading.Thread(
                target=self._read_stream,
                args=(process.stdout, "STDOUT"),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_stream,
                args=(process.stderr, "STDERR"),
                daemon=True,
            ).start()

            self._monitor_task = asyncio.create_task(self._monitor_process_exit(process))

        try:
            await self._wait_until_ready(cfg.comfy.startup_timeout_seconds)
            async with self._lock:
                self._status = "running"
                self._last_error = None
            self._append_log("[MANAGER] ComfyUI backend is healthy")
            LOGGER.debug("ComfyUI backend healthy on %s", self.api_url)
        except Exception as ex:
            async with self._lock:
                self._status = "error"
                self._last_error = str(ex)
            LOGGER.exception("Failed to start ComfyUI backend")
            await self.stop()
            raise

        return self.snapshot()

    async def stop(self) -> BackendStatus:
        async with self._lock:
            process = self._process
            if not process:
                self._status = "stopped"
                self._port = None
                return self.snapshot()

            self._expected_shutdown = True
            self._append_log("[MANAGER] Stopping ComfyUI")
            LOGGER.debug("Stopping ComfyUI process pid=%s", process.pid)

        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.to_thread(process.wait)

        async with self._lock:
            self._process = None
            self._port = None
            self._status = "stopped"
            self._expected_shutdown = False

        return self.snapshot()

    async def restart(self) -> BackendStatus:
        await self.stop()
        return await self.start()

    async def ensure_running(self) -> BackendStatus:
        status = self.snapshot()
        if status.status == "running" and status.health_ok:
            return status
        return await self.start()

    def snapshot(self) -> BackendStatus:
        cfg = self._config_provider()
        process = self._process
        running = process is not None and process.poll() is None
        with self._log_lock:
            log_tail = list(self._log_tail)

        status = self._status
        if running and status != "starting":
            status = "running"

        return BackendStatus(
            status=status,  # type: ignore[arg-type]
            port=self._port if running else None,
            pid=process.pid if running else None,
            api_url=self.api_url if running else None,
            web_url=self.web_url if running else None,
            start_script=cfg.comfy.start_script,
            health_ok=running and status == "running",
            auto_restart=cfg.comfy.auto_restart,
            last_error=self._last_error,
            log_tail=log_tail,
        )
