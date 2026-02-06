#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
OPTIONAL_CUSTOM_NODE_PACKAGES = [
    "diffusers",
    "opencv-python-headless",
    "gguf",
    "PyWavelets",
    "fal-client",
    "imageio-ffmpeg",
]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    workdir = cwd or PROJECT_ROOT
    printable = " ".join(shlex.quote(arg) for arg in command)
    print(f"[setup] $ {printable}")
    subprocess.run(command, cwd=str(workdir), check=True)


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> Path:
    python_path = venv_python_path(venv_dir)
    if python_path.exists():
        print(f"[setup] Reusing virtual environment: {venv_dir}")
        return python_path
    print(f"[setup] Creating virtual environment: {venv_dir}")
    run_command([sys.executable, "-m", "venv", str(venv_dir)])
    return python_path


def install_project_dependencies(python_path: Path, with_dev: bool) -> None:
    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    target = ".[dev]" if with_dev else "."
    run_command([str(python_path), "-m", "pip", "install", "-e", target], cwd=PROJECT_ROOT)


def read_configured_comfy_start_script(python_path: Path) -> Path:
    helper_script = (
        "from swarmui_clone.config import app_root\n"
        "from swarmui_clone.services.settings_service import SettingsService, resolve_settings_path\n"
        "settings_path = resolve_settings_path(app_root())\n"
        "cfg = SettingsService(settings_path).load()\n"
        "print(cfg.comfy.start_script)\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", helper_script],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    start_script = result.stdout.strip()
    if not start_script:
        raise RuntimeError("Unable to read comfy.start_script from config.")
    return Path(start_script).expanduser()


def install_comfy_dependencies(python_path: Path, start_script: Path) -> None:
    comfy_dir = start_script.parent
    requirements_files = [
        comfy_dir / "requirements.txt",
        comfy_dir / "manager_requirements.txt",
    ]
    existing_files = [path for path in requirements_files if path.exists()]
    if not existing_files:
        print(
            "[setup] Skipping Comfy dependency install. No requirements files found under "
            f"{comfy_dir}"
        )
        return

    command = [str(python_path), "-m", "pip", "install"]
    for requirements_file in existing_files:
        command.extend(["-r", str(requirements_file)])
    run_command(command)


def install_custom_node_extras(python_path: Path) -> None:
    run_command([str(python_path), "-m", "pip", "install", *OPTIONAL_CUSTOM_NODE_PACKAGES])


def install_system_dependencies() -> None:
    if shutil.which("ffmpeg"):
        print("[setup] ffmpeg is already installed.")
        return

    system_name = platform.system()
    if system_name == "Darwin" and shutil.which("brew"):
        print("[setup] Installing system dependency: ffmpeg (Homebrew)")
        run_command(["brew", "install", "ffmpeg"])
        return

    if system_name == "Linux" and shutil.which("apt-get") and hasattr(os, "geteuid") and os.geteuid() == 0:
        print("[setup] Installing system dependency: ffmpeg (apt)")
        run_command(["apt-get", "update"])
        run_command(["apt-get", "install", "-y", "ffmpeg"])
        return

    print("[setup] ffmpeg is not installed and could not be auto-installed.")
    if system_name == "Darwin":
        print("[setup] Install manually with: brew install ffmpeg")
    elif system_name == "Linux":
        print("[setup] Install manually with: sudo apt-get install -y ffmpeg")
    else:
        print("[setup] Install ffmpeg using your platform package manager.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap python-swarmui dependencies on a fresh system."
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV_DIR),
        help="Virtual environment directory to create/use (default: .venv).",
    )
    parser.add_argument(
        "--with-dev",
        action="store_true",
        help="Install development dependencies from [dev] extras.",
    )
    parser.add_argument(
        "--skip-comfy",
        action="store_true",
        help="Skip installing ComfyUI requirements.txt and manager_requirements.txt.",
    )
    parser.add_argument(
        "--with-custom-node-extras",
        action="store_true",
        help="Install common optional custom-node Python dependencies.",
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Skip system dependency installation checks (ffmpeg).",
    )
    args = parser.parse_args()

    venv_dir = Path(args.venv).expanduser()
    python_path = ensure_venv(venv_dir)

    install_project_dependencies(python_path, with_dev=args.with_dev)

    if not args.skip_comfy:
        start_script = read_configured_comfy_start_script(python_path)
        install_comfy_dependencies(python_path, start_script)

    if args.with_custom_node_extras:
        install_custom_node_extras(python_path)

    if not args.skip_system_deps:
        install_system_dependencies()

    print("[setup] Complete.")
    print(f"[setup] Activate venv: source {venv_dir}/bin/activate")
    print("[setup] Start app: python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
