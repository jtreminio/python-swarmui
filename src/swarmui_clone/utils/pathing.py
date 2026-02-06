from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_relative_path(base: Path, candidate: str) -> Path:
    clean = Path(candidate)
    combined = (base / clean).resolve()
    base_resolved = base.resolve()
    try:
        combined.relative_to(base_resolved)
    except ValueError:
        raise ValueError("Path escapes base directory")
    return combined
