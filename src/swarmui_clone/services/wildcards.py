from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from swarmui_clone.config import resolve_path

WILDCARD_PATTERN = re.compile(r"__([A-Za-z0-9_\-/]+)__")


@dataclass
class WildcardExpansion:
    text: str
    tokens: list[str]


class WildcardService:
    def __init__(self, config_provider) -> None:
        self._config_provider = config_provider

    @property
    def root(self) -> Path:
        cfg = self._config_provider()
        return resolve_path(cfg.paths.wildcards_root)

    def list_wildcards(self) -> list[str]:
        if not self.root.exists():
            return []
        names: list[str] = []
        for path in self.root.rglob("*.txt"):
            names.append(path.relative_to(self.root).with_suffix("").as_posix())
        return sorted(names)

    def _read_options(self, token: str) -> list[str]:
        wildcard_file = (self.root / f"{token}.txt").resolve()
        if not wildcard_file.exists():
            return []
        lines = wildcard_file.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    def expand(self, text: str, seed: int, max_depth: int = 10) -> WildcardExpansion:
        if not text:
            return WildcardExpansion(text="", tokens=[])

        rng = random.Random(seed)
        used: list[str] = []
        current = text

        for _ in range(max_depth):
            matches = list(WILDCARD_PATTERN.finditer(current))
            if not matches:
                break
            updated = current
            for match in matches:
                token = match.group(1)
                options = self._read_options(token)
                if not options:
                    continue
                replacement = rng.choice(options)
                updated = updated.replace(match.group(0), replacement, 1)
                used.append(token)
            if updated == current:
                break
            current = updated

        return WildcardExpansion(text=current, tokens=used)
