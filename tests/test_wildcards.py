from __future__ import annotations

from swarmui_clone.config import AppConfig
from swarmui_clone.services.wildcards import WildcardService


def test_wildcard_expansion_is_deterministic(tmp_path):
    (tmp_path / "animals.txt").write_text("cat\ndog\n", encoding="utf-8")
    (tmp_path / "colors.txt").write_text("red\nblue\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.paths.wildcards_root = str(tmp_path)
    service = WildcardService(lambda: cfg)

    one = service.expand("a __colors__ __animals__", seed=99)
    two = service.expand("a __colors__ __animals__", seed=99)

    assert one.text == two.text
    assert one.tokens == ["colors", "animals"]


def test_missing_wildcard_keeps_token(tmp_path):
    cfg = AppConfig()
    cfg.paths.wildcards_root = str(tmp_path)
    service = WildcardService(lambda: cfg)

    result = service.expand("__missing__ token", seed=1)

    assert result.text == "__missing__ token"
    assert result.tokens == []
