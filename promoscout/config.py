from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(p.parent)
    return cfg


def load_aliases(cfg: dict) -> tuple[dict[str, str], set[str]]:
    p = Path(cfg.get("_root", ROOT)) / cfg.get("aliases_file", "aliases.yaml")
    if not p.exists():
        return {}, set()
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return {k.upper(): v for k, v in (d.get("canonical") or {}).items()}, {x.upper() for x in d.get("hidden") or []}


def fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"
