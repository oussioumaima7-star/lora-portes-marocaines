"""Chargement des configurations YAML avec héritage simple (clé `inherit`)."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        base.update(cfg)
        cfg = base
    return cfg
