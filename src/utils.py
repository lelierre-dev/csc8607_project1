"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python)."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproductibilité (au prix de perf potentielles)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto')."""
    prefer = (prefer or "auto").lower()
    if prefer in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if prefer in {"mps", "metal"}:
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if prefer == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (YAML) dans out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Nom stable + timestamp pour debug
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_file = out_path / f"config_snapshot_{ts}.yaml"
    with snapshot_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def ensure_dir(path: str | os.PathLike[str]) -> str:
    """Crée le dossier si nécessaire et retourne le chemin en str."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Fusionne récursivement updates dans base (copie)."""
    out: dict[str, Any] = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out