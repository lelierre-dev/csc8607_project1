"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse

import itertools
from datetime import datetime
from typing import Any

import yaml
from torch.utils.tensorboard import SummaryWriter

from src.train import train_from_config
from src.utils import deep_update, ensure_dir


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_hparam_grid(hparams: dict[str, list[Any]]):
    keys = list(hparams.keys())
    values = [hparams[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=True))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    base_config = _load_config(args.config)
    hparams = (base_config.get("hparams", {}) or {})
    if not hparams:
        raise ValueError("Aucun hyperparamètre trouvé dans la section 'hparams' de la config.")

    paths_cfg = (base_config.get("paths", {}) or {})
    runs_dir = ensure_dir(paths_cfg.get("runs_dir", "./runs"))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_dir = ensure_dir(f"{runs_dir}/grid_search_{ts}")
    summary_writer = SummaryWriter(log_dir=summary_dir)

    best = {"acc": -1.0, "run": None, "hparams": None}

    for i, hp in enumerate(_iter_hparam_grid(hparams), start=1):
        # Convention: on supporte lr/batch_size/weight_decay au niveau train.optimizer/train.batch_size
        updates: dict[str, Any] = {}
        if "lr" in hp:
            updates = deep_update(updates, {"train": {"optimizer": {"lr": float(hp["lr"])}}})
        if "weight_decay" in hp:
            updates = deep_update(
                updates, {"train": {"optimizer": {"weight_decay": float(hp["weight_decay"])}}}
            )
        if "batch_size" in hp:
            updates = deep_update(updates, {"train": {"batch_size": int(hp["batch_size"])}})

        # Permet aussi de tuner le modèle directement si présent: stage_repeats / stage_channels
        if "stage_repeats" in hp:
            updates = deep_update(updates, {"model": {"stage_repeats": hp["stage_repeats"]}})
        if "stage_channels" in hp:
            updates = deep_update(updates, {"model": {"stage_channels": hp["stage_channels"]}})

        cfg = deep_update(base_config, updates)
        run_name = f"gs_{i:03d}_" + "_".join([f"{k}={hp[k]}" for k in sorted(hp.keys())])
        result = train_from_config(cfg, run_name=run_name, max_epochs=int(args.epochs), max_steps=args.max_steps)
        acc = float(result["best_val_accuracy"])

        # Log HParams (affichage dans TensorBoard)
        summary_writer.add_hparams(
            {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in hp.items()},
            {"best_val_accuracy": acc},
            run_name=run_name,
        )

        if acc > best["acc"]:
            best = {"acc": acc, "run": result["run_dir"], "hparams": hp}

        print(f"[{i}] acc={acc:.4f} run={result['run_dir']}")

    summary_writer.flush()
    summary_writer.close()
    print(f"Meilleur run: acc={best['acc']:.4f} run={best['run']} hparams={best['hparams']}")

if __name__ == "__main__":
    main()