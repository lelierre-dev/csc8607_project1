"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import ensure_dir, get_device, save_config_snapshot, set_seed


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1)
    parser.add_argument("--num_iters", type=int, default=200)
    args = parser.parse_args()

    config = _load_config(args.config)
    train_cfg = (config.get("train", {}) or {})
    paths_cfg = (config.get("paths", {}) or {})

    set_seed(int(train_cfg.get("seed", 42)))
    runs_dir = ensure_dir(paths_cfg.get("runs_dir", "./runs"))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = str(Path(runs_dir) / f"lr_finder_{ts}")
    ensure_dir(run_dir)
    save_config_snapshot(config, run_dir)

    writer = SummaryWriter(log_dir=run_dir)

    device = torch.device(get_device(train_cfg.get("device", "auto")))
    train_loader, _, _, meta = get_dataloaders(config)

    config.setdefault("model", {})
    config["model"].setdefault("num_classes", meta.get("num_classes", 200))
    config["model"].setdefault("input_shape", list(meta.get("input_shape", (3, 64, 64))))
    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.min_lr))

    min_lr = float(args.min_lr)
    max_lr = float(args.max_lr)
    num_iters = int(args.num_iters)
    mult = (max_lr / min_lr) ** (1 / max(1, num_iters - 1))
    lr = min_lr

    model.train()
    it = 0
    data_iter = iter(train_loader)
    while it < num_iters:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        writer.add_scalar("lr_finder/loss", float(loss.item()), it)
        writer.add_scalar("lr_finder/lr", float(lr), it)

        lr *= mult
        it += 1

    writer.flush()
    writer.close()
    print(f"LR finder terminé. Logs: {run_dir}")

if __name__ == "__main__":
    main()