"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import count_parameters, ensure_dir, get_device, save_config_snapshot, set_seed


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@torch.no_grad()
def _eval_one_epoch(model: nn.Module, loader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_from_config(
    config: dict[str, Any],
    run_name: str | None = None,
    override_seed: int | None = None,
    overfit_small: bool = False,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    train_cfg: dict[str, Any] = (config or {}).get("train", {}) or {}
    paths_cfg: dict[str, Any] = (config or {}).get("paths", {}) or {}

    seed = override_seed if override_seed is not None else train_cfg.get("seed", 42)
    set_seed(int(seed))

    runs_dir = ensure_dir(paths_cfg.get("runs_dir", "./runs"))
    artifacts_dir = ensure_dir(paths_cfg.get("artifacts_dir", "./artifacts"))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"run_{ts}"
    run_dir = str(Path(runs_dir) / run_name)
    ensure_dir(run_dir)

    save_config_snapshot(config, run_dir)
    writer = SummaryWriter(log_dir=run_dir)

    # Device
    device_str = get_device(train_cfg.get("device", "auto"))
    device = torch.device(device_str)
    print(f"Device sélectionné: {device}")

    # Data
    train_loader, val_loader, _, meta = get_dataloaders(config)
    if overfit_small or bool(train_cfg.get("overfit_small", False)):
        # Sur-apprentissage sur un petit subset configurable
        from torch.utils.data import Subset

        overfit_size = int(train_cfg.get("overfit_small_size", 256))
        n = min(overfit_size, len(train_loader.dataset))
        idx = list(range(n))
        subset = Subset(train_loader.dataset, idx)
        train_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )

    # Optionnel: réduire la taille de la validation pour aller plus vite (debug)
    val_subset_size = train_cfg.get("val_subset_size", None)
    if val_subset_size is not None:
        from torch.utils.data import Subset

        n_val = min(int(val_subset_size), len(val_loader.dataset))
        val_loader = torch.utils.data.DataLoader(
            Subset(val_loader.dataset, list(range(n_val))),
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
        )

    print(f"Taille train utilisée: {len(train_loader.dataset)}")
    print(f"Taille val utilisée:   {len(val_loader.dataset)}")

    # Model
    if "model" not in config:
        config["model"] = {}
    config["model"].setdefault("num_classes", meta.get("num_classes", 200))
    config["model"].setdefault("input_shape", list(meta.get("input_shape", (3, 64, 64))))
    model = build_model(config).to(device)
    writer.add_text("model/summary", str(model))
    writer.add_scalar("model/params", count_parameters(model), 0)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optim
    optim_cfg = train_cfg.get("optimizer", {}) or {}
    opt_name = (optim_cfg.get("name") or "adam").lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    momentum = float(optim_cfg.get("momentum", 0.9))
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (minimal)
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_name = (sched_cfg.get("name") or "none").lower()
    scheduler = None
    if sched_name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "cosine":
        t_max = int(max_epochs or train_cfg.get("epochs", 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    # Loop
    epochs = int(max_epochs or train_cfg.get("epochs", 10))
    global_step = 0
    best_val_acc = -math.inf
    best_ckpt_path = str(Path(artifacts_dir) / "best.ckpt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = int(x.size(0))
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

            if max_steps is not None and global_step >= int(max_steps):
                break

        train_loss_epoch = running_loss / max(1, seen)
        val_loss, val_acc = _eval_one_epoch(model, val_loader, device, criterion)

        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "meta": meta,
                    "epoch": epoch,
                    "val_accuracy": float(val_acc),
                },
                best_ckpt_path,
            )

        if scheduler is not None:
            scheduler.step()

        if max_steps is not None and global_step >= int(max_steps):
            break

    writer.flush()
    writer.close()
    return {
        "run_dir": run_dir,
        "best_ckpt": best_ckpt_path,
        "best_val_accuracy": float(best_val_acc),
        "steps": int(global_step),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    # Ajoutez d'autres arguments si nécessaire (batch_size, lr, etc.)
    args = parser.parse_args()

    config = _load_config(args.config)
    run_name = args.run_name
    if run_name is None:
        try:
            user_input = input("Nom de run (ENTER pour auto): ").strip()
            if user_input:
                run_name = user_input
        except EOFError:
            pass

    result = train_from_config(
        config,
        run_name=run_name,
        override_seed=args.seed,
        overfit_small=bool(args.overfit_small),
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )
    print(
        f"Run: {result['run_dir']} | Best ckpt: {result['best_ckpt']} | Best val acc: {result['best_val_accuracy']:.4f}"
    )

if __name__ == "__main__":
    main()