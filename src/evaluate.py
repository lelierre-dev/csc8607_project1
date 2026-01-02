"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse

from typing import Any

import torch
import yaml
from torch import nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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
    return total_loss / max(1, total), correct / max(1, total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Dataloaders + meta
    _, _, test_loader, meta = get_dataloaders(config)
    config.setdefault("model", {})
    config["model"].setdefault("num_classes", meta.get("num_classes", 200))
    config["model"].setdefault("input_shape", list(meta.get("input_shape", (3, 64, 64))))

    device_str = get_device((config.get("train", {}) or {}).get("device", "auto"))
    device = torch.device(device_str)
    print(f"Device sélectionné: {device}")

    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    test_loss, test_acc = _evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()