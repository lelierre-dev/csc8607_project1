"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation (vision) ou None."""
    augment: dict[str, Any] = (config or {}).get("augment", {}) or {}
    ops: list[Any] = []

    if augment.get("random_flip"):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    random_crop = augment.get("random_crop")
    # Support minimal: True => crop 64 avec padding 4, dict => paramètres
    if random_crop is True:
        ops.append(transforms.RandomCrop(size=64, padding=4))
    elif isinstance(random_crop, dict):
        size = random_crop.get("size", 64)
        padding = random_crop.get("padding", 0)
        pad_if_needed = bool(random_crop.get("pad_if_needed", False))
        ops.append(
            transforms.RandomCrop(
                size=size,
                padding=padding,
                pad_if_needed=pad_if_needed,
            )
        )

    color_jitter = augment.get("color_jitter")
    if isinstance(color_jitter, dict):
        ops.append(
            transforms.ColorJitter(
                brightness=color_jitter.get("brightness", 0.0),
                contrast=color_jitter.get("contrast", 0.0),
                saturation=color_jitter.get("saturation", 0.0),
                hue=color_jitter.get("hue", 0.0),
            )
        )
    elif color_jitter is True:
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

    return transforms.Compose(ops) if ops else None