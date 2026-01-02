"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


def to_tensor_if_needed(img):
    """Convertit une image PIL en Tensor uint8 (CxHxW), ou retourne le Tensor tel quel.

    Doit être au niveau module pour être picklable (DataLoader workers).
    """
    if isinstance(img, torch.Tensor):
        return img
    return F.pil_to_tensor(img)

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement (vision)."""
    preprocess: dict[str, Any] = (config or {}).get("preprocess", {}) or {}

    ops: list[Any] = []

    resize = preprocess.get("resize")
    if resize:
        # ex: [64, 64]
        ops.append(transforms.Resize(tuple(resize)))

    # PIL -> Tensor (uint8, CxHxW). Si l'entrée est déjà un Tensor, on le garde.
    ops.append(transforms.Lambda(to_tensor_if_needed))
    # uint8 -> float32, et remise à l'échelle [0, 1] automatiquement
    ops.append(transforms.ConvertImageDtype(torch.float32))

    normalize = preprocess.get("normalize")
    if normalize and isinstance(normalize, dict):
        mean = normalize.get("mean")
        std = normalize.get("std")
    else:
        mean = None
        std = None

    # Par défaut, stats ImageNet (convenable comme baseline)
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)