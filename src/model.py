"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class StageSpec:
    repeats: int
    channels: int


class TinyImageNetCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        stages: list[StageSpec],
    ):
        super().__init__()
        if len(stages) != 3:
            raise ValueError("Ce modèle attend exactement 3 stages.")

        c_in = input_channels
        blocks: list[nn.Module] = []
        for stage_idx, spec in enumerate(stages, start=1):
            for _ in range(spec.repeats):
                blocks.append(
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=spec.channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                blocks.append(nn.BatchNorm2d(spec.channels))
                blocks.append(nn.ReLU(inplace=True))
                c_in = spec.channels

            if stage_idx < 3:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(stages[-1].channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        return self.head(x)

def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config."""
    model_cfg: dict[str, Any] = (config or {}).get("model", {}) or {}
    num_classes = int(model_cfg.get("num_classes", 200))

    input_shape = model_cfg.get("input_shape") or [3, 64, 64]
    input_channels = int(input_shape[0])

    repeats = model_cfg.get("stage_repeats") or model_cfg.get("repeats") or [2, 2, 2]
    channels = model_cfg.get("stage_channels") or model_cfg.get("channels") or [64, 128, 256]
    if len(repeats) != 3 or len(channels) != 3:
        raise ValueError("model.stage_repeats et model.stage_channels doivent être des listes de longueur 3.")

    stages = [
        StageSpec(repeats=int(repeats[0]), channels=int(channels[0])),
        StageSpec(repeats=int(repeats[1]), channels=int(channels[1])),
        StageSpec(repeats=int(repeats[2]), channels=int(channels[2])),
    ]

    return TinyImageNetCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        stages=stages,
    )