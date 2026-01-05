"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms.functional import pil_to_tensor

from src.augmentation import get_augmentation_transforms
from src.preporcessing import get_preprocess_transforms


def _pick_split_name(available: list[str], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in available:
        return preferred
    return None


def _guess_train_split(available: list[str]) -> Optional[str]:
    for cand in ["train", "training"]:
        if cand in available:
            return cand
    return available[0] if available else None


def _guess_val_split(available: list[str]) -> Optional[str]:
    for cand in ["validation", "valid", "val", "dev"]:
        if cand in available:
            return cand
    return None


def _guess_test_split(available: list[str]) -> Optional[str]:
    for cand in ["test", "testing"]:
        if cand in available:
            return cand
    return None


@dataclass
class HFVisionItemSpec:
    image_key: str = "image"
    label_key: str = "label"


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, transform=None, item_spec: HFVisionItemSpec | None = None):
        self.hf_ds = hf_ds
        self.transform = transform
        self.item_spec = item_spec or HFVisionItemSpec()

        # Prépare un mapping str->int si nécessaire
        self._label_str_to_int: dict[str, int] | None = None
        try:
            feat = getattr(hf_ds, "features", None)
            if feat and self.item_spec.label_key in feat:
                label_feat = feat[self.item_spec.label_key]
                names = getattr(label_feat, "names", None)
                if names:
                    self._label_str_to_int = {name: i for i, name in enumerate(names)}
        except Exception:
            self._label_str_to_int = None

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        item = self.hf_ds[idx]
        img = item.get(self.item_spec.image_key)
        y = item.get(self.item_spec.label_key)

        # HF peut renvoyer une image PIL directement, ou une structure avec path
        if isinstance(img, dict) and "path" in img:
            from PIL import Image

            img = Image.open(img["path"])

        if hasattr(img, "convert"):
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(y, str) and self._label_str_to_int is not None:
            y = self._label_str_to_int[y]

        y = int(y)
        return img, y


class CachedVisionDataset(Dataset):
    """Met en cache en RAM les images (Tensor uint8 CxHxW) et labels (int).

    Important: on cache *avant* augmentation/normalisation pour garder l'aléa d'augment.
    """

    def __init__(self, base: Dataset, transform=None, max_items: int | None = None):
        self.transform = transform

        n = len(base)
        if max_items is not None:
            n = min(n, int(max_items))

        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for i in range(n):
            img, y = base[i]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            # PIL -> uint8 tensor
            if isinstance(img, torch.Tensor):
                x = img
            else:
                x = pil_to_tensor(img)
            xs.append(x.contiguous())
            ys.append(int(y))

        self._x = xs
        self._y = torch.tensor(ys, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int):
        x = self._x[idx]
        y = int(self._y[idx].item())
        if self.transform is not None:
            x = self.transform(x)
        return x, y

def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    À implémenter.
    """
    from datasets import load_dataset
    from datasets.download.download_config import DownloadConfig

    dataset_cfg: dict[str, Any] = (config or {}).get("dataset", {}) or {}
    train_cfg: dict[str, Any] = (config or {}).get("train", {}) or {}

    dataset_name = dataset_cfg.get("name") or "zh-plus/tiny-imagenet"
    cache_root = dataset_cfg.get("root") or "./data"
    cache_dir = str(Path(cache_root).expanduser())

    keep_in_memory = dataset_cfg.get("keep_in_memory", None)
    local_files_only = bool(dataset_cfg.get("local_files_only", False))
    download_config = DownloadConfig(cache_dir=cache_dir, local_files_only=local_files_only)

    hf_ds = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        download_config=download_config,
        download_mode="reuse_cache_if_exists",
    )
    available_splits = list(hf_ds.keys())

    split_cfg = dataset_cfg.get("split") or {}
    train_split = _pick_split_name(available_splits, split_cfg.get("train")) or _guess_train_split(available_splits)
    val_split = _pick_split_name(available_splits, split_cfg.get("val")) or _guess_val_split(available_splits)
    test_split = _pick_split_name(available_splits, split_cfg.get("test")) or _guess_test_split(available_splits)

    if train_split is None:
        raise ValueError(f"Impossible de déterminer le split train. Splits disponibles: {available_splits}")

    # Tiny ImageNet : généralement train/val séparés; si test absent, on renvoie val comme test.
    if val_split is None:
        val_split = train_split
    if test_split is None:
        test_split = val_split

    preprocess_tf = get_preprocess_transforms(config)
    augment_tf = get_augmentation_transforms(config)

    # Important: augmentation avant ToTensor/Normalize (dans preprocess)
    if augment_tf is not None:
        from torchvision import transforms

        train_tf = transforms.Compose([augment_tf, preprocess_tf])
    else:
        train_tf = preprocess_tf

    eval_tf = preprocess_tf

    item_spec = HFVisionItemSpec(image_key="image", label_key="label")

    cache_cfg = bool(dataset_cfg.get("cache_in_ram", False))
    cache_max_items = dataset_cfg.get("cache_max_items", None)

    base_train = HFDatasetWrapper(hf_ds[train_split], transform=None, item_spec=item_spec)
    base_val = HFDatasetWrapper(hf_ds[val_split], transform=None, item_spec=item_spec)
    base_test = HFDatasetWrapper(hf_ds[test_split], transform=None, item_spec=item_spec)

    if cache_cfg:
        # Estimation mémoire (uint8): 3*64*64 ~ 12KB / image (+ overhead Python)
        print("[data] Cache en RAM activé (uint8). Cela peut consommer plusieurs Go selon la machine.")
        train_ds = CachedVisionDataset(base_train, transform=train_tf, max_items=cache_max_items)
        val_ds = CachedVisionDataset(base_val, transform=eval_tf, max_items=cache_max_items)
        test_ds = CachedVisionDataset(base_test, transform=eval_tf, max_items=cache_max_items)
    else:
        train_ds = HFDatasetWrapper(hf_ds[train_split], transform=train_tf, item_spec=item_spec)
        val_ds = HFDatasetWrapper(hf_ds[val_split], transform=eval_tf, item_spec=item_spec)
        test_ds = HFDatasetWrapper(hf_ds[test_split], transform=eval_tf, item_spec=item_spec)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    shuffle = bool(dataset_cfg.get("shuffle", True))
    pin_memory = bool(torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Meta minimal demandé par le template
    meta = {
        "num_classes": 200,
        "input_shape": (3, 64, 64),
        "splits": {"train": train_split, "val": val_split, "test": test_split},
        "dataset_name": dataset_name,
    }
    return train_loader, val_loader, test_loader, meta