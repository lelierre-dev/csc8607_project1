import yaml, torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from src.preporcessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms

torch.manual_seed(42)

cfg = yaml.safe_load(open("configs/config.yaml"))
dataset_cfg = cfg.get("dataset", {}) or {}
cache_dir = dataset_cfg.get("root", "./data")
name = dataset_cfg.get("name", "zh-plus/tiny-imagenet")
train_split = (dataset_cfg.get("split") or {}).get("train", "train")

hf = load_dataset(name, cache_dir=cache_dir,
                  download_config=DownloadConfig(cache_dir=cache_dir),
                  download_mode="reuse_cache_if_exists")
ds = hf[train_split]

pre_tf = get_preprocess_transforms(cfg)
aug_tf = get_augmentation_transforms(cfg)
train_tf = transforms.Compose([aug_tf, pre_tf]) if aug_tf else pre_tf

idx = list(range(16))
imgs = [ds[i]["image"] for i in idx]

xs_noaug = torch.stack([pre_tf(img) for img in imgs])
xs_aug   = torch.stack([train_tf(img) for img in imgs])

mean = cfg["preprocess"]["normalize"]["mean"] or [0.485,0.456,0.406]
std  = cfg["preprocess"]["normalize"]["std"]  or [0.229,0.224,0.225]
mean = torch.tensor(mean)[:,None,None]
std  = torch.tensor(std)[:,None,None]
def unnorm(x): return (x*std)+mean

out = Path("artifacts"); out.mkdir(exist_ok=True)
save_image(make_grid(unnorm(xs_noaug).clamp(0,1), nrow=8), out/"examples_noaug_same.png")
save_image(make_grid(unnorm(xs_aug).clamp(0,1), nrow=8), out/"examples_aug_same.png")
print("OK -> artifacts/examples_noaug_same.png, artifacts/examples_aug_same.png")