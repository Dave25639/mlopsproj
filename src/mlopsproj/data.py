import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

import pytorch_lightning as L
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    logger.debug(f"Read {len(lines)} lines from {path}")
    return lines


@dataclass(frozen=True)
class Food101Paths:
    data_dir: str

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_dir, "raw/images")

    @property
    def meta_dir(self) -> str:
        return os.path.join(self.data_dir, "meta")

    @property
    def classes_txt(self) -> str:
        return os.path.join(self.meta_dir, "classes.txt")

    def split_txt(self, split: str) -> str:
        return os.path.join(self.meta_dir, f"{split}.txt")


def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    """
    ViT-friendly transforms. Train uses random augmentations; val/test is deterministic.
    Normalization uses mean/std = 0.5 to align with common HF ViT preproc for this checkpoint.
    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if train:
        logger.debug(f"Building training transforms with img_size={img_size}")
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    logger.debug(f"Building evaluation transforms with img_size={img_size}")
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def deterministic_split_indices(
    n: int,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Returns (train_indices, val_indices) as deterministic shuffled split of range(n).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")
    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n - 1, n_val))  # ensure both splits non-empty

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    logger.info(
        f"Split {n} samples into {len(train_idx)} train and {len(val_idx)} val")
    return train_idx, val_idx

#TODO: pad with black pixels, scale down, then no need for transform
class Food101Dataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # TODO: load directly with numpy for efficiency
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class Food101DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        img_size: int = 224,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        split_seed: int = 42
    ):
        super().__init__()
        self.save_hyperparameters()
        self.paths = Food101Paths(data_dir)

    def setup(self, stage: Optional[str] = None):
        # 1. Load the master dataset (Logic inherited from ImageFolder)
        full_dataset = ImageFolder(root=self.paths.images_dir)

        # 2. Map meta files to master indices
        train_files = set(_read_lines(self.paths.split_txt("train")))
        test_files = set(_read_lines(self.paths.split_txt("test")))

        all_train_indices = [i for i, (path, _) in enumerate(full_dataset.imgs)
                             if self._get_rel_path(path) in train_files]
        test_indices = [i for i, (path, _) in enumerate(full_dataset.imgs)
                        if self._get_rel_path(path) in test_files]

        # 3. Create Splits
        if stage == "fit" or stage is None:
            # Deterministically split the train indices into train/val
            tr_local_idx, va_local_idx = deterministic_split_indices(
                len(all_train_indices), self.hparams.val_fraction, self.hparams.split_seed
            )

            # Map back to master dataset indices
            train_idx = [all_train_indices[i] for i in tr_local_idx]
            val_idx = [all_train_indices[i] for i in va_local_idx]

            self.train_dataset = Subset(full_dataset, train_idx)
            self.val_dataset = Subset(full_dataset, val_idx)

            # Use distinct transforms for train vs val
            self.train_dataset.dataset.transform = build_transforms(self.hparams.img_size, train=True)
            self.val_dataset.dataset.transform = build_transforms(self.hparams.img_size, train=False)

        if stage == "test" or stage is None:
            self.test_dataset = Subset(full_dataset, test_indices)
            self.test_dataset.dataset.transform = build_transforms(self.hparams.img_size, train=False)

    def _get_rel_path(self, full_path: str) -> str:
        rel = os.path.relpath(full_path, self.paths.images_dir)
        return os.path.splitext(rel)[0]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Usage:
    #   FOOD101_DIR=/path/to/food-101 python data.py
    data_dir = "/workspaces/mlopsproj/data"

    dm = Food101DataModule(
        data_dir=data_dir,
        batch_size=8,
        num_workers=2,
        img_size=224,
        val_fraction=0.1,
        split_seed=42,
    )
    dm.prepare_data()
    dm.setup()

    xb, yb = next(iter(dm.train_dataloader()))
    print("Train batch:", xb.shape, yb.shape, xb.dtype, yb.dtype)

    xb, yb = next(iter(dm.val_dataloader()))
    print("Val batch:", xb.shape, yb.shape, xb.dtype, yb.dtype)

    xb, yb = next(iter(dm.test_dataloader()))
    print("Test batch:", xb.shape, yb.shape, xb.dtype, yb.dtype)

    # type: ignore[attr-defined]
    print("First 5 classes:", dm.train_dataset.dataset.classes[:5])
