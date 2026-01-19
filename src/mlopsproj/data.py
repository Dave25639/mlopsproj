# data.py
"""
Local Food-101 loader (images + metadata) for PyTorch Lightning.

Your requested split policy:
  - Use meta/train.txt for BOTH train and val (we create a deterministic split from it)
  - Use meta/test.txt for test only

Expected folder layout (Food-101 style):
    <data_dir>/
      images/
        apple_pie/xxxx.jpg
        baby_back_ribs/yyyy.jpg
        ...
      meta/
        classes.txt
        train.txt
        test.txt

Where meta/train.txt and meta/test.txt contain one sample per line like:
    apple_pie/1005649
    baby_back_ribs/100102
(i.e., relative path under images/ WITHOUT the .jpg extension)

This file provides:
  - Food101Dataset: torch Dataset yielding (pixel_values, label)
  - Food101DataModule: Lightning DataModule yielding train/val/test DataLoaders

Notes:
  - Augmentations are applied on-the-fly for train only.
  - Val/test are deterministic.
  - Normalization uses mean/std=(0.5,0.5,0.5) to match common HF ViT preproc for
    'google/vit-base-patch16-224-in21k'.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import pytorch_lightning as L


# -----------------------
# Helpers
# -----------------------

def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


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
    return train_idx, val_idx


# -----------------------
# Dataset
# -----------------------

class Food101Dataset(Dataset):
    """
    Base dataset that indexes Food-101 samples from meta/<split>.txt.

    If 'sample_list' is provided, it overrides reading meta/<split>.txt and is treated as the
    list of relative sample paths (e.g., 'apple_pie/1005649').

    Returns:
        (pixel_values, label) where:
          - pixel_values: FloatTensor [3, H, W]
          - label: LongTensor scalar (class index)
    """

    def __init__(
        self,
        data_dir: str,
        # "train" or "test" (optional if sample_list provided)
        split: Optional[str] = None,
        sample_list: Optional[Sequence[str]] = None,
        transform: Optional[transforms.Compose] = None,
        image_ext: str = ".jpg",
        strict_files: bool = True,
    ):
        self.paths = Food101Paths(data_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.strict_files = strict_files

        # 1) Class mapping
        self.classes: List[str] = _read_lines(self.paths.classes_txt)
        if not self.classes:
            raise ValueError(f"No classes found in {self.paths.classes_txt}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 2) Sample list
        if sample_list is None:
            if split is None:
                raise ValueError(
                    "Provide either split='train'/'test' or sample_list=[...].")
            split_file = self.paths.split_txt(split)
            rel_samples = _read_lines(split_file)
        else:
            rel_samples = list(sample_list)

        if not rel_samples:
            raise ValueError("No samples found for dataset.")

        # 3) Build (img_path, label_idx)
        items: List[Tuple[str, int]] = []
        missing: List[str] = []

        for rel in rel_samples:
            cls = rel.split("/", 1)[0]
            if cls not in self.class_to_idx:
                raise ValueError(f"Class '{cls}' not found in classes.txt")
            label = self.class_to_idx[cls]
            img_path = os.path.join(
                self.paths.images_dir, rel + self.image_ext)

            if self.strict_files and (not os.path.isfile(img_path)):
                missing.append(img_path)
                continue

            items.append((img_path, label))

        if self.strict_files and missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} image files referenced by metadata were not found. "
                f"First missing paths:\n{preview}"
            )

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            pixel_values = transforms.ToTensor()(img)

        return pixel_values, torch.tensor(label, dtype=torch.long)


# -----------------------
# Lightning DataModule
# -----------------------

class Food101DataModule(L.LightningDataModule):
    """
    Split policy (as requested):
      - meta/train.txt -> split into train + val (deterministic, seedable)
      - meta/test.txt  -> test only
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 224,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
        strict_files: bool = True,
        data_fraction: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.strict_files = strict_files
        self.data_fraction = data_fraction

        self.num_classes: int = 101

        self.train_ds: Optional[Food101Dataset] = None
        self.val_ds: Optional[Food101Dataset] = None
        self.test_ds: Optional[Food101Dataset] = None

    def prepare_data(self):
        # Nothing to download (data is local), but we can validate structure lightly.
        paths = Food101Paths(self.data_dir)
        for p in [paths.images_dir, paths.meta_dir, paths.classes_txt, paths.split_txt("train"), paths.split_txt("test")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Expected path not found: {p}")

    def setup(self, stage: Optional[str] = None):
        # Read the canonical train/test lists once here
        paths = Food101Paths(self.data_dir)
        train_list = _read_lines(paths.split_txt("train"))
        test_list = _read_lines(paths.split_txt("test"))

        # Subset data if data_fraction < 1.0 (for quick testing)
        if self.data_fraction < 1.0:
            n_train = max(1, int(len(train_list) * self.data_fraction))
            n_test = max(1, int(len(test_list) * self.data_fraction))

            # Use deterministic subset
            g = torch.Generator()
            g.manual_seed(self.split_seed)

            train_indices = torch.randperm(len(train_list), generator=g)[
                :n_train].tolist()
            test_indices = torch.randperm(len(test_list), generator=g)[
                :n_test].tolist()

            train_list = [train_list[i] for i in train_indices]
            test_list = [test_list[i] for i in test_indices]

            print(
                f"Using {self.data_fraction*100:.1f}% of data: {len(train_list)} train samples, {len(test_list)} test samples")

        # Deterministically split train_list -> train/val
        tr_idx, va_idx = deterministic_split_indices(
            n=len(train_list),
            val_fraction=self.val_fraction,
            seed=self.split_seed,
        )
        train_samples = [train_list[i] for i in tr_idx]
        val_samples = [train_list[i] for i in va_idx]

        train_tf = build_transforms(self.img_size, train=True)
        eval_tf = build_transforms(self.img_size, train=False)

        # Build datasets
        self.train_ds = Food101Dataset(
            data_dir=self.data_dir,
            sample_list=train_samples,
            transform=train_tf,
            strict_files=self.strict_files,
        )
        self.val_ds = Food101Dataset(
            data_dir=self.data_dir,
            sample_list=val_samples,
            transform=eval_tf,
            strict_files=self.strict_files,
        )
        self.test_ds = Food101Dataset(
            data_dir=self.data_dir,
            sample_list=test_list,
            transform=eval_tf,
            strict_files=self.strict_files,
        )

        self.num_classes = len(self.train_ds.classes)

    def _dl(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup() before requesting dataloaders."
        return self._dl(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup() before requesting dataloaders."
        return self._dl(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup() before requesting dataloaders."
        return self._dl(self.test_ds, shuffle=False)


if __name__ == "__main__":
    # Usage:
    #   FOOD101_DIR=/path/to/food-101 python data.py
    data_dir = os.environ.get("FOOD101_DIR", "/workspaces/mlopsproj/data")

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

    print("Num classes:", dm.num_classes)
    # type: ignore[attr-defined]
    print("First 5 classes:", dm.train_ds.classes[:5])
