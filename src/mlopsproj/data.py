import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytorch_lightning as L
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


# ---------------------------
# Custom Dataset wrapper
# ---------------------------

class FoodDataset(Dataset):
    """Simple wrapper around file/label pairs."""
    def __init__(self, file_label_pairs: List[Tuple[str, int]], classes: List[str], transform=None):
        self.samples = file_label_pairs
        self.classes = classes
        self.num_classes = len(classes)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Paths
# ---------------------------

@dataclass(frozen=True)
class DataPaths:
    data_dir: str

    @property
    def images_root(self) -> Path:
        return Path(self.data_dir) / "processed" / "images"

    @property
    def train_dir(self) -> Path:
        return self.images_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.images_root / "test"


# ---------------------------
# Transforms
# ---------------------------

def build_transforms(train: bool = False):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if train:
        # optional augmentations
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # default for val/test
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


# ---------------------------
# DataModule
# ---------------------------

class Food101DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        seed: int = 42,
        augment_train: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.paths = DataPaths(data_dir)

        # placeholders
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.classes: List[str] = []
        self.num_classes: int = 0

    def setup(self, stage: Optional[str] = None):
        logger.info("Setting up datasets")

        # ----------------- FULL TRAIN DATA -----------------
        full_train_dataset = ImageFolder(root=self.paths.train_dir)
        self.classes = full_train_dataset.classes
        self.num_classes = len(self.classes)

        train_tf = build_transforms(train=self.hparams.augment_train)
        val_tf = build_transforms(train=False)

        # get file/label pairs
        samples = full_train_dataset.imgs

        # ----------------- TRAIN / VAL SPLIT -----------------
        if stage in ("fit", None):
            labels = [label for _, label in samples]
            train_idx, val_idx = train_test_split(
                range(len(labels)),
                test_size=self.hparams.val_fraction,
                stratify=labels,
                random_state=self.hparams.seed
            )

            train_pairs = [samples[i] for i in train_idx]
            val_pairs = [samples[i] for i in val_idx]

            self.train_dataset = FoodDataset(train_pairs, self.classes, transform=train_tf)
            self.val_dataset = FoodDataset(val_pairs, self.classes, transform=val_tf)

        # ----------------- TEST DATA -----------------
        if stage in ("test", None):
            test_dataset = ImageFolder(root=self.paths.test_dir)
            self.test_dataset = FoodDataset(test_dataset.imgs, test_dataset.classes, transform=val_tf)

        logger.info(f"Classes ({self.num_classes}): {self.classes}")


    # ----------------- DATALOADERS -----------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )


# ---------------------------
# Smoke test
# ---------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    data_dir = "/workspaces/mlopsproj/data"

    dm = Food101DataModule(
        data_dir=data_dir,
        batch_size=8,
        num_workers=2,
        val_fraction=0.1,
        seed=42,
        augment_train=True
    )

    dm.setup()

    xb, yb = next(iter(dm.train_dataloader()))
    print("Train batch:", xb.shape, yb.shape)

    xb, yb = next(iter(dm.val_dataloader()))
    print("Val batch:", xb.shape, yb.shape)

    xb, yb = next(iter(dm.test_dataloader()))
    print("Test batch:", xb.shape, yb.shape)

    print("Classes:", dm.classes[:5])
