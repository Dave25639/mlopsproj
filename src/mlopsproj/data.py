import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch
import typer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)
app = typer.Typer()

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
# Utils
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

def show_image_and_target(img, target, class_names):
    img = img * 0.5 + 0.5
    plt.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(class_names[target])
    plt.axis('off')

def build_transforms(train: bool = False):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if train:
        # Training with augmentations
        return transforms.Compose([
            transforms.Resize(256),                    # ADD THIS
            transforms.CenterCrop(224),                # ADD THIS
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

    # Validation/test without augmentations
    return transforms.Compose([
        transforms.Resize(256),                        # ADD THIS
        transforms.CenterCrop(224),                    # ADD THIS
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

@app.command()
def smoke_test(datadir: str = "data/") -> None:
    logger.info(f"Starting smoke test: {datadir}")

    dm = Food101DataModule(data_dir=datadir, batch_size=8, num_workers=0, augment_train=True, val_fraction=0.2, split_seed=42)
    dm.setup()

    xb, yb = next(iter(dm.train_dataloader()))
    # Use print for the "normal" look
    print(f"✅ Train Load Success | Batch Shape: {xb.shape}")
    print(f"✅ Labels: {yb.tolist()}")

    print(f"Dataset Classes: {dm.classes[:5]}")

@app.command()
def dataset_statistics(datadir: str = "data/", output_dir: str = "src/mlopsproj/outputs/cml") -> None:
    """Compute Food-101 dataset statistics and save visualizations."""
    import logging
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)

    # 1. Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing dataset statistics for: {datadir}")

    # 2. Initialize DataModule
    dm = Food101DataModule(data_dir=datadir, batch_size=32)
    dm.setup(stage="fit")
    dm.setup(stage="test")

    # 3. Prepare Markdown Report Content
    report = [
        "# FOOD-101 DATASET STATISTICS",
        f"- **Number of classes:** {dm.num_classes}",
        f"- **Train samples:** {len(dm.train_dataset)}",
        f"- **Val samples:** {len(dm.val_dataset)}",
        f"- **Test samples:** {len(dm.test_dataset)}",
        "\n## Visualizations"
    ]

    # -----------------------
    # Sample Images
    # -----------------------
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img, label = dm.train_dataset[random.randint(0, len(dm.train_dataset)-1)]
        plt.sca(ax)  # Set the current axis
        show_image_and_target(img, label, dm.classes)

    plt.tight_layout()
    sample_img_path = out_path / "food101_sample_images.png"
    plt.savefig(sample_img_path, dpi=150)
    plt.close()

    report.append(f"![Sample Images]({sample_img_path.name})")

    # -----------------------
    # Train Label Distribution
    # -----------------------
    train_labels = [label for _, label in dm.train_dataset.samples]
    plt.figure(figsize=(12, 4))
    plt.hist(train_labels, bins=range(dm.num_classes + 1), color='skyblue', edgecolor='black')
    plt.xticks(range(dm.num_classes), dm.classes, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Train Label Distribution")
    plt.tight_layout()
    train_dist_path = out_path / "train_label_distribution.png"
    plt.savefig(train_dist_path, dpi=150)
    plt.close()

    report.append(f"![Train Label Distribution]({train_dist_path.name})")

    # -----------------------
    # Write Markdown Report
    # -----------------------
    md_report_path = out_path / "data_statistics.md"
    md_report_path.write_text("\n".join(report))

    logger.info(f"✓ Statistics complete! Report saved to {md_report_path}")


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
        data_fraction: float = 1,
        split_seed: int = 42,
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
                random_state=self.hparams.split_seed
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
    app()
