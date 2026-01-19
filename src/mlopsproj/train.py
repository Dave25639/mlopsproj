# train.py
"""
Training script for Food-101 classification using ViT.

Usage:
    python train.py --data_dir /path/to/food-101 --max_epochs 10
    python train.py --help  # see all options
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as L

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from .data import Food101DataModule
from .model import ViTClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ViT on Food-101",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to Food-101 dataset directory",
    )
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="DataLoader workers")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of train set to use for validation"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.1 = 10%% for quick testing)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze ViT backbone (only train classifier head)",
    )
    parser.add_argument("--dropout", type=float,
                        default=0.1, help="Classifier dropout")

    # Training arguments
    parser.add_argument("--max_epochs", type=int,
                        default=10, help="Maximum epochs")
    parser.add_argument("--learning_rate", type=float,
                        default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int,
                        default=500, help="Warmup steps")

    # Trainer arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator type"
    )
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices")
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps (simulate larger batch)",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )

    # Checkpointing and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="food101_vit",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (0 to disable)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_test",
        action="store_true",
        help="Run test set evaluation after training",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 batch for debugging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    L.seed_everything(args.seed, workers=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================
    # Data
    # ==================
    datamodule = Food101DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        data_fraction=args.data_fraction,
    )

    # Calculate max_steps for lr scheduler
    # (steps per epoch) * epochs
    datamodule.prepare_data()
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataloader()
                          ) // args.accumulate_grad_batches
    max_steps = steps_per_epoch * args.max_epochs

    # ==================
    # Model
    # ==================
    model = ViTClassifier(
        model_name=args.model_name,
        num_classes=datamodule.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    )

    # ==================
    # Callbacks
    # ==================
    callbacks = []

    # Model checkpointing (save best model based on validation accuracy)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val/acc",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Rich progress bar (prettier than default)
    callbacks.append(RichProgressBar())

    # ==================
    # Logger
    # ==================
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=args.experiment_name,
    )

    # ==================
    # Trainer
    # ==================
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        deterministic=True,  # reproducibility
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    # ==================
    # Training
    # ==================
    print("\n" + "=" * 80)
    print(f"Starting training: {args.experiment_name}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Data fraction: {args.data_fraction*100:.1f}%")
    print(
        f"Batch size: {args.batch_size} (accumulate: {args.accumulate_grad_batches})")
    print(
        f"Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Total steps: {max_steps}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=datamodule)

    # ==================
    # Testing
    # ==================
    if args.run_test:
        print("\n" + "=" * 80)
        print("Running test set evaluation...")
        print("=" * 80 + "\n")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # ==================
    # Summary
    # ==================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(
        f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"Logs saved to: {output_dir / 'logs'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
