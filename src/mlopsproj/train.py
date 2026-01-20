# train.py
"""
Training script for Food-101 classification using ViT.

Features:
  - Python logging to file and console
  - W&B and TensorBoard logging
  - Multiple profiler options (simple, advanced, pytorch)
  - Custom callbacks for monitoring
  - Comprehensive metrics tracking

Usage:
    python train.py --data_dir /path/to/food-101 --max_epochs 10
    python train.py --data_dir /path/to/food-101 --use_wandb --profiler pytorch
    python train.py --help  # see all options
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler

from .callbacks import (
    DetailedProgressCallback,
    GradientLoggingCallback,
    LogPredictionsCallback,
    MetricsHistoryCallback,
)
from .data import Food101DataModule
from .model import ViTClassifier


def setup_logging(output_dir: Path, experiment_name: str, verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        output_dir: Directory to save log files
        experiment_name: Name of the experiment for log file
        verbose: If True, set DEBUG level, else INFO
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{experiment_name}.log"

    # Configure root logger
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Console handler (less detailed)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce noise from other libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ViT on Food-101 with comprehensive logging and profiling",
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

    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging (requires wandb login)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="food101-vit",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team) name",
    )
    parser.add_argument(
        "--log_predictions",
        action="store_true",
        help="Log sample predictions to W&B (only works with --use_wandb)",
    )
    parser.add_argument(
        "--log_gradients",
        action="store_true",
        help="Log gradient norms during training",
    )

    # Profiling arguments
    parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Profiler type (None to disable profiling)",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    return parser.parse_args()


def create_profiler(profiler_type: Optional[str], output_dir: Path):
    """
    Create profiler based on type.

    Args:
        profiler_type: Type of profiler ('simple', 'advanced', 'pytorch', or None)
        output_dir: Directory to save profiler output

    Returns:
        Profiler instance or None
    """
    if profiler_type is None:
        return None

    profiler_dir = output_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    if profiler_type == "simple":
        logger.info("Using SimpleProfiler")
        return SimpleProfiler(
            dirpath=str(profiler_dir),
            filename="simple_profile"
        )

    elif profiler_type == "advanced":
        logger.info("Using AdvancedProfiler")
        from pytorch_lightning.profilers import AdvancedProfiler
        return AdvancedProfiler(
            dirpath=str(profiler_dir),
            filename="advanced_profile"
        )

    elif profiler_type == "pytorch":
        logger.info("Using PyTorchProfiler")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            logger.info("CUDA profiling enabled")

        return PyTorchProfiler(
            dirpath=str(profiler_dir),
            filename="pytorch_profile",
            group_by_input_shapes=True,
            emit_nvtx=torch.cuda.is_available(),
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profiler_dir)),
            with_stack=True,
        )

    return None


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging FIRST
    setup_logging(output_dir, args.experiment_name, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info("=" * 80)

    # Log all arguments
    logger.info("Configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Set seed for reproducibility
    L.seed_everything(args.seed, workers=True)
    logger.info(f"Random seed set to {args.seed}")

    # ==================
    # Data
    # ==================
    logger.info("Initializing datamodule...")
    datamodule = Food101DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        data_fraction=args.data_fraction,
    )

    # Calculate max_steps for lr scheduler
    logger.info("Preparing data and calculating training steps...")
    datamodule.prepare_data()
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataloader()
                          ) // args.accumulate_grad_batches
    max_steps = steps_per_epoch * args.max_epochs

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {max_steps}")

    # ==================
    # Model
    # ==================
    logger.info("Initializing model...")
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
    logger.info("Setting up callbacks...")
    callbacks = []

    # Model checkpointing
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
    logger.info("Added ModelCheckpoint callback")

    # Early stopping
    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val/acc",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(
            f"Added EarlyStopping callback (patience={args.early_stopping_patience})")

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    logger.info("Added LearningRateMonitor callback")

    # Rich progress bar
    callbacks.append(RichProgressBar())

    # Metrics history
    metrics_history = MetricsHistoryCallback()
    callbacks.append(metrics_history)
    logger.info("Added MetricsHistoryCallback")

    # Detailed progress
    callbacks.append(DetailedProgressCallback())
    logger.info("Added DetailedProgressCallback")

    # Gradient logging
    if args.log_gradients:
        grad_callback = GradientLoggingCallback(log_every_n_steps=50)
        callbacks.append(grad_callback)
        logger.info("Added GradientLoggingCallback")

    # Prediction logging (W&B only)
    if args.log_predictions and args.use_wandb:
        pred_callback = LogPredictionsCallback(
            num_samples=8, log_every_n_epochs=1)
        callbacks.append(pred_callback)
        logger.info("Added LogPredictionsCallback")
    elif args.log_predictions and not args.use_wandb:
        logger.warning(
            "--log_predictions requires --use_wandb, skipping prediction logging")

    # ==================
    # Logger
    # ==================
    logger.info("Setting up experiment logger...")
    if args.use_wandb:
        try:
            import wandb

            # Check if logged in
            if not wandb.api.api_key:
                logger.error(
                    "W&B API key not found. Please run 'wandb login' first.")
                logger.info("Falling back to TensorBoard")
                args.use_wandb = False
            else:
                exp_logger = WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.experiment_name,
                    save_dir=output_dir / "logs",
                    log_model="all",  # Log model checkpoints
                )

                # Log hyperparameters
                exp_logger.experiment.config.update({
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "model_name": args.model_name,
                    "data_fraction": args.data_fraction,
                    "freeze_backbone": args.freeze_backbone,
                    "max_epochs": args.max_epochs,
                    "warmup_steps": args.warmup_steps,
                    "dropout": args.dropout,
                })

                logger.info(
                    f"Using W&B logger (project: {args.wandb_project})")

        except ImportError:
            logger.error(
                "wandb not installed. Install with: pip install wandb")
            logger.info("Falling back to TensorBoard")
            args.use_wandb = False

    if not args.use_wandb:
        exp_logger = TensorBoardLogger(
            save_dir=output_dir / "logs",
            name=args.experiment_name,
        )
        logger.info("Using TensorBoard logger")

    # ==================
    # Profiler
    # ==================
    profiler = create_profiler(args.profiler, output_dir)
    if profiler:
        logger.info(f"Profiling enabled: {args.profiler}")
    else:
        logger.info("Profiling disabled")

    # ==================
    # Trainer
    # ==================
    logger.info("Creating trainer...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=exp_logger,
        profiler=profiler,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    # ==================
    # Training Summary
    # ==================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: Food-101 ({args.data_fraction*100:.1f}% of data)")
    logger.info(f"Num classes: {datamodule.num_classes}")
    logger.info(f"Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"Val samples: {len(datamodule.val_dataset)}")
    logger.info(f"Test samples: {len(datamodule.test_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.accumulate_grad_batches}")
    logger.info(
        f"Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Total steps: {max_steps}")
    logger.info(f"Freeze backbone: {args.freeze_backbone}")
    logger.info(f"Accelerator: {args.accelerator}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Logger: {'W&B' if args.use_wandb else 'TensorBoard'}")
    logger.info(f"Profiler: {args.profiler if args.profiler else 'None'}")
    logger.info("=" * 80 + "\n")

    # ==================
    # Training
    # ==================
    try:
        logger.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    # ==================
    # Testing
    # ==================
    if args.run_test:
        logger.info("\n" + "=" * 80)
        logger.info("Running test set evaluation...")
        logger.info("=" * 80 + "\n")
        try:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")
            logger.info("Testing completed successfully!")
        except Exception as e:
            logger.error(f"Testing failed with error: {e}", exc_info=True)

    # ==================
    # Summary
    # ==================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(
        f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Logs saved to: {output_dir / 'logs'}")
    logger.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")

    if profiler:
        logger.info(f"Profiler output saved to: {output_dir / 'profiler'}")

    if args.use_wandb:
        logger.info(f"View results at: {exp_logger.experiment.url}")

    logger.info("=" * 80 + "\n")

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
