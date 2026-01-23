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

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

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

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging FIRST
    setup_logging(output_dir, cfg.experiment_name, cfg.logging.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"Starting experiment: {cfg.experiment_name}")
    logger.info("=" * 80)

    # Log all arguments
    logger.info("Configuration:")
    for arg, value in sorted(vars(cfg).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    logger.info(f"Random seed set to {cfg.seed}")

    # ==================
    # Data
    # ==================
    logger.info("Initializing datamodule...")
    datamodule = Food101DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.seed,
        data_fraction=cfg.data.data_fraction,
    )

    # Calculate max_steps for lr scheduler
    logger.info("Preparing data and calculating training steps...")
    datamodule.prepare_data()
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataloader()) // cfg.train.accumulate_grad_batches
    max_steps = steps_per_epoch * cfg.train.max_epochs

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {max_steps}")

    # ==================
    # Model
    # ==================
    logger.info("Initializing model...")
    model = ViTClassifier(
        model_name=cfg.model.architecture.model_name,
        num_classes=datamodule.num_classes,
        learning_rate=cfg.model.architecture.learning_rate,
        weight_decay=cfg.model.architecture.weight_decay,
        warmup_steps=cfg.model.architecture.warmup_steps,
        max_steps=max_steps,
        freeze_backbone=cfg.model.architecture.freeze_backbone,
        dropout=cfg.model.architecture.dropout,
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
    if cfg.train.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val/acc",
            patience=cfg.train.early_stopping_patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(
            f"Added EarlyStopping callback (patience={cfg.train.early_stopping_patience})")

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
    if cfg.logging.log_gradients:
        grad_callback = GradientLoggingCallback(log_every_n_steps=50)
        callbacks.append(grad_callback)
        logger.info("Added GradientLoggingCallback")

    # Prediction logging (W&B only)
    if cfg.logging.log_predictions and cfg.logging.use_wandb:
        pred_callback = LogPredictionsCallback(
            num_samples=8, log_every_n_epochs=1)
        callbacks.append(pred_callback)
        logger.info("Added LogPredictionsCallback")
    elif cfg.logging.log_predictions and not cfg.logging.use_wandb:
        logger.warning(
            "--log_predictions requires --use_wandb, skipping prediction logging")

    # ==================
    # Logger
    # ==================
    logger.info("Setting up experiment logger...")
    if cfg.logging.use_wandb:
        try:
            import wandb

            # Check if logged in
            if not wandb.api.api_key:
                logger.error(
                    "W&B API key not found. Please run 'wandb login' first.")
                logger.info("Falling back to TensorBoard")
                cfg.logging.use_wandb = False
            else:
                exp_logger = WandbLogger(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    name=cfg.experiment_name,
                    save_dir=output_dir / "logs",
                    log_model="all",  # Log model checkpoints
                )

                # Log hyperparameters
                exp_logger.experiment.config.update({
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.model.architecture.learning_rate,
                    "weight_decay": cfg.model.architecture.weight_decay,
                    "model_name": cfg.model.architecture.model_name,
                    "data_fraction": cfg.data.data_fraction,
                    "freeze_backbone": cfg.model.architecture.freeze_backbone,
                    "max_epochs": cfg.train.max_epochs,
                    "warmup_steps": cfg.model.architecture.warmup_steps,
                    "dropout": cfg.model.architecture.dropout,
                })

                logger.info(
                    f"Using W&B logger (project: {cfg.logging.wandb_project})")

        except ImportError:
            logger.error(
                "wandb not installed. Install with: pip install wandb")
            logger.info("Falling back to TensorBoard")
            cfg.logging.use_wandb = False

    if not cfg.logging.use_wandb:
        exp_logger = TensorBoardLogger(
            save_dir=output_dir / "logs",
            name=cfg.experiment_name,
        )
        logger.info("Using TensorBoard logger")

    # ==================
    # Profiler
    # ==================
    profiler = create_profiler(cfg.profiler, output_dir)
    if profiler:
        logger.info(f"Profiling enabled: {cfg.profiler}")
    else:
        logger.info("Profiling disabled")

    # ==================
    # Trainer
    # ==================
    logger.info("Creating trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        logger=exp_logger,
        profiler=profiler,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        deterministic=True,
        fast_dev_run=cfg.train.fast_dev_run,
        log_every_n_steps=10,
    )

    # ==================
    # Training Summary
    # ==================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {cfg.model.architecture.model_name}")
    logger.info(f"Dataset: Food-101 ({cfg.data.data_fraction*100:.1f}% of data)")
    logger.info(f"Num classes: {datamodule.num_classes}")
    logger.info(f"Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"Val samples: {len(datamodule.val_dataset)}")
    logger.info(f"Test samples: {len(datamodule.test_dataset)}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Gradient accumulation: {cfg.train.accumulate_grad_batches}")
    logger.info(
        f"Effective batch size: {cfg.batch_size * cfg.train.accumulate_grad_batches}")
    logger.info(f"Learning rate: {cfg.model.architecture.learning_rate}")
    logger.info(f"Weight decay: {cfg.model.architecture.weight_decay}")
    logger.info(f"Warmup steps: {cfg.model.architecture.warmup_steps}")
    logger.info(f"Max epochs: {cfg.train.max_epochs}")
    logger.info(f"Total steps: {max_steps}")
    logger.info(f"Freeze backbone: {cfg.model.architecture.freeze_backbone}")
    logger.info(f"Logger: {'W&B' if cfg.logging.use_wandb else 'TensorBoard'}")
    logger.info(f"Profiler: {cfg.profiler if cfg.profiler else 'None'}")
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
    if cfg.run_test:
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

    if cfg.logging.use_wandb:
        logger.info(f"View results at: {exp_logger.experiment.url}")

    logger.info("=" * 80 + "\n")

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
