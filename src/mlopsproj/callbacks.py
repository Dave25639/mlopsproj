# callbacks.py
"""
Custom callbacks for enhanced logging and monitoring.

Includes:
  - LogPredictionsCallback: Logs sample predictions to W&B
  - MetricsHistoryCallback: Tracks metrics history
  - GradientLoggingCallback: Logs gradient norms
"""

from __future__ import annotations

import logging
from typing import Optional, List

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class LogPredictionsCallback(Callback):
    """
    Log sample predictions to W&B at the end of each validation epoch.

    Shows actual images with predicted vs true labels.
    """

    def __init__(self, num_samples: int = 8, log_every_n_epochs: int = 1):
        """
        Args:
            num_samples: Number of samples to log
            log_every_n_epochs: Log predictions every N epochs
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        logger.info(f"LogPredictionsCallback initialized: logging {num_samples} samples "
                    f"every {log_every_n_epochs} epoch(s)")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log predictions at the end of validation epoch."""
        # Only log every N epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Check if we're using W&B
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return

        # Import wandb only if needed
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            if not isinstance(trainer.logger, WandbLogger):
                return
        except ImportError:
            logger.warning("wandb not installed, skipping prediction logging")
            return

        try:
            # Get a batch from validation dataloader
            val_batch = next(iter(trainer.val_dataloaders))
            imgs, labels = val_batch

            # Limit to num_samples
            imgs = imgs[:self.num_samples].to(pl_module.device)
            labels = labels[:self.num_samples]

            # Get predictions
            pl_module.eval()
            with torch.no_grad():
                logits = pl_module(imgs)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)

            # Get class names if available
            class_names = None
            if hasattr(trainer.datamodule, 'train_ds') and hasattr(trainer.datamodule.train_ds, 'classes'):
                class_names = trainer.datamodule.train_ds.classes

            # Denormalize images for visualization
            imgs_vis = self._denormalize(imgs)

            # Create W&B images with predictions
            wandb_images = []
            for i, (img, pred, label, prob) in enumerate(zip(imgs_vis, preds, labels, probs)):
                # Get class names
                pred_class = class_names[pred] if class_names else str(
                    pred.item())
                true_class = class_names[label] if class_names else str(
                    label.item())
                confidence = prob[pred].item()

                # Create caption
                caption = f"Pred: {pred_class} ({confidence:.2%})\nTrue: {true_class}"

                # Convert tensor to numpy and transpose for W&B (C, H, W) -> (H, W, C)
                img_np = img.cpu().numpy().transpose(1, 2, 0)

                wandb_images.append(wandb.Image(img_np, caption=caption))

            # Log to W&B
            trainer.logger.experiment.log({
                "val/predictions": wandb_images,
                "epoch": trainer.current_epoch
            })

            logger.info(
                f"Logged {len(wandb_images)} prediction samples to W&B")

        except Exception as e:
            logger.warning(f"Failed to log predictions: {e}")

    @staticmethod
    def _denormalize(imgs: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
        """Denormalize images for visualization."""
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(imgs.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(imgs.device)
        imgs = imgs * std + mean
        return torch.clamp(imgs, 0, 1)


class MetricsHistoryCallback(Callback):
    """
    Track and log metrics history throughout training.

    Useful for analyzing training dynamics and debugging.
    """

    def __init__(self):
        super().__init__()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []
        logger.info("MetricsHistoryCallback initialized")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Track training loss."""
        if 'train/loss_epoch' in trainer.callback_metrics:
            loss = trainer.callback_metrics['train/loss_epoch'].item()
            self.train_losses.append(loss)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Track validation metrics."""
        if 'val/loss' in trainer.callback_metrics:
            loss = trainer.callback_metrics['val/loss'].item()
            self.val_losses.append(loss)

        if 'val/acc' in trainer.callback_metrics:
            acc = trainer.callback_metrics['val/acc'].item()
            self.val_accs.append(acc)

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log final metrics summary."""
        logger.info("=" * 80)
        logger.info("TRAINING METRICS SUMMARY")
        logger.info("=" * 80)

        if self.train_losses:
            logger.info(f"Train Loss - Start: {self.train_losses[0]:.4f}, "
                        f"End: {self.train_losses[-1]:.4f}, "
                        f"Best: {min(self.train_losses):.4f}")

        if self.val_losses:
            logger.info(f"Val Loss - Start: {self.val_losses[0]:.4f}, "
                        f"End: {self.val_losses[-1]:.4f}, "
                        f"Best: {min(self.val_losses):.4f}")

        if self.val_accs:
            logger.info(f"Val Accuracy - Start: {self.val_accs[0]:.4f}, "
                        f"End: {self.val_accs[-1]:.4f}, "
                        f"Best: {max(self.val_accs):.4f}")

        logger.info("=" * 80)


class GradientLoggingCallback(Callback):
    """
    Log gradient norms to monitor training stability.

    Helps identify gradient explosion/vanishing issues.
    """

    def __init__(self, log_every_n_steps: int = 50):
        """
        Args:
            log_every_n_steps: Log gradients every N steps
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        logger.info(
            f"GradientLoggingCallback initialized: logging every {log_every_n_steps} steps")

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer
    ) -> None:
        """Log gradient norms before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Calculate gradient norms
        total_norm = 0.0
        num_params = 0

        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1

        total_norm = total_norm ** 0.5

        # Log to logger
        pl_module.log("train/grad_norm", total_norm,
                      on_step=True, on_epoch=False, logger=True)

        # Log warning if gradient is too large
        if total_norm > 10.0:
            logger.warning(
                f"Large gradient norm detected: {total_norm:.4f} at step {trainer.global_step}")


class DetailedProgressCallback(Callback):
    """
    Log detailed progress information at key training milestones.
    """

    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log training start info."""
        logger.info("=" * 80)
        logger.info("TRAINING STARTED")
        logger.info("=" * 80)
        logger.info(f"Max epochs: {trainer.max_epochs}")
        logger.info(f"Total training batches: {trainer.num_training_batches}")
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'batch_size'):
            logger.info(f"Batch size: {trainer.datamodule.batch_size}")
        logger.info("=" * 80)

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log epoch start."""
        logger.info(f"\n{'='*80}")
        logger.info(
            f"Starting Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        logger.info(f"{'='*80}")

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log training end info."""
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {trainer.current_epoch + 1}")
        logger.info(f"Total steps: {trainer.global_step}")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test callback initialization
    pred_cb = LogPredictionsCallback(num_samples=8)
    metrics_cb = MetricsHistoryCallback()
    grad_cb = GradientLoggingCallback(log_every_n_steps=50)
    progress_cb = DetailedProgressCallback()

    print("All callbacks initialized successfully!")
