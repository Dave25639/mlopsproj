# model.py
"""
PyTorch Lightning module for fine-tuning google/vit-base-patch16-224-in21k
on Food-101 (or other image classification tasks).

This module:
  - Loads the pretrained ViT model from Hugging Face
  - Adds a classification head for num_classes
  - Implements training/validation/test steps with metrics
  - Supports configurable optimizer and learning rate scheduling
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torchmetrics import Accuracy, MetricCollection

try:
    import lightning as L
except Exception:
    import pytorch_lightning as L  # type: ignore

from transformers import ViTForImageClassification, ViTConfig


class ViTClassifier(L.LightningModule):
    """
    Vision Transformer classifier for image classification.

    Fine-tunes google/vit-base-patch16-224-in21k (or similar) for classification tasks.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 101,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: Optional[int] = None,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Linear warmup steps
            max_steps: Total training steps (for lr scheduling)
            freeze_backbone: If True, only train the classification head
            dropout: Dropout rate for classifier head
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.freeze_backbone = freeze_backbone

        # Load pretrained ViT with custom classification head
        config = ViTConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.classifier_dropout_prob = dropout

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,  # classifier head will be different size
        )

        # Optionally freeze the backbone (only train classifier head)
        if freeze_backbone:
            for param in self.model.vit.parameters():
                param.requires_grad = False

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "acc_top5": Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
        }, prefix="train/")

        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            pixel_values: [batch_size, 3, 224, 224]

        Returns:
            logits: [batch_size, num_classes]
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        metrics: MetricCollection
    ) -> Dict[str, torch.Tensor]:
        """Shared step logic for train/val/test."""
        pixel_values, labels = batch
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)

        return {"loss": loss, "logits": logits, "preds": preds, "labels": labels}

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, self.train_metrics)
        self.log("train/loss", outputs["loss"],
                 on_step=True, on_epoch=True, prog_bar=True)
        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self._shared_step(batch, self.val_metrics)
        self.log("val/loss", outputs["loss"],
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        outputs = self._shared_step(batch, self.test_metrics)
        self.log("test/loss", outputs["loss"], on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure AdamW optimizer with linear warmup + cosine decay.
        """
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if self.max_steps is None:
            # No scheduler if max_steps not provided
            return optimizer

        # Linear warmup + cosine decay
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# -----------------------
# Sanity check
# -----------------------

if __name__ == "__main__":
    # Quick test
    model = ViTClassifier(
        num_classes=101,
        learning_rate=3e-4,
        freeze_backbone=False,
    )

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 101, (batch_size,))

    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")  # Should be [4, 101]

    # Test training step
    loss = model.training_step((dummy_input, dummy_labels), 0)
    print(f"Loss: {loss.item():.4f}")

    print("\nModel initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
