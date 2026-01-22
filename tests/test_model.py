"""
Unit tests for model construction and training.

Tests cover:
- Model initialization
- Forward pass
- Training/validation/test steps
- Optimizer configuration
- Freeze backbone functionality
"""

import pytest
import torch
from mlopsproj.model import ViTClassifier


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 5, (batch_size,))
    return images, labels


@pytest.fixture
def small_model():
    """Create a small model for testing (5 classes)."""
    return ViTClassifier(
        num_classes=5,
        learning_rate=1e-4,
        freeze_backbone=True,
    )


def test_model_initialization(small_model):
    """Test that model initializes correctly."""
    assert small_model is not None
    assert small_model.num_classes == 5
    assert small_model.learning_rate == 1e-4
    assert small_model.freeze_backbone is True


def test_model_forward_pass(small_model, dummy_batch):
    """Test forward pass returns correct shape."""
    images, _ = dummy_batch

    # Forward pass
    logits = small_model(images)

    # Check output shape: [batch_size, num_classes]
    assert logits.shape == (4, 5)
    assert logits.dtype == torch.float32


def test_model_training_step(small_model, dummy_batch):
    """Test training step returns loss."""
    images, labels = dummy_batch

    # Training step
    loss = small_model.training_step((images, labels), batch_idx=0)

    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0  # Loss should be positive


def test_model_validation_step(small_model, dummy_batch):
    """Test validation step doesn't crash."""
    images, labels = dummy_batch

    # Validation step should not return anything (or return None)
    result = small_model.validation_step((images, labels), batch_idx=0)
    assert result is None


def test_model_test_step(small_model, dummy_batch):
    """Test test step doesn't crash."""
    images, labels = dummy_batch

    # Test step should not return anything (or return None)
    result = small_model.test_step((images, labels), batch_idx=0)
    assert result is None




def test_model_freeze_backbone():
    """Test that freeze_backbone actually freezes parameters."""
    # Model with frozen backbone
    frozen_model = ViTClassifier(
        num_classes=5,
        freeze_backbone=True,
    )

    # Model with unfrozen backbone
    unfrozen_model = ViTClassifier(
        num_classes=5,
        freeze_backbone=False,
    )

    # Count trainable parameters
    frozen_trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    unfrozen_trainable = sum(p.numel() for p in unfrozen_model.parameters() if p.requires_grad)

    # Frozen model should have fewer trainable parameters
    assert frozen_trainable < unfrozen_trainable
    # Frozen model should only have classifier head trainable (very few params)
    assert frozen_trainable < 10000  # Should be just a few thousand


def test_model_different_num_classes():
    """Test model works with different numbers of classes."""
    for num_classes in [2, 5, 10, 101]:
        model = ViTClassifier(num_classes=num_classes)
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        logits = model(images)
        assert logits.shape == (batch_size, num_classes)

        # Training step
        loss = model.training_step((images, labels), batch_idx=0)
        assert loss.item() > 0


def test_model_metrics_initialization(small_model):
    """Test that metrics are initialized correctly."""
    assert hasattr(small_model, 'train_metrics')
    assert hasattr(small_model, 'val_metrics')
    assert hasattr(small_model, 'test_metrics')


def test_model_loss_computation(small_model, dummy_batch):
    """Test that loss is computed correctly."""
    images, labels = dummy_batch

    # Get logits
    logits = small_model(images)

    # Compute loss manually
    manual_loss = small_model.criterion(logits, labels)

    # Get loss from training step
    step_loss = small_model.training_step((images, labels), batch_idx=0)

    # They should be approximately equal (within numerical precision)
    assert torch.allclose(manual_loss, step_loss, atol=1e-5)


def test_model_with_scheduler():
    """Test model with learning rate scheduler."""
    model = ViTClassifier(
        num_classes=5,
        max_steps=100,
        warmup_steps=10,
    )

    optimizer_config = model.configure_optimizers()

    # Should have scheduler when max_steps is provided
    assert "lr_scheduler" in optimizer_config
    assert optimizer_config["lr_scheduler"]["interval"] == "step"


def test_model_without_scheduler():
    """Test model without learning rate scheduler."""
    model = ViTClassifier(
        num_classes=5,
        max_steps=None,  # No scheduler
    )

    optimizer_config = model.configure_optimizers()

    # Should only return optimizer when max_steps is None
    # (or dict with just optimizer)
    if isinstance(optimizer_config, dict):
        assert "optimizer" in optimizer_config
