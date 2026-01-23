# log_to_wandb.py
"""
Script to retroactively log a completed training run to Weights & Biases.

This script logs the metrics and hyperparameters from a training run that
was completed without wandb logging enabled.

Usage:
    wandb login  # Make sure you're logged in first
    python log_to_wandb.py
"""

from datetime import datetime
from pathlib import Path

import wandb


def main():
    # Initialize wandb with the correct project and entity
    run = wandb.init(
        project="MLOPS",
        entity="nicolai-ege-leivestad-danmarks-tekniske-universitet-dtu",
        name="test-run-retroactive",
        job_type="retroactive_log",
        tags=["retroactive", "food101", "vit"],
        notes="Retroactively logged from completed training run on 2026-01-23",
        config={
            # Data config
            "data_dir": "data",
            "val_fraction": 0.1,
            "data_fraction": 1.0,
            "data_augmentation": False,

            # Model config
            "model_name": "google/vit-base-patch16-224-in21k",
            "freeze_backbone": True,
            "dropout": 0.1,
            "learning_rate": 0.003,
            "weight_decay": 0.001,
            "warmup_steps": 500,

            # Training config
            "batch_size": 4,
            "max_epochs": 1,
            "accumulate_grad_batches": True,
            "early_stopping_patience": 3,
            "num_workers": 0,

            # Other config
            "seed": 42,
            "profiler": "simple",
            "run_test": True,
        }
    )

    # Log training metrics (from epoch 0)
    # Based on your training output:
    # Epoch 0/0: train/loss_epoch: 0.461, val/loss: 0.229, val/acc: 0.9307, val/acc_top5: 1.0
    total_steps = 844  # Total steps from your training

    # Log epoch-level metrics
    run.log({
        "train/loss_epoch": 0.461,
        "val/loss": 0.229,
        "val/acc": 0.9307,
        "val/acc_top5": 1.0,
        "epoch": 0,
    }, step=total_steps)

    # Log test metrics (from test evaluation)
    # test/acc: 0.9472, test/acc_top5: 1.0, test/loss: 0.1587
    run.log({
        "test/acc": 0.9472,
        "test/acc_top5": 1.0,
        "test/loss": 0.1587,
    }, step=total_steps)

    # Log summary metrics (these appear in the summary table)
    run.summary.update({
        "best_val_acc": 0.9307,
        "best_val_loss": 0.229,
        "test_acc": 0.9472,
        "test_loss": 0.1587,
        "total_steps": total_steps,
        "total_epochs": 1,
    })

    # Log the checkpoint as an artifact
    checkpoint_path = Path("outputs/checkpoints/epoch=00-val/acc=0.9307-v1.ckpt")
    if checkpoint_path.exists():
        try:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{run.id}",
                type="model",
                description="Best model checkpoint from retroactive logging"
            )
            artifact.add_file(str(checkpoint_path))
            run.log_artifact(artifact)
            print(f"✅ Logged checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Could not log checkpoint: {e}")

    # Finish the run
    run.finish()

    print("=" * 80)
    print("✅ Metrics successfully logged to wandb!")
    print(f"📊 View your run at: {run.url}")
    print("=" * 80)

if __name__ == "__main__":
    main()
