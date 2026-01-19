# sweep.py
"""
Run W&B hyperparameter sweeps for Food-101 ViT training.

This script wraps train.py to work with W&B sweeps, allowing you to run
hyperparameter optimization experiments.

Usage:
    # Initialize a sweep
    python sweep.py --data_dir /path/to/food-101 --init_sweep

    # Run a sweep agent (after initializing)
    python sweep.py --data_dir /path/to/food-101 --sweep_id <sweep_id>

    # Or use the W&B CLI directly:
    wandb sweep sweep_config.yaml  # Returns sweep_id
    wandb agent <sweep_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run W&B hyperparameter sweeps for Food-101 ViT training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sweep control
    parser.add_argument(
        "--init_sweep",
        action="store_true",
        help="Initialize a new sweep and print sweep ID",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Sweep ID to run (from wandb sweep or --init_sweep)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (default: unlimited)",
    )

    # Data arguments (required for sweep init)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to Food-101 dataset directory",
    )

    # Sweep configuration
    parser.add_argument(
        "--sweep_config",
        type=str,
        default="sweep_config.yaml",
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="food101-vit-sweep",
        help="W&B project name for sweep",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (team) name",
    )

    return parser.parse_args()


def load_sweep_config(config_path: str, data_dir: str) -> dict:
    """
    Load and update sweep configuration.

    Args:
        config_path: Path to sweep config YAML
        data_dir: Data directory to inject into config

    Returns:
        Sweep configuration dictionary
    """
    import yaml

    logger.info(f"Loading sweep config from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update data_dir in config
    if 'parameters' not in config:
        config['parameters'] = {}

    config['parameters']['data_dir'] = {'value': data_dir}

    logger.info(f"Sweep method: {config.get('method', 'unknown')}")
    logger.info(f"Metric: {config.get('metric', {}).get('name', 'unknown')} "
                f"({config.get('metric', {}).get('goal', 'unknown')})")

    return config


def init_sweep(args: argparse.Namespace) -> str:
    """
    Initialize a W&B sweep.

    Args:
        args: Parsed arguments

    Returns:
        Sweep ID
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    # Check if logged in
    if not wandb.api.api_key:
        logger.error("W&B API key not found. Please run 'wandb login' first.")
        sys.exit(1)

    # Load sweep configuration
    config = load_sweep_config(args.sweep_config, args.data_dir)

    # Initialize sweep
    logger.info(f"Initializing sweep in project: {args.project}")
    sweep_id = wandb.sweep(
        config,
        project=args.project,
        entity=args.entity,
    )

    logger.info("=" * 80)
    logger.info("SWEEP INITIALIZED")
    logger.info("=" * 80)
    logger.info(f"Sweep ID: {sweep_id}")
    logger.info(f"Project: {args.project}")
    logger.info(f"\nTo run this sweep, use:")
    logger.info(
        f"  python sweep.py --data_dir {args.data_dir} --sweep_id {sweep_id}")
    logger.info(f"\nOr use the W&B CLI:")
    logger.info(
        f"  wandb agent {args.entity + '/' if args.entity else ''}{args.project}/{sweep_id}")
    logger.info("=" * 80)

    return sweep_id


def run_sweep(args: argparse.Namespace):
    """
    Run a W&B sweep agent.

    Args:
        args: Parsed arguments
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    # Check if logged in
    if not wandb.api.api_key:
        logger.error("W&B API key not found. Please run 'wandb login' first.")
        sys.exit(1)

    if not args.sweep_id:
        logger.error("--sweep_id is required to run a sweep")
        logger.info("Initialize a sweep first with --init_sweep")
        sys.exit(1)

    # Import train function
    try:
        from src.mlopsproj.train import main as train_main
    except ImportError:
        logger.error(
            "Could not import train.py. Make sure it's in the same directory.")
        sys.exit(1)

    def train_fn():
        """Wrapper function for W&B sweep."""
        # W&B will inject hyperparameters via wandb.config
        # train.py will read them from command-line args, which we need to handle

        # For now, just call the main training function
        # The train.py script will read wandb.config when using --use_wandb
        train_main()

    logger.info("=" * 80)
    logger.info("STARTING SWEEP AGENT")
    logger.info("=" * 80)
    logger.info(f"Sweep ID: {args.sweep_id}")
    logger.info(f"Count: {args.count if args.count else 'unlimited'}")
    logger.info("=" * 80)

    # Run the sweep agent
    wandb.agent(
        args.sweep_id,
        function=train_fn,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )

    logger.info("Sweep agent completed")


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("W&B Sweep Runner")
    logger.info("=" * 80)

    if args.init_sweep:
        # Initialize a new sweep
        sweep_id = init_sweep(args)

        # Optionally run the sweep immediately
        if args.sweep_id is None:
            logger.info("\nSweep initialized but not started.")
            logger.info("Run the sweep with:")
            logger.info(
                f"  python sweep.py --data_dir {args.data_dir} --sweep_id {sweep_id}")

    elif args.sweep_id:
        # Run an existing sweep
        run_sweep(args)

    else:
        logger.error("Please specify either --init_sweep or --sweep_id")
        logger.info("\nExamples:")
        logger.info("  # Initialize a sweep:")
        logger.info(
            "  python sweep.py --data_dir /path/to/food-101 --init_sweep")
        logger.info("\n  # Run a sweep:")
        logger.info(
            "  python sweep.py --data_dir /path/to/food-101 --sweep_id <sweep_id>")
        sys.exit(1)


if __name__ == "__main__":
    main()
