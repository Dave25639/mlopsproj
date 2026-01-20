import os
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlopsproj"
PYTHON_VERSION = "3.12"


# Project commands hh
@task
def preprocess_data(ctx: Context, input_dir: str = "data/raw", output_dir: str = "data/processed") -> None:
    """
    Preprocess data.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.data {input_dir} {output_dir}",
        echo=True,
        pty=not WINDOWS
    )


@task
def train(
    ctx: Context,
    data_dir: str = "data",
    batch_size: int = 4,
    max_epochs: int = 2,
    data_fraction: float = 1,
    learning_rate: float = 3e-4,
    use_wandb: bool = False,
    experiment_name: str = "food101_vit",
) -> None:
    """
    Train model with small dataset for quick testing.

    Args:
        data_dir: Path to Food-101 dataset directory
        batch_size: Batch size for training (default: 4)
        max_epochs: Maximum number of training epochs (default: 2)
        data_fraction: Fraction of dataset to use (default: 0.001 for quick testing)
        learning_rate: Learning rate
        use_wandb: Enable Weights & Biases logging
        experiment_name: Name of the experiment
        **kwargs: Additional arguments passed to train.py
    """
    # Build base command
    cmd_parts = [
        f"uv run python -m {PROJECT_NAME}.train",
        f"--data_dir {data_dir}",
        f"--batch_size {batch_size}",
        f"--max_epochs {max_epochs}",
        f"--data_fraction {data_fraction}",
        f"--learning_rate {learning_rate}",
        f"--experiment_name {experiment_name}",
    ]

    # Add optional flags
    if use_wandb:
        cmd_parts.append("--use_wandb")

    # # Add any additional kwargs as command-line arguments
    # for key, value in kwargs.items():
    #     if value is not None:
    #         if isinstance(value, bool) and value:
    #             cmd_parts.append(f"--{key}")
    #         elif not isinstance(value, bool):
    #             cmd_parts.append(f"--{key} {value}")

    cmd = " ".join(cmd_parts)
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context, coverage: bool = True) -> None:
    """
    Run tests.

    Args:
        coverage: Whether to generate coverage report
    """
    if coverage:
        ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
        ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)
    else:
        ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """
    Build docker images.

    Args:
        progress: Docker build progress type (plain, auto, tty)
    """
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS
    )


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run(
        "uv run mkdocs serve --config-file docs/mkdocs.yaml",
        echo=True,
        pty=not WINDOWS
    )
