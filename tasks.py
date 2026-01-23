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
def train(ctx: Context) -> None:
    """
    Train model with small dataset for quick testing.
    """
    ctx.run(f"uv run python -m {PROJECT_NAME}.train", echo=True, pty=not WINDOWS)


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
