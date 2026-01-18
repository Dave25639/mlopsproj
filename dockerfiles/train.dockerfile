FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

# DVC Data Pull
RUN uv run dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN uv run dvc config core.no_scm true
RUN uv run dvc pull

ENTRYPOINT ["uv", "run", "src/mlopsproj/train.py"]
