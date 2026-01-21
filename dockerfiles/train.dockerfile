FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV DVC_REMOTE=gcpstore
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY .dvc .dvc/
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen
COPY configs configs/

# Create a simple entrypoint (easier syntax)
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'echo "Container starting..."' >> /entrypoint.sh && \
    echo 'if [ "$DVC_REMOTE" = "gcpstore" ]; then' >> /entrypoint.sh && \
    echo '  echo "Using GCP remote for data"' >> /entrypoint.sh && \
    echo '  dvc pull --remote gcpstore 2>/dev/null || true' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '  echo "Using DagsHub remote for data"' >> /entrypoint.sh && \
    echo '  dvc pull --remote origin 2>/dev/null || true' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "python", "-m", "mlopsproj.train"]
