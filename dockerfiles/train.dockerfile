FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y build-essential gcc git

WORKDIR /app

# 1. Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY configs/ ./configs/

# 2. Sync dependencies (ensure dvc[s3] is in your pyproject.toml)
RUN uv sync --frozen --no-install-project

# 3. Setup DVC (Crucial Changes Here)
# DO NOT run 'dvc init' - it overwrites your config.
# Instead, copy your existing .dvc folder structure
COPY .dvc/config .dvc/config
COPY data/*.dvc ./data/

# 4. Handle Credentials
# Since config.local is usually gitignored, we set credentials via Build Args
RUN uv run dvc remote modify origin --local access_key_id f05217b3ffc457f5a681247e72f533482e108658

RUN uv run dvc remote modify origin --local secret_access_key f05217b3ffc457f5a681247e72f533482e108658

RUN uv run dvc config core.no_scm true

# 5. Pull and Organize
RUN uv run dvc pull
RUN uv run python src/mlopsproj/organize_data_script.py

ENTRYPOINT ["uv", "run", "python", "-m", "src.mlopsproj.train"]
