# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.11.14-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FLWR_HOME=/root/.flwr \
    FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1 \
    FLWR_DISABLE_UPDATE_CHECK=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --no-install-project

# Bake MNIST and Fashion-MNIST into the image so simulated clients never race to download it at runtime.
RUN python -c "from torchvision import datasets; [d('/app/data', train=t, download=True) for d in (datasets.MNIST, datasets.FashionMNIST) for t in (True, False)]"

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev

CMD ["sh", "scripts/run_simulation.sh"]
