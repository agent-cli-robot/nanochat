# syntax=docker/dockerfile:1.7

# Base with CUDA runtime for GPU (driver provided by host)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.cargo/bin:/root/.local/bin:${PATH}" \
    NANOCHAT_BASE_DIR="/root/.cache/nanochat"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    pkg-config \
    curl \
    git \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (venv + resolver) and rust (for rustbpe via maturin)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Copy project metadata first for better layer caching
COPY pyproject.toml uv.lock ./
COPY rustbpe/Cargo.toml rustbpe/Cargo.lock ./rustbpe/
# Include sources required for building the local project and rust extension
COPY README.md ./
COPY nanochat ./nanochat
COPY scripts ./scripts
COPY tasks ./tasks
COPY dev ./dev
COPY rustbpe/src ./rustbpe/src

# Create venv and install runtime deps (GPU extra pulls cu128 wheels)
RUN uv venv && uv pip install maturin==1.9.4 && \
    uv sync --extra gpu

# Copy run scripts (after deps for better caching)
COPY speedrun.sh run1000.sh ./

# Build the rustbpe extension
RUN uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

EXPOSE 8000

# Default to serving the web UI; override in compose for training
CMD ["uv", "run", "python", "-m", "scripts.chat_web", "--host", "0.0.0.0", "--port", "8000"]
