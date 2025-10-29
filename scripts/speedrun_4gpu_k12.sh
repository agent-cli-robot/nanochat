#!/bin/bash

# 4-GPU speedrun variant that seeds the K-12 HF dataset
# and uses 4 processes per node for torchrun.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer and dataset preparation (K-12 via HF)

# Install Rust / Cargo for rustbpe build
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# If base_data is empty, populate with K-12 HF preset shards
BASE_DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
mkdir -p "$BASE_DATA_DIR"
if [ -z "$(ls -A "$BASE_DATA_DIR" 2>/dev/null)" ]; then
    echo "Base data empty; preparing K-12 shards via HF preset..."
    # Write local parquet shards (column 'text'), default 500k rows per shard
    # You can adjust rows-per-shard if you want larger/smaller shard sizes.
    python -m nanochat.dataset hf --preset k12 --rows-per-shard 500000
else
    echo "Found existing shards in $BASE_DATA_DIR; skipping K-12 preparation."
fi

# Train tokenizer on ~2B characters of data (adjust to taste)
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle "$NANOCHAT_BASE_DIR"
fi

# Pretrain d20 with 4 GPUs
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=4 -m scripts.base_loss
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining + SFT

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=4 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report generate

echo "Done. Report at: $NANOCHAT_BASE_DIR/report/report.md (also copied to project root)."

