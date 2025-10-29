# Getting Started

Welcome to nanochat — a minimal, full‑stack, end‑to‑end ChatGPT‑style stack you can train, evaluate, and serve yourself. This guide walks you from a clean machine to training and serving your own model.

If you just want the shortest path: see Quick Start. If you’re on Windows, see Windows Notes.

## Prerequisites

- OS: Linux is recommended for training. macOS (MPS) and CPU are supported for experimentation. Windows users: prefer WSL2 or a Linux cloud VM.
- Python: 3.10+
- Package/venv: uv
- Rust toolchain: rustup (for building `rustbpe` via maturin)
- GPU (optional but recommended): NVIDIA with CUDA 12.8 drivers for best results; multi‑GPU recommended for training runs.
- Shell tools: curl, unzip (for dataset/eval bundle downloads); screen (optional, for long runs)
- Disk: ~30–50 GB free for artifacts, datasets, and checkpoints (varies by run)

Tip: For cloud, an 8xH100 node is ideal for the full speedrun. For local tinkering, CPU/MPS works but will be slow.

## Install uv

uv manages the virtual environment and resolves Python dependencies (including PyTorch wheels from the right index).

- Linux/macOS:
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows PowerShell:
  - `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`

Verify with `uv --version`.

## Clone and create a virtual environment

From the project root:

1) Create the venv
- `uv venv`

2) Activate the venv
- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

## Install dependencies

The repo declares CPU/GPU “extras” so uv installs the correct PyTorch wheels via custom indexes defined in `pyproject.toml:1`.

- GPU build (CUDA 12.8):
  - `uv sync --extra gpu`
- CPU‑only build:
  - `uv sync --extra cpu`

Alternatively, editable installs:
- GPU: `uv pip install -e .[gpu]`
- CPU: `uv pip install -e .[cpu]`

Note: On first install, maturin will build the `rustbpe` extension; you need a Rust toolchain.

## Install Rust toolchain and build rustbpe

- Linux/macOS: `curl https://sh.rustup.rs -sSf | sh -s -- -y && source "$HOME/.cargo/env"`
- Windows: Install rustup via the official installer or `winget install Rustlang.Rustup` and restart your shell.

Manually build the extension if needed:
- `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`

## Quick Start (full pipeline)

The fastest way to train, evaluate, and serve a ~$100 tier model is the speedrun script:

- Linux on a multi‑GPU host:
  - `bash speedrun.sh` (`speedrun.sh:1`)

Recommended for long runs: use screen and log output:
- `screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh`

What the script does (at a glance):
- Sets up uv and venv, installs deps (GPU extra)
- Builds the Rust tokenizer (`rustbpe`) and trains a BPE vocab
- Downloads datasets and an eval bundle
- Trains the base model (pretraining), then mid‑trains and SFTs
- Optionally runs RL (commented by default)
- Generates a consolidated `report.md` with metrics and samples

Outputs and state live under the base directory (default):
- `~/.cache/nanochat` — can be changed with `NANOCHAT_BASE_DIR` env (`speedrun.sh:1`)

Once complete, launch the Chat UI:
- `python -m scripts.chat_web` (`scripts/chat_web.py:1`)
- Visit the printed URL (e.g., `http://<host>:8000/`). On cloud, ensure the port is open and use the public IP.

## Inference and serving

Run the web server from any machine with the checkpoints available (CPU/MPS works too, just slower):

- Single‑GPU or CPU/MPS:
  - `python -m scripts.chat_web`
- Multi‑GPU (CUDA only):
  - `python -m scripts.chat_web --num-gpus 4`

Useful options (`scripts/chat_web.py:1`):
- `-i, --source sft|mid|rl` — which checkpoint set to load (default: sft)
- `-g, --model-tag d20|d26|d32|...` — which depth tag to load (auto‑guessed to largest if omitted)
- `-s, --step <int>` — which training step to load (defaults to last saved)
- `--device-type cuda|cpu|mps` — override device autodetection
- `-p, --port 8000` and `--host 0.0.0.0` — network bindings

Checkpoint layout and selection are handled by the loader (`nanochat/checkpoint_manager.py:1`). By default, the loader picks the largest available model tag and the latest step when `--model-tag`/`--step` are omitted.

## CLI chat

For a quick text‑only chat in terminal you can use the CLI script (after training):
- Example one‑shot prompt: `python -m scripts.chat_cli -p "Why is the sky blue?"` (`speedrun.sh:1`)

## Running parts of the pipeline manually

Below is a representative flow if you prefer to drive stages yourself.

1) Tokenizer
- Download shards: `python -m nanochat.dataset -n 8` (foreground) and optionally `python -m nanochat.dataset -n 240 &` (background)
- Train BPE: `python -m scripts.tok_train --max_chars=2000000000`
- Evaluate compression: `python -m scripts.tok_eval`

Alternative datasets (HF)
- You can pre-populate the base data directory from Hugging Face datasets and keep the rest of the pipeline unchanged. Two presets are included:
  - K‑12 Corpus: `python -m nanochat.dataset hf --preset k12 --rows-per-shard 500000`
  - OPC Annealing Synthetic: `python -m nanochat.dataset hf --preset opc --rows-per-shard 500000`
- Custom HF dataset:
  - `python -m nanochat.dataset hf --dataset CrowdMind/K-12Corpus --split train --text-field text --rows-per-shard 500000`
- After shards exist in `~/.cache/nanochat/base_data`, `speedrun.sh` will detect them and skip FineWeb downloading automatically.

2) Base model training and eval
- Ensure eval bundle exists (the speedrun downloads it automatically)
- Train: `torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=<tag>`
- Loss/eval and samples: `torchrun --standalone --nproc_per_node=8 -m scripts.base_loss`
- CORE eval: `torchrun --standalone --nproc_per_node=8 -m scripts.base_eval`

3) Midtraining + Chat SFT
- Midtrain: `torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=<tag>`
- Chat eval (mid): `torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid`
- SFT: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=<tag>`
- Chat eval (sft): `torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft`

4) (Optional) Reinforcement Learning
- Train: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=<tag>`
- Eval RL (GSM8K only): `torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K`

5) Report
- `python -m nanochat.report generate` — produces `report.md` and copies a convenience copy to the project root (`speedrun.sh:1`)

## Windows notes

- Best experience: Use WSL2 with an Ubuntu distro for training, then follow Linux commands verbatim. GPU passthrough requires compatible drivers and WSL CUDA support.
- Native Windows:
  - You can install uv and Rust and run CPU/MPS inference and many scripts.
  - `speedrun.sh` is a Bash script; run it inside WSL, or replicate steps manually using the commands above.
  - PowerShell activation: `.\.venv\Scripts\Activate.ps1`

## Environment and paths

- Base dir: Set `NANOCHAT_BASE_DIR` to change where datasets, checkpoints, and reports live. Defaults to `~/.cache/nanochat` (`speedrun.sh:1`).
- Checkpoints: Organized under base dir in subfolders (`base_checkpoints`, `mid_checkpoints`, `chatsft_checkpoints`, `chatrl_checkpoints`) per `nanochat/checkpoint_manager.py:1`.

## Testing

- Run tests: `uv run pytest -q` (`pyproject.toml:1`)
- Slow tests are marked with `@pytest.mark.slow` and can be excluded with `-m "not slow"`.

## Troubleshooting

- PyTorch wheel / CUDA mismatch:
  - Ensure you used the correct extra: `--extra gpu` installs CUDA 12.8 wheels via the configured index. For CPU, use `--extra cpu`.
- Rust/maturin build errors:
  - Make sure rustup is installed and up to date, then re‑run `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`.
- Long run stability:
  - Use `screen` and log to a file as shown above. Ensure sufficient disk space in the base dir.
- Web UI not reachable:
  - Confirm the server binds on `0.0.0.0` and the port is open: `python -m scripts.chat_web --host 0.0.0.0 --port 8000`.
- Loading checkpoints:
  - If multiple model depths exist, the loader auto‑selects the largest unless you pass `--model-tag`.

## Docker

You can build and run nanochat with Docker. The image includes uv, Rust/maturin (for `rustbpe`), and installs the GPU extra of PyTorch (CUDA 12.8 wheels). GPU is optional but recommended for training.

Files:
- `docker/Dockerfile` — Base image (Ubuntu 22.04 + CUDA runtime), uv, Rust, deps, builds `rustbpe`.
- `docker/docker-compose.web.yml` — Runs the web chat UI (port 8000), optional GPU.
- `docker/docker-compose.train.yml` — Runs the full `speedrun.sh` training pipeline, GPU enabled by default.

Build once (from repo root):
- Web: `docker compose -f docker/docker-compose.web.yml build`
- Train: `docker compose -f docker/docker-compose.train.yml build`

Persistent data
- Both compose files mount a named volume `nanochat_cache` at `/root/.cache/nanochat` to persist datasets, checkpoints, and reports across container restarts. To use a host bind mount instead, replace the volume with `./.cache/nanochat:/root/.cache/nanochat`.

Run the web UI
- Start: `docker compose -f docker/docker-compose.web.yml up -d`
- Logs: `docker compose -f docker/docker-compose.web.yml logs -f`
- Browse: `http://localhost:8000/`
- Options via env:
  - `MODEL_SOURCE=sft|mid|rl` (default sft)
  - `MODEL_TAG=d20|d26|d32|...` (defaults to largest available)
  - `MODEL_STEP=<int>` (defaults to latest)

Run training (speedrun)
- Start: `docker compose -f docker/docker-compose.train.yml up`
- Optional env: set `WANDB_RUN=<name>` in the compose file to enable Weights & Biases logging; otherwise it defaults to a local “dummy” run.

Use alternative HF datasets with Docker
- Prepare shards inside the trainer image (one-off):
  - `docker compose -f docker/docker-compose.train.yml run --rm trainer uv run python -m nanochat.dataset hf --preset k12 --rows-per-shard 500000`
- Then start training normally; the script will detect existing shards and skip FineWeb downloading.

GPU notes
- The compose files request all available NVIDIA GPUs via `deploy.resources.reservations.devices` (Compose V2 + NVIDIA Container Toolkit). Ensure Docker Desktop/Engine has GPU support and host drivers installed. For CPU‑only runs, comment out the `deploy.resources` block.

Shared memory
- Both compose files set `shm_size: 1g` to improve PyTorch and dataloader stability. Increase if you hit shared memory issues.

After training completes, you can start the web UI compose (above) to serve the freshly produced checkpoints.

## Next steps

- Train bigger models and variants (see `run1000.sh:1` for a larger d32 run).
- Customize midtraining and SFT data to shape model behavior (see `dev/` references in `README.md:1`).
- Extend evaluation or integrate external benchmarks.

For a high‑level overview and context, read `README.md:1`. For end‑to‑end in a single command on a beefy GPU box, use `speedrun.sh:1`.
