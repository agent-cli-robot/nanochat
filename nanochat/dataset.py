"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool
from typing import Optional

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# Defaults for the FineWeb-Edu dataset (sharded parquet)
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


# -----------------------------------------------------------------------------
# Alternative HF datasets -> convert to local parquet shards with column 'text'

HF_PRESETS = {
    # Convenience presets; override columns with --text-field if needed
    "k12": {"name": "CrowdMind/K-12Corpus", "subset": None, "split": "train", "text_field": "text"},
    "opc": {"name": "CrowdMind/opc-annealing-corpus-synthetic", "subset": None, "split": "train", "text_field": "text"},
}


def prepare_hf_dataset(
    name: str,
    subset: Optional[str],
    split: str,
    text_field: str = "text",
    rows_per_shard: int = 500_000,
    limit_rows: Optional[int] = None,
):
    """
    Download an HF dataset and materialize it as local parquet shards (column 'text'),
    compatible with the existing pretraining data loader.
    """
    assert load_dataset is not None, "datasets library not available; install 'datasets' dependency"

    ds = load_dataset(name, subset, split=split)
    total = len(ds)
    if limit_rows is not None:
        total = min(total, int(limit_rows))

    os.makedirs(DATA_DIR, exist_ok=True)
    buf = []
    shard_idx = 0
    written = 0

    def flush():
        nonlocal buf, shard_idx, written
        if not buf:
            return
        table = pa.table({"text": pa.array(buf, type=pa.string())})
        path = os.path.join(DATA_DIR, index_to_filename(shard_idx))
        pq.write_table(table, path)
        print(f"Wrote {len(buf)} rows to {path}")
        written += len(buf)
        shard_idx += 1
        buf = []

    batch = 10_000
    for i in range(0, total, batch):
        batch_ds = ds[i : min(i + batch, total)]
        texts = batch_ds[text_field]
        for t in texts:
            if t is None:
                continue
            s = str(t)
            if not s:
                continue
            buf.append(s)
            if len(buf) >= rows_per_shard:
                flush()
    flush()
    print(f"Prepared {written} rows into {shard_idx} shards at {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare base/pretraining dataset shards")
    sub = parser.add_subparsers(dest="source", required=False)

    # fineweb (default) downloader
    p_fw = sub.add_parser("fineweb", help="Download FineWeb-Edu shards (default)")
    p_fw.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = all")
    p_fw.add_argument("-w", "--num-workers", type=int, default=4, help="Parallel download workers")

    # hf converter
    p_hf = sub.add_parser("hf", help="Convert an HF dataset to local parquet shards")
    p_hf.add_argument("--dataset", type=str, default=None, help="HF dataset name, e.g. 'CrowdMind/K-12Corpus'")
    p_hf.add_argument("--subset", type=str, default=None, help="HF dataset subset name (optional)")
    p_hf.add_argument("--split", type=str, default="train", help="HF split (default: train)")
    p_hf.add_argument("--text-field", type=str, default="text", help="Name of the text field (default: text)")
    p_hf.add_argument("--rows-per-shard", type=int, default=500_000, help="Rows per local parquet shard")
    p_hf.add_argument("--limit-rows", type=int, default=None, help="Limit number of rows (for quick tests)")
    p_hf.add_argument("--preset", type=str, choices=list(HF_PRESETS.keys()), help="Use a known preset: k12 or opc")

    args = parser.parse_args()

    # default behavior: fineweb downloader if no subcommand provided
    if args.source in [None, "fineweb"]:
        num = MAX_SHARD + 1 if getattr(args, "num_files", -1) == -1 else min(args.num_files, MAX_SHARD + 1)
        ids_to_download = list(range(num))
        if len(ids_to_download) == 0:
            print("Nothing to download (num-files=0)")
            raise SystemExit(0)
        print(f"Downloading {len(ids_to_download)} shards using {getattr(args, 'num_workers', 4)} workers...")
        print(f"Target directory: {DATA_DIR}")
        print()
        with Pool(processes=getattr(args, "num_workers", 4)) as pool:
            results = pool.map(download_single_file, ids_to_download)
        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
    elif args.source == "hf":
        if args.preset:
            preset = HF_PRESETS[args.preset]
            name = preset["name"]
            subset = preset["subset"]
            split = preset["split"]
            text_field = preset["text_field"]
        else:
            assert args.dataset, "--dataset is required unless --preset is provided"
            name = args.dataset
            subset = args.subset
            split = args.split
            text_field = args.text_field
        prepare_hf_dataset(name, subset, split, text_field, args.rows_per_shard, args.limit_rows)
