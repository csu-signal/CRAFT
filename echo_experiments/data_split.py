"""
Train/eval structure pools. Training uses a generated pool; the 20-structure benchmark is held out.

Usage:
    python data_split.py --n 500          # writes data/train_structures.json
"""
import argparse
import json
import sys
from pathlib import Path

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from structure_generator_v2 import generate_dataset

# differs from the benchmark's generation seed (42) to avoid overlap
DEFAULT_TRAIN_SEED = 1000
DEFAULT_TRAIN_POOL_PATH = Path(__file__).resolve().parent / "data" / "train_structures.json"
BENCHMARK_PATH = _CRAFT_ROOT / "data" / "structures_dataset_20.json"


def build_training_pool(n=500, seed=DEFAULT_TRAIN_SEED, out_path=DEFAULT_TRAIN_POOL_PATH):
    structures = generate_dataset(n=n, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(structures, f)
    print(f"wrote {len(structures)} training-only structures -> {out_path}")
    return structures


def load_training_pool(path=DEFAULT_TRAIN_POOL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python data_split.py` first to generate the training pool."
        )
    with open(path) as f:
        return json.load(f)


def load_benchmark_structures(path=BENCHMARK_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--out", type=str, default=str(DEFAULT_TRAIN_POOL_PATH))
    args = parser.parse_args()
    build_training_pool(n=args.n, seed=args.seed, out_path=args.out)
