from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import build_manifests, print_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/test manifests for RGB crop images.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/dataset.yaml.")
    parser.add_argument("--splits", type=Path, default=None, help="Path to configs/splits.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_manifests(args.config, args.splits, write_outputs=True)
    print_summary(result.stats)
    print("Wrote metadata/manifests/train.csv")
    print("Wrote metadata/manifests/test.csv")
    print("Wrote metadata/class_to_idx.json")
    print("Wrote metadata/dataset_stats.json")


if __name__ == "__main__":
    main()
