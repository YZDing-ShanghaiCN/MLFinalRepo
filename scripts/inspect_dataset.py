from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import compute_stats, print_summary, scan_dataset, write_json, assert_no_leakage  # noqa: E402
from src.utils.config import load_dataset_config, load_splits_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect configured dataset and refresh dataset_stats.json.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/dataset.yaml.")
    parser.add_argument("--splits", type=Path, default=None, help="Path to configs/splits.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    splits = load_splits_config(args.splits)
    rows_by_split, class_to_idx, report = scan_dataset(config, splits)
    stats = compute_stats(rows_by_split, class_to_idx, report, config=config)
    assert_no_leakage(report)
    write_json(PROJECT_ROOT / "metadata" / "dataset_stats.json", stats)
    print_summary(stats)
    print("Wrote metadata/dataset_stats.json")


if __name__ == "__main__":
    main()
