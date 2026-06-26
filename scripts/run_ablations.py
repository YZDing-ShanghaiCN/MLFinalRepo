from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml  # noqa: E402
from src.utils.paths import project_root, resolve_project_path  # noqa: E402


EXPERIMENT_CONFIGS = [
    Path("configs/ablations/baseline_gap_dropout.yaml"),
    Path("configs/ablations/flatten_dropout.yaml"),
    Path("configs/ablations/gap_no_dropout.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all ablation experiments in a fixed order.")
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments inside ../outputs/ablation only.")
    return parser.parse_args()


def ablation_root() -> Path:
    return (project_root().parent / "outputs" / "ablation").resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def output_dir_from_config(config_path: Path) -> Path:
    config = load_yaml(config_path)
    output_dir = resolve_project_path(config["experiment"]["output_dir"])
    if not is_relative_to(output_dir, ablation_root()):
        raise ValueError(f"Refusing to use output directory outside {ablation_root()}: {output_dir}")
    return output_dir


def is_completed(output_dir: Path) -> bool:
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.csv"
    if not metrics_path.is_file() or not history_path.is_file():
        return False
    metrics = load_yaml(metrics_path) if metrics_path.suffix in {".yaml", ".yml"} else None
    if metrics is None:
        import json

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    if metrics.get("status") != "completed":
        return False
    with history_path.open("r", encoding="utf-8") as file:
        return max(0, sum(1 for _ in file) - 1) == 30


def safe_remove_output_dir(output_dir: Path) -> None:
    if not is_relative_to(output_dir, ablation_root()):
        raise ValueError(f"Refusing to remove directory outside {ablation_root()}: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def run_one(config_path: Path, *, force: bool) -> bool:
    output_dir = output_dir_from_config(config_path)
    if force:
        safe_remove_output_dir(output_dir)
    elif is_completed(output_dir):
        print(f"Skipping completed ablation: {config_path}")
        return True

    command = [sys.executable, str(project_root() / "scripts" / "train_ablation.py"), "--config", str(project_root() / config_path)]
    print(f"Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=project_root(), check=False)
    if completed.returncode != 0:
        print(f"FAILED: {config_path} exited with code {completed.returncode}")
        return False
    return True


def main() -> None:
    args = parse_args()
    failures: list[Path] = []
    for config_path in EXPERIMENT_CONFIGS:
        try:
            if not run_one(config_path, force=args.force):
                failures.append(config_path)
        except Exception as exc:  # noqa: BLE001 - preserve completed outputs and report the failing group
            print(f"FAILED: {config_path}: {exc}")
            failures.append(config_path)

    summary_script = project_root() / "scripts" / "summarize_ablations.py"
    subprocess.run([sys.executable, str(summary_script)], cwd=project_root(), check=False)
    if failures:
        failed = ", ".join(str(path) for path in failures)
        raise SystemExit(f"Ablation run finished with failures: {failed}")
    print("All ablation experiments completed.")


if __name__ == "__main__":
    main()
