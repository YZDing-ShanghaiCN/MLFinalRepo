from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def natural_key(path: Path) -> list[object]:
    """Sort names like apple_2 before apple_10."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def is_rgb_image(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    if "depth" in name or "mask" in name:
        return False
    return True


def read_image_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if path.suffix.lower() == ".png" and len(data) >= 24 and data.startswith(PNG_SIGNATURE):
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return height, width

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def choose_test_num(num_folders: int) -> int:
    if num_folders <= 1:
        return 0

    min_test = math.ceil(num_folders * 2 / 3)
    max_test = math.floor(num_folders * 4 / 5)
    max_test = min(max_test, num_folders - 1)

    if min_test <= max_test:
        target = num_folders * 0.75
        return min(
            range(min_test, max_test + 1),
            key=lambda value: (abs(value - target), -value),
        )

    return min(max(1, math.ceil(num_folders * 0.75)), num_folders - 1)


def collect_split_stats(folders: list[Path]) -> dict[str, object]:
    rgb_files: list[Path] = []
    for folder in folders:
        rgb_files.extend(file for file in folder.iterdir() if file.is_file() and is_rgb_image(file))

    max_height: int | None = None
    max_width: int | None = None
    for image_path in rgb_files:
        size = read_image_size(image_path)
        if size is None:
            continue
        height, width = size
        max_height = height if max_height is None else max(max_height, height)
        max_width = width if max_width is None else max(max_width, width)

    return {
        "folders_num": len(folders),
        "folders_list": [folder.name for folder in folders],
        "data_num": len(rgb_files),
        "max_height": max_height,
        "max_width": max_width,
    }


def yaml_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, list):
        return "[" + ", ".join(yaml_value(item) for item in value) + "]"
    return str(value)


def render_yaml(config: dict[str, dict[str, dict[str, object]]]) -> str:
    lines: list[str] = []
    for split_name in ("train", "test"):
        lines.append(f"{split_name}:")
        for class_name, stats in config[split_name].items():
            lines.append(f"    {class_name}:")
            for key in ("folders_num", "folders_list", "data_num", "max_height", "max_width"):
                lines.append(f"        {key}: {yaml_value(stats[key])}")
    return "\n".join(lines) + "\n"


def build_config(dataset_root: Path) -> dict[str, dict[str, dict[str, object]]]:
    config: dict[str, dict[str, dict[str, object]]] = {"train": {}, "test": {}}
    class_dirs = sorted(
        (path for path in dataset_root.iterdir() if path.is_dir()),
        key=natural_key,
    )

    for class_dir in class_dirs:
        instance_folders = sorted(
            (path for path in class_dir.iterdir() if path.is_dir()),
            key=natural_key,
        )
        test_num = choose_test_num(len(instance_folders))
        train_num = len(instance_folders) - test_num
        train_folders = instance_folders[:train_num]
        test_folders = instance_folders[train_num:]

        config["train"][class_dir.name] = collect_split_stats(train_folders)
        config["test"][class_dir.name] = collect_split_stats(test_folders)

    return config


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser(description="Build Washington RGB-D train/test YAML config.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=workspace_root / "rgbd-dataset_eval",
        help="Path to rgbd-dataset_eval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "config" / "dataset_config.yaml",
        help="Output YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config = build_config(dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_yaml(config), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
