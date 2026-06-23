from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rgb_classifier import build_model_from_config  # noqa: E402
from src.training.trainer import create_manifest_loader, select_device, set_seed  # noqa: E402
from src.utils.config import load_dataset_config, load_model_config, load_train_config  # noqa: E402
from src.utils.paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit one fixed batch to verify gradients and labels.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--success-accuracy", type=float, default=0.95)
    return parser.parse_args()


def _save_curve(path: Path, history: list[dict[str, float]]) -> None:
    steps = [row["step"] for row in history]
    losses = [row["loss"] for row in history]
    accuracies = [row["accuracy"] for row in history]
    fig, axis_loss = plt.subplots(figsize=(6, 4))
    axis_loss.plot(steps, losses, label="loss", color="tab:blue")
    axis_loss.set_xlabel("Step")
    axis_loss.set_ylabel("Loss")
    axis_acc = axis_loss.twinx()
    axis_acc.plot(steps, accuracies, label="accuracy", color="tab:orange")
    axis_acc.set_ylabel("Accuracy")
    axis_loss.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(42)
    train_config = load_train_config()
    model_config = load_model_config()
    dataset_config = load_dataset_config()
    train_manifest = resolve_project_path(train_config["data"]["train_manifest"])
    if not train_manifest.is_file():
        raise FileNotFoundError(f"train_sub manifest not found: {train_manifest}. Run python scripts/build_validation_split.py first.")

    device = select_device(str(train_config.get("training", {}).get("device", "auto")))
    loader = create_manifest_loader(
        train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        return_metadata=False,
        dataset_config=dataset_config,
        num_workers=0,
        pin_memory=False,
        seed=42,
    )
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    model = build_model_from_config(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    history: list[dict[str, float]] = []
    success = False

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        accuracy = float((preds == labels).float().mean().item())
        history.append({"step": float(step), "loss": float(loss.item()), "accuracy": accuracy})
        if step == 1 or step % 25 == 0:
            print(f"Step {step:03d}: loss={loss.item():.6f}, accuracy={accuracy:.4f}")
        if accuracy >= args.success_accuracy:
            success = True
            print(f"Success at step {step}: loss={loss.item():.6f}, accuracy={accuracy:.4f}")
            break

    output_dir = resolve_project_path(train_config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final = history[-1]
    result = {
        "success": success,
        "success_threshold": args.success_accuracy,
        "steps_run": int(final["step"]),
        "final_loss": final["loss"],
        "final_accuracy": final["accuracy"],
        "history": history,
        "device": str(device),
    }
    with (output_dir / "overfit_one_batch.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    _save_curve(output_dir / "overfit_one_batch_curve.png", history)
    print(f"Final loss: {final['loss']:.6f}")
    print(f"Final accuracy: {final['accuracy']:.4f}")
    print(f"Reached >= {args.success_accuracy:.2%}: {success}")
    print("Wrote outputs/experiments/baseline_cnn_gap/overfit_one_batch.json")
    print("Wrote outputs/experiments/baseline_cnn_gap/overfit_one_batch_curve.png")


if __name__ == "__main__":
    main()
