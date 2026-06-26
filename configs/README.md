# Configuration Guide

本目录保存项目所有 YAML 配置。除非要改变实验本身，否则建议优先通过脚本参数或新增配置文件扩展实验，不直接覆盖已有配置。

## Core Configs

| File | Purpose |
|---|---|
| `dataset.yaml` | 数据根目录、输入尺寸、图像过滤、确定性预处理和 DataLoader 默认参数 |
| `splits.yaml` | 原始 train/test 实例划分 |
| `validation_split.yaml` | 从训练实例中固定划分验证实例 |
| `model.yaml` | 默认 CNN-GAP 分类模型结构 |
| `train.yaml` | 默认训练入口 `scripts/train.py` 使用的训练参数 |

## Ablation Configs

目录：`configs/ablations/`

| File | Experiment | Output |
|---|---|---|
| `baseline_gap_dropout.yaml` | GAP + Dropout(0.3) | `../outputs/ablation/baseline_gap_dropout` |
| `flatten_dropout.yaml` | Flatten + Dropout(0.3) | `../outputs/ablation/flatten_dropout` |
| `gap_no_dropout.yaml` | GAP + Identity dropout | `../outputs/ablation/gap_no_dropout` |

三组消融实验共享：

```yaml
learning_rate: 0.004
batch_size: 32
epochs: 30
seed: 42
```

消融训练只使用：

```text
metadata/manifests/train_sub.csv
metadata/manifests/val.csv
```

不读取 `test.csv`。

## Test Configs

目录：`configs/test/`

| File | Purpose | Output |
|---|---|---|
| `baseline_test.yaml` | 对 `baseline_gap_dropout/best_model.pt` 做独立测试集评估 | `../results/test/baseline_gap_dropout` |

测试配置只允许使用：

```text
metadata/manifests/test.csv
```

不要用 `train_sub.csv` 或 `val.csv` 代替测试集。

## Path Conventions

脚本会把相对路径按 `ml_repo` 项目根目录解析。因此：

- `metadata/manifests/test.csv` 指向 `ml_repo/metadata/manifests/test.csv`
- `../outputs/ablation/...` 指向 `mlfinal/outputs/ablation/...`
- `../results/test/...` 指向 `mlfinal/results/test/...`

这样可以从父目录运行命令：

```powershell
cd C:\Users\dyz18\Desktop\code\mlfinal
python .\ml_repo\scripts\run_ablations.py
```

也可以从 `ml_repo` 内部运行多数脚本。

## Safety Rules

- 不把模型权重、日志、CSV、PNG 写入 `ml_repo` 内部，除非是轻量级元数据或配置。
- 不把测试集指标用于训练、调参或 checkpoint 选择。
- 不在消融训练中读取测试集。
- `--force` 只覆盖对应实验输出目录，不应删除其他结果目录。

