# RGB-D Object Classification

本仓库实现了一个基于 RGB crop 图像的 13 类物体分类流程，覆盖数据清单构建、确定性预处理、CNN-GAP 基线训练、消融实验、独立测试集评估、结果汇总与可视化。

项目代码位于 `ml_repo/`，原始数据默认位于同级目录 `../rgbd-dataset_eval/`。训练、消融和测试结果默认写入项目父目录下的 `outputs/` 或 `results/`，避免把模型权重、CSV、日志和图片混入代码仓库。

## Project Layout

```text
ml_repo/
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── validation_split.yaml
│   ├── ablations/
│   └── test/
├── metadata/
│   ├── class_to_idx.json
│   └── manifests/
│       ├── train.csv
│       ├── train_sub.csv
│       ├── val.csv
│       └── test.csv
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

主要外部输出目录：

```text
../outputs/
├── ablation/
├── experiments1/
└── image/

../results/
└── test/
```

## Environment

Windows PowerShell + Conda 环境示例：

```powershell
conda activate mlearn
cd C:\Users\dyz18\Desktop\code\mlfinal
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

安装依赖：

```powershell
pip install -r .\ml_repo\requirements.txt
```

依赖包括 PyTorch、Pillow、PyYAML、matplotlib、pytest、numpy 和 scikit-learn。

## Dataset

默认数据根目录在 [configs/dataset.yaml](configs/dataset.yaml) 中配置：

```yaml
dataset:
  root: ../rgbd-dataset_eval
  input_size: 128
  num_classes: 13
```

项目只读取 RGB crop 图像，过滤规则为：

- 文件扩展名：`.png`、`.jpg`、`.jpeg`、`.bmp`
- 文件名必须包含 `crop`
- 排除包含 `depth` 或 `mask` 的文件

预处理固定为：

- PIL 读取并转换为 RGB
- 长边缩放到 128
- 短边居中 padding 到 `128 x 128`
- 输出 `float32`、`C x H x W`、范围 `[0, 1]`
- 不使用随机增强

类别共 13 类：

```text
apple, binder, coffee_mug, dry_battery, greens, kleenex, lightbulb,
lime, mushroom, notebook, pitcher, sponge, water_bottle
```

## Data Manifests

manifest 位于 `metadata/manifests/`：

- `train.csv`：原始训练划分
- `train_sub.csv`：从训练划分中扣除验证实例后的训练子集
- `val.csv`：验证集，仅用于模型选择和消融比较
- `test.csv`：独立测试集，仅用于最终测试评估

重新生成基础 manifest：

```powershell
python .\ml_repo\scripts\build_manifests.py
```

重新生成训练/验证拆分：

```powershell
python .\ml_repo\scripts\build_validation_split.py
```

检查数据集：

```powershell
python .\ml_repo\scripts\inspect_dataset.py
```

预览 DataLoader：

```powershell
python .\ml_repo\scripts\preview_dataloader.py
```

## Baseline Model

默认基线模型见 [configs/model.yaml](configs/model.yaml)：

```text
Conv feature extractor
→ AdaptiveAvgPool2d((1, 1))
→ Flatten
→ BatchNorm1d
→ Linear
→ ReLU
→ Dropout(0.3)
→ Linear classifier
```

核心结构：

- 输入：`B x 3 x 128 x 128`
- 三个卷积阶段：`32 → 64 → 128`
- 每阶段两层 `Conv2d + BatchNorm2d + ReLU`
- 每阶段后接 `MaxPool2d(2)`
- 输出类别数：13
- 默认可训练参数量：305901

查看模型：

```powershell
python .\ml_repo\scripts\inspect_model.py
```

## Training

默认训练配置见 [configs/train.yaml](configs/train.yaml)。

关键参数：

```yaml
learning_rate: 0.004
epochs: 30
train_batch_size: 32
val_batch_size: 64
optimizer: adamw
loss: cross_entropy
label_smoothing: 0.1
scheduler: reduce_on_plateau
monitor: val_macro_f1
```

运行基线训练：

```powershell
python .\ml_repo\scripts\train.py --config .\ml_repo\configs\train.yaml
```

快速 smoke run：

```powershell
python .\ml_repo\scripts\train.py --config .\ml_repo\configs\train.yaml --epochs 2
```

训练输出默认写入 `../outputs/experiments1/final/`，包括：

- `best_model.pt`
- `last_model.pt`
- `history.csv`
- `training_summary.json`
- `val_metrics.json`
- 验证集预测、混淆矩阵和曲线图

## Ablation Experiments

消融配置位于 `configs/ablations/`：

| Experiment | Pooling | Dropout | LR | Batch | Epochs |
|---|---|---:|---:|---:|---:|
| `baseline_gap_dropout` | GAP | 0.3 | 0.004 | 32 | 30 |
| `flatten_dropout` | Flatten | 0.3 | 0.004 | 32 | 30 |
| `gap_no_dropout` | GAP | 0.0 | 0.004 | 32 | 30 |

批量运行：

```powershell
python .\ml_repo\scripts\run_ablations.py
```

强制重跑：

```powershell
python .\ml_repo\scripts\run_ablations.py --force
```

单独运行：

```powershell
python .\ml_repo\scripts\train_ablation.py --config .\ml_repo\configs\ablations\baseline_gap_dropout.yaml
python .\ml_repo\scripts\train_ablation.py --config .\ml_repo\configs\ablations\flatten_dropout.yaml
python .\ml_repo\scripts\train_ablation.py --config .\ml_repo\configs\ablations\gap_no_dropout.yaml
```

汇总消融结果：

```powershell
python .\ml_repo\scripts\summarize_ablations.py
```

消融输出默认写入：

```text
../outputs/ablation/
```

汇总表：

```text
../outputs/ablation/ablation_summary.csv
```

## Held-Out Test Evaluation

独立测试评估只用于最终报告，不用于调参或模型选择。

当前测试配置：

[configs/test/baseline_test.yaml](configs/test/baseline_test.yaml)

评估对象：

```text
baseline_gap_dropout/best_model.pt
```

运行：

```powershell
python .\ml_repo\scripts\evaluate_baseline_test.py `
  --config .\ml_repo\configs\test\baseline_test.yaml
```

强制重跑：

```powershell
python .\ml_repo\scripts\evaluate_baseline_test.py `
  --config .\ml_repo\configs\test\baseline_test.yaml `
  --force
```

测试评估会：

- 只加载 `best_model.pt`
- 只使用 `metadata/manifests/test.csv`
- 设置 `model.eval()`
- 使用 `torch.inference_mode()`
- 不创建 optimizer
- 不反向传播
- 不更新模型参数

结果默认写入：

```text
../results/test/baseline_gap_dropout/
```

主要输出：

- `test_metrics.json`
- `test_summary.csv`
- `classification_report.csv`
- `predictions.csv`
- `misclassified_samples.csv`
- `most_confused_class_pairs.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `resolved_test_config.yaml`
- `evaluation.log`

## Visualization

训练曲线图可以根据 `../outputs/ablation/baseline_gap_dropout/history.csv` 生成并保存到：

```text
../outputs/image/
```

当前常用图：

- `baseline_loss_curve.png`
- `baseline_learning_rate_curve.png`
- `baseline_accuracy_curve.png`

注意：训练历史中逐 epoch 曲线包含训练集与验证集指标。独立测试集只在最终 checkpoint 上评估一次，因此不应伪造成逐 epoch test 曲线。

## Tests

运行新增/指定测试：

```powershell
python -m pytest .\ml_repo\tests\test_ablation_models.py -v
python -m pytest .\ml_repo\tests\test_baseline_evaluation.py -v
```

运行全部测试：

```powershell
python -m pytest .\ml_repo\tests -v
```

测试覆盖：

- manifest 和类别映射
- 数据集与 DataLoader
- 预处理几何与张量格式
- CNN 前向/反向
- 训练 smoke test
- 消融模型前向与配置
- Baseline checkpoint 严格加载
- 独立测试集评估指标与输出路径约束

## Reproducibility Notes

- 数据路径通过 `src/utils/paths.py` 按项目根目录解析，不依赖当前 PowerShell 工作目录。
- 验证集来自 `train.csv` 的固定实例拆分，不使用测试集调参。
- 消融实验只使用 `train_sub.csv` 和 `val.csv`。
- 独立测试评估只使用 `test.csv`，且只加载选定好的 `best_model.pt`。
- 大型运行结果建议保存在项目父目录的 `outputs/` 和 `results/` 中，不提交到代码仓库。

