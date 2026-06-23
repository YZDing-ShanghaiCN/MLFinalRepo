# RGB-D RGB 图像分类数据管线

本项目完成机器学习项目第一阶段的数据管线搭建，目标是把 `rgbd-dataset_eval` 中的 RGB 裁剪图像整理成可复现的 manifest、统计结果、PyTorch Dataset/DataLoader 和预览图。

## 当前阶段范围

已实现数据读取、固定 train/test 实例划分、RGB crop 扫描、确定性预处理、manifest 生成、数据统计、DataLoader 预览和 pytest 测试。当前阶段没有实现 CNN、GAP、分类头、训练循环、模型权重保存或准确率评估。

## 目录结构

```text
ml_repo
├── configs
│   ├── dataset.yaml
│   └── splits.yaml
├── metadata
│   └── manifests
├── outputs
│   └── previews
├── scripts
│   ├── build_manifests.py
│   ├── inspect_dataset.py
│   └── preview_dataloader.py
├── src
│   ├── data
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   ├── manifest.py
│   │   └── transforms.py
│   └── utils
│       ├── config.py
│       └── paths.py
└── tests
    ├── test_dataset.py
    └── test_transforms.py
```

## 数据集路径约定

默认数据集路径写在 `configs/dataset.yaml` 中：

```yaml
dataset:
  root: ../rgbd-dataset_eval
```

该相对路径始终相对于 `ml_repo` 项目根目录解析，不依赖运行脚本时的当前工作目录。原始数据集位于仓库外部，不会被复制进仓库。

## 类别

共 13 类：`apple`、`binder`、`coffee_mug`、`dry_battery`、`greens`、`kleenex`、`lightbulb`、`lime`、`mushroom`、`notebook`、`pitcher`、`sponge`、`water_bottle`。

## 划分方式

`configs/splits.yaml` 按物体实例文件夹固定划分 train/test，而不是按图片随机划分。这样可以避免同一个实例的相邻视角同时出现在训练集和测试集，减少数据泄漏。

## 图像筛选规则

实际数据目录中每个实例文件夹包含 `_crop.png`、`_depthcrop.png`、`_maskcrop.png` 和 `_loc.txt`。本阶段只读取普通 RGB crop 图像：扩展名为 `.png/.jpg/.jpeg/.bmp`，文件名包含 `crop`，并排除包含 `depth` 或 `mask` 的文件，匹配大小写不敏感。

## 预处理流程

所有输入图像使用 PIL 读取并强制转为 RGB，然后长边缩放到 128，保持宽高比，短边中心 padding 到 128。padding 默认使用黑色 RGB=(0,0,0)。最终输出 `torch.float32`、通道顺序为 `C×H×W`、像素范围为 `[0,1]` 的 `3×128×128` Tensor。train 和 test 使用完全相同的确定性预处理。

## Windows CPU 运行方式

```powershell
cd C:\Users\dyz18\Desktop\code\mlfinal\ml_repo
```

如果使用本机的 conda 环境：

```powershell
conda activate mlearn
```

若当前 Windows/conda 组合在导入 PyTorch 时提示 `libiomp5md.dll already initialized`，可仅在本次终端会话中设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

安装依赖：

```powershell
pip install -r requirements.txt
```

生成 manifest 和统计：

```powershell
python scripts\build_manifests.py
```

重新扫描并检查数据统计：

```powershell
python scripts\inspect_dataset.py
```

生成 DataLoader 预览：

```powershell
python scripts\preview_dataloader.py
```

运行测试：

```powershell
pytest tests -v
```

## 第一阶段预期输出

```text
metadata/manifests/train.csv
metadata/manifests/test.csv
metadata/class_to_idx.json
metadata/dataset_stats.json
outputs/previews/train_batch.png
outputs/previews/test_batch.png
outputs/previews/resize_pad_examples.png
```

预期扫描数量为 train 6894、test 2284。脚本会打印实际数量、预期数量和差异；如果不一致，不会伪造结果，会按实际有效文件继续生成统计。

## 第二阶段：CNN-GAP 分类基线

第二阶段在第一阶段数据管线之上增加一个端到端训练的浅层 CNN-GAP 分类基线。模型输入为 `B×3×128×128`，包含 3 个卷积阶段：`3→32`、`32→64`、`64→128`，每个阶段使用两层 `Conv2d + BatchNorm + ReLU`，随后 `MaxPool2d(2)`。CNN 输出经过 `AdaptiveAvgPool2d((1, 1))` 做 Global Average Pooling，再展平为 128 维特征，送入 `BatchNorm1d + Linear + ReLU + Dropout + Linear` 分类头，输出 13 类 logits。

GAP 用于把空间特征压缩为固定长度向量，减少分类头参数量，并让模型关注通道级响应。GAP 没有可训练参数，但位于计算图中，梯度会从分类头继续回传到 CNN。模型最后不包含 Softmax，训练时直接使用 `CrossEntropyLoss`。

validation 从原 train manifest 中按物体实例划分，test 不参与调参，也不会在训练脚本中做 epoch 级评估。最佳模型只根据 `val_macro_f1` 保存。

### 第二阶段命令

```powershell
cd C:\Users\dyz18\Desktop\code\mlfinal\ml_repo
conda activate mlearn
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python scripts\build_validation_split.py
python scripts\inspect_model.py
python scripts\overfit_one_batch.py
python scripts\train.py --config configs\train.yaml
python scripts\evaluate.py --checkpoint outputs\experiments\baseline_cnn_gap\best_model.pt --split val
pytest tests -v
```

CPU smoke training 可以覆盖 epoch 数，不修改正式配置：

```powershell
python scripts\train.py --config configs\train.yaml --epochs 2
```

最终 test 评估请单独显式执行，不要用于调参：

```powershell
python scripts\evaluate.py --checkpoint outputs\experiments\baseline_cnn_gap\best_model.pt --split test
```

### 第二阶段输出

实验输出位于 `outputs/experiments/baseline_cnn_gap`，包括 `best_model.pt`、`last_model.pt`、`history.csv`、`training_summary.json`、`train_log.txt`、loss/accuracy/macro F1/learning rate 曲线、`val_metrics.json`、`val_predictions.csv`、`val_confusion_matrix.csv/png`、`val_per_class_metrics.csv`，以及配置快照 `config_snapshot/`。

训练和评估指标包括 loss、accuracy、macro precision、macro recall、macro F1；验证和测试额外输出每类别 precision/recall/F1/support 和混淆矩阵。混淆矩阵横轴为 predicted，纵轴为 true。
