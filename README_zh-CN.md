# 高度近视眼部图像分类

本仓库基于 MMPreTrain，新增了一套用于二分类眼部图像分类的工作流。
本文档只说明本 fork 新增的内容。

论文：**Deep Learning-Based Prediction of Peripheral Retinal Abnormalities in
High Myopia from Posterior Pole Crops of Ultra-Widefield Fundus Images**。

## 新增工作流

- 面向眼部区域分类的 DaViT-small 和 ResNet50 配置。
- 用于 `abnormal` 与 `normal` 二分类的阳性类指标。
- 验证和测试阶段的 ROC/AUC 曲线输出。
- 推理阶段按时间戳保存输出目录和指标报告。
- 推理结果的 Grad-CAM 热力图生成。

## 关键文件

| 文件 | 用途 |
| --- | --- |
| `configs/davit/davit-eye.py` | DaViT-small 训练和评估配置。 |
| `configs/resnet/resnet50-eye.py` | ResNet50 训练和评估配置。 |
| `configs/davit/davit-eye-inference.py` | DaViT 推理、报告、ROC 和热力图脚本。 |
| `configs/resnet/resnet50-eye-inference.py` | ResNet50 推理、报告、ROC 和热力图脚本。 |
| `control.sh` | 新增训练配置的简单启动脚本。 |
| `mmpretrain/evaluation/metrics/auc.py` | 支持保存 ROC 曲线的 AUC 指标。 |
| `mmpretrain/evaluation/metrics/single_label.py` | 新增二分类阳性类指标。 |
| `mmpretrain/utils/heatmap.py` | Grad-CAM 热力图工具。 |
| `mmpretrain/utils/inference_outputs.py` | 推理输出目录和指标报告工具。 |

## 数据集结构

新增配置默认读取 `data/eye_area`：

```text
data/eye_area/
  train/
    abnormal/
    normal/
  val/
    abnormal/
    normal/
  test/
    abnormal/
    normal/
```

类别文件夹由 MMPreTrain 的 `CustomDataset` 读取。推理脚本默认使用
`abnormal` 作为 ROC/AUC 和二分类指标的阳性类。

## 预训练权重

训练配置默认从 `pretrain/` 读取 backbone 权重：

```text
pretrain/
  davit-small.pth
  resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth
```

如果权重放在其他位置，请相应修改配置中的 checkpoint 路径或 `load_from`。

## 训练

默认训练 DaViT：

```bash
bash control.sh train
```

也可以显式选择模型：

```bash
bash control.sh train davit
bash control.sh train resnet
```

等价的直接命令：

```bash
python3 tools/train.py configs/davit/davit-eye.py
python3 tools/train.py configs/resnet/resnet50-eye.py
```

训练结果会写入 MMEngine/MMPreTrain 的 `work_dirs` 目录，例如
`work_dirs/davit-eye/` 和 `work_dirs/resnet50-eye/`。

## 指标

新增配置会评估：

- `Accuracy`
- 阳性类 `precision`、`recall` 和 `f1-score`
- `ConfusionMatrix`
- 带 ROC 曲线输出的 `AUC`

两个配置都将类别索引 `0` 作为阳性类。按默认文件夹命名时，类别索引
`0` 对应 `abnormal`。

## 推理

运行 DaViT 推理：

```bash
python3 configs/davit/davit-eye-inference.py \
  --checkpoint work_dirs/davit-eye/epoch_100.pth \
  --image-dir data/eye_area/test
```

运行 ResNet50 推理：

```bash
python3 configs/resnet/resnet50-eye-inference.py \
  --checkpoint work_dirs/resnet50-eye/epoch_100.pth \
  --image-dir data/eye_area/test
```

也可以传入单张或多张图片路径，和 `--image-dir` 一起使用或单独使用。

默认情况下，每次推理会创建：

```text
inference_work_dir/<model-name>/<timestamp>/
  metrics_test.txt
  roc_curve_test.png
  visualizations/
  heatmap/
```

如果 `--image-dir` 指向带类别子目录的有标签测试集，脚本还会计算指标、
ROC/AUC、sensitivity、specificity 和 bootstrap 置信区间。使用 `--no-ci`
可跳过置信区间计算。

## 热力图

推理脚本默认启用 Grad-CAM 热力图。如果没有安装可选依赖，或者只需要数值
预测，可以关闭热力图：

```bash
python3 configs/davit/davit-eye-inference.py \
  --checkpoint work_dirs/davit-eye/epoch_100.pth \
  --image-dir data/eye_area/test \
  --no-heatmap
```

DaViT 推理脚本会自动启用 ViT-like Grad-CAM 处理。ResNet50 推理脚本使用
标准卷积特征图。

## 额外依赖

这些新增功能可能需要最小 MMPreTrain 安装之外的包：

```bash
pip install scikit-learn matplotlib "grad-cam>=1.3.7,<1.5.0"
```

`grad-cam` 只在生成热力图时需要。
