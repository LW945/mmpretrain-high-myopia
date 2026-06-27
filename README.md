# High Myopia Eye Classification

This repository is based on MMPreTrain and adds a focused workflow for binary
eye-image classification. The README intentionally documents only the additions
in this fork.

Paper: **Deep Learning-Based Prediction of Peripheral Retinal Abnormalities in
High Myopia from Posterior Pole Crops of Ultra-Widefield Fundus Images**.

## Added workflow

- DaViT-small and ResNet50 configs for eye-area classification.
- Binary positive-class metrics for `abnormal` vs `normal` classification.
- ROC/AUC plotting during validation and test.
- Timestamped inference output directories with metrics reports.
- Grad-CAM heatmap generation for inference results.

## Key files

| File | Purpose |
| --- | --- |
| `configs/davit/davit-eye.py` | DaViT-small training/evaluation config. |
| `configs/resnet/resnet50-eye.py` | ResNet50 training/evaluation config. |
| `configs/davit/davit-eye-inference.py` | DaViT inference, reports, ROC, and heatmaps. |
| `configs/resnet/resnet50-eye-inference.py` | ResNet50 inference, reports, ROC, and heatmaps. |
| `control.sh` | Small wrapper for the added training configs. |
| `mmpretrain/evaluation/metrics/auc.py` | ROC/AUC metric with optional curve saving. |
| `mmpretrain/evaluation/metrics/single_label.py` | Added binary positive-class metrics. |
| `mmpretrain/utils/heatmap.py` | Grad-CAM heatmap helper. |
| `mmpretrain/utils/inference_outputs.py` | Timestamped inference output/report helper. |

## Dataset layout

The added configs expect the dataset under `data/eye_area`:

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

Class folders are read by MMPreTrain's `CustomDataset`. The inference scripts
use `abnormal` as the default positive class for ROC/AUC and binary metrics.

## Pretrained checkpoints

The training configs expect backbone checkpoints in `pretrain/`:

```text
pretrain/
  davit-small.pth
  resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth
```

Set `load_from` or the config checkpoint paths differently if your checkpoints
live elsewhere.

## Train

DaViT is the default:

```bash
bash control.sh train
```

Choose a model explicitly:

```bash
bash control.sh train davit
bash control.sh train resnet
```

Equivalent direct commands:

```bash
python3 tools/train.py configs/davit/davit-eye.py
python3 tools/train.py configs/resnet/resnet50-eye.py
```

Training outputs are written to MMEngine/MMPreTrain `work_dirs` paths such as
`work_dirs/davit-eye/` and `work_dirs/resnet50-eye/`.

## Metrics

The added configs evaluate:

- `Accuracy`
- positive-class `precision`, `recall`, and `f1-score`
- `ConfusionMatrix`
- `AUC` with ROC curve output

Both configs treat class index `0` as the positive label. With the expected
folder names, this corresponds to `abnormal`.

## Inference

Run DaViT inference:

```bash
python3 configs/davit/davit-eye-inference.py \
  --checkpoint work_dirs/davit-eye/epoch_100.pth \
  --image-dir data/eye_area/test
```

Run ResNet50 inference:

```bash
python3 configs/resnet/resnet50-eye-inference.py \
  --checkpoint work_dirs/resnet50-eye/epoch_100.pth \
  --image-dir data/eye_area/test
```

You can also pass individual image paths instead of, or in addition to,
`--image-dir`.

By default, each inference run creates:

```text
inference_work_dir/<model-name>/<timestamp>/
  metrics_test.txt
  roc_curve_test.png
  visualizations/
  heatmap/
```

If `--image-dir` points to a labeled folder tree with class subdirectories, the
script also computes metrics, ROC/AUC, sensitivity, specificity, and bootstrap
confidence intervals. Use `--no-ci` to skip confidence intervals.

## Heatmaps

Grad-CAM heatmaps are enabled by default in the inference scripts. Disable them
when the optional dependency is unavailable or when only numeric predictions are
needed:

```bash
python3 configs/davit/davit-eye-inference.py \
  --checkpoint work_dirs/davit-eye/epoch_100.pth \
  --image-dir data/eye_area/test \
  --no-heatmap
```

The DaViT inference script configures ViT-like Grad-CAM handling automatically.
The ResNet50 script uses standard convolutional feature maps.

## Extra dependencies

The forked additions use packages that may not be present in a minimal
MMPreTrain install:

```bash
pip install scikit-learn matplotlib "grad-cam>=1.3.7,<1.5.0"
```

`grad-cam` is only required for heatmap generation.
