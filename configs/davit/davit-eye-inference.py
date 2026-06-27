# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain.evaluation.metrics.auc import AUC
from mmpretrain.evaluation.metrics.single_label import (
    Accuracy, ConfusionMatrix, SingleLabelMetric,
    calculate_binary_classification_metrics)
from mmpretrain.utils import (build_inference_output_paths,
                              generate_gradcam_heatmaps,
                              save_inference_metrics_report,
                              validate_gradcam_method)


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_CI_LEVEL = 0.95
DEFAULT_CI_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CI_SEED = 0


def collect_images(image_paths, image_dir):
    images = [str(Path(path)) for path in image_paths]

    if image_dir is not None:
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f'Image directory does not exist: {image_dir}')

        dir_images = sorted(
            path for path in image_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        images.extend(str(path) for path in dir_images)

    if not images:
        raise ValueError('No test images found. Pass images or --image-dir.')

    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'Image files do not exist: {missing}')

    return images


def build_class_to_idx(image_dir):
    if image_dir is None:
        return None

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None

    class_names = sorted(
        path.name for path in image_dir.iterdir() if path.is_dir())
    if len(class_names) <= 1:
        return None

    return {class_name: idx for idx, class_name in enumerate(class_names)}


def resolve_binary_positive_label(class_names, positive_class):
    if len(class_names) != 2:
        return 1
    if positive_class not in class_names:
        raise ValueError(
            f'Positive class "{positive_class}" not found in {class_names}.')
    return class_names.index(positive_class)


def build_eval_tensors(image_paths, results, class_to_idx):
    if class_to_idx is None:
        return None, None

    gt_labels = []
    pred_scores = []
    for image_path, result in zip(image_paths, results):
        gt_class = Path(image_path).parent.name
        gt_label = class_to_idx.get(gt_class)
        scores = result.get('pred_scores')
        if gt_label is None or scores is None:
            continue
        gt_labels.append(gt_label)
        pred_scores.append(scores)

    if not gt_labels:
        return None, None

    return torch.tensor(np.stack(pred_scores)), torch.tensor(gt_labels)


def evaluate_with_mmpretrain(pred_scores,
                             gt_labels,
                             roc_path,
                             with_auc=True,
                             positive_label=1,
                             smooth_roc=False,
                             smooth_roc_points=400):
    metrics = {}

    accuracy = Accuracy.calculate(pred_scores, gt_labels, topk=(1, ), thrs=(0., ))
    metrics['top1'] = float(accuracy[0][0].item())

    confusion_matrix = ConfusionMatrix.calculate(pred_scores, gt_labels)
    metrics['confusion_matrix'] = confusion_matrix.cpu().numpy()
    if confusion_matrix.size(0) == 2:
        binary_metrics = calculate_binary_classification_metrics(
            confusion_matrix, positive_label=positive_label)
        metrics['precision'] = float(binary_metrics['precision'].item())
        metrics['recall'] = float(binary_metrics['recall'].item())
        metrics['f1-score'] = float(binary_metrics['f1-score'].item())
    else:
        precision, recall, f1_score, _ = SingleLabelMetric.calculate(
            pred_scores, gt_labels, thrs=(0., ), average='macro')[0]
        metrics['precision'] = float(precision.item())
        metrics['recall'] = float(recall.item())
        metrics['f1-score'] = float(f1_score.item())

    if with_auc:
        auc_metric = AUC(
            average='macro',
            plot_roc=True,
            roc_save_path=roc_path,
            positive_label=positive_label,
            smooth_display=smooth_roc,
            smooth_display_points=smooth_roc_points)
        auc_results = [{
            'pred_score': score.cpu().numpy(),
            'gt_label': int(label.item())
        } for score, label in zip(pred_scores, gt_labels)]
        metrics.update(auc_metric.compute_metrics(auc_results))

    return metrics


def calculate_metric_values(pred_scores,
                            gt_labels,
                            class_names,
                            positive_class,
                            with_auc=True):
    """Calculate inference metrics without writing side-effect artifacts."""
    metrics = {}

    pred_scores = torch.as_tensor(pred_scores)
    gt_labels = torch.as_tensor(gt_labels)

    accuracy = Accuracy.calculate(pred_scores, gt_labels, topk=(1, ),
                                  thrs=(0., ))
    metrics['top1'] = float(accuracy[0][0].item())

    confusion_matrix = ConfusionMatrix.calculate(pred_scores, gt_labels)

    if len(class_names) == 2:
        positive_label = resolve_binary_positive_label(class_names,
                                                       positive_class)
        binary_metrics = calculate_binary_classification_metrics(
            confusion_matrix, positive_label=positive_label)
        metrics['precision'] = float(binary_metrics['precision'].item())
        metrics['recall'] = float(binary_metrics['recall'].item())
        metrics['f1-score'] = float(binary_metrics['f1-score'].item())
        metrics['sensitivity'] = float(binary_metrics['sensitivity'].item())
        metrics['specificity'] = float(binary_metrics['specificity'].item())
    else:
        precision, recall, f1_score, _ = SingleLabelMetric.calculate(
            pred_scores, gt_labels, thrs=(0., ), average='macro')[0]
        metrics['precision'] = float(precision.item())
        metrics['recall'] = float(recall.item())
        metrics['f1-score'] = float(f1_score.item())

    if with_auc and len(torch.unique(gt_labels)) >= 2:
        try:
            positive_label = resolve_binary_positive_label(
                class_names, positive_class)
            auc_metric = AUC(
                average='macro',
                plot_roc=False,
                positive_label=positive_label)
            auc_results = [{
                'pred_score': score.cpu().numpy(),
                'gt_label': int(label.item())
            } for score, label in zip(pred_scores, gt_labels)]
            metrics.update(auc_metric.compute_metrics(auc_results))
        except ValueError:
            pass

    return metrics


def calculate_bootstrap_confidence_intervals(pred_scores,
                                             gt_labels,
                                             class_names,
                                             positive_class,
                                             ci_level=DEFAULT_CI_LEVEL,
                                             bootstrap_samples=(
                                                 DEFAULT_CI_BOOTSTRAP_SAMPLES),
                                             seed=DEFAULT_CI_SEED,
                                             with_auc=True):
    """Calculate percentile bootstrap confidence intervals for metrics."""
    pred_scores_np = torch.as_tensor(pred_scores).cpu().numpy()
    gt_labels_np = torch.as_tensor(gt_labels).cpu().numpy()
    num_samples = len(gt_labels_np)

    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0

    point_metrics = calculate_metric_values(
        pred_scores_np,
        gt_labels_np,
        class_names,
        positive_class,
        with_auc=with_auc)
    bootstrap_values = {metric_name: [] for metric_name in point_metrics}

    for _ in range(bootstrap_samples):
        sample_indices = rng.integers(0, num_samples, num_samples)
        sample_metrics = calculate_metric_values(
            pred_scores_np[sample_indices],
            gt_labels_np[sample_indices],
            class_names,
            positive_class,
            with_auc=with_auc)
        for metric_name, metric_value in sample_metrics.items():
            if metric_name in bootstrap_values:
                bootstrap_values[metric_name].append(metric_value)

    intervals = {}
    for metric_name, values in bootstrap_values.items():
        if not values:
            continue
        lower, upper = np.percentile(values,
                                     [alpha * 100.0, (1.0 - alpha) * 100.0])
        intervals[metric_name] = dict(
            low=float(lower),
            high=float(upper),
            valid_samples=len(values),
        )

    return dict(
        level=ci_level,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        metrics=intervals,
    )


def add_confidence_intervals(metrics,
                             pred_scores,
                             gt_labels,
                             class_names,
                             positive_class,
                             ci_level,
                             bootstrap_samples,
                             seed,
                             with_auc=True):
    metrics['confidence_intervals'] = calculate_bootstrap_confidence_intervals(
        pred_scores,
        gt_labels,
        class_names,
        positive_class,
        ci_level=ci_level,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        with_auc=with_auc)


def add_binary_metrics(metrics, class_names, positive_class):
    if len(class_names) != 2:
        metrics['binary_metrics_message'] = (
            'Sensitivity/specificity skipped: need binary classes.')
        return

    if positive_class not in class_names:
        metrics['binary_metrics_message'] = (
            f'Sensitivity/specificity skipped: positive class '
            f'"{positive_class}" not found in {class_names}.')
        return

    positive_label = class_names.index(positive_class)
    binary_metrics = calculate_binary_classification_metrics(
        metrics['confusion_matrix'], positive_label=positive_label)
    metrics['positive_class'] = positive_class
    metrics['precision'] = float(binary_metrics['precision'].item())
    metrics['recall'] = float(binary_metrics['recall'].item())
    metrics['f1-score'] = float(binary_metrics['f1-score'].item())
    metrics['sensitivity'] = float(binary_metrics['sensitivity'].item())
    metrics['specificity'] = float(binary_metrics['specificity'].item())


def format_metric_with_ci(metrics, metric_name):
    value = metrics[metric_name]
    ci_block = metrics.get('confidence_intervals', {})
    ci = ci_block.get('metrics', {})
    if metric_name not in ci:
        return f'{value:.4f}'
    interval = ci[metric_name]
    ci_percent = ci_block.get('level', DEFAULT_CI_LEVEL) * 100.0
    return (
        f'{value:.4f}  {ci_percent:.0f}% CI '
        f'[{interval["low"]:.4f}, {interval["high"]:.4f}]')


def print_metrics(metrics, class_names, roc_path):
    print(f'\nFinal accuracy/top1: {format_metric_with_ci(metrics, "top1")}')
    print('positive-class/precision: '
          f'{format_metric_with_ci(metrics, "precision")}')
    print('positive-class/recall: '
          f'{format_metric_with_ci(metrics, "recall")}')
    print('positive-class/f1-score: '
          f'{format_metric_with_ci(metrics, "f1-score")}')
    print('confusion_matrix/result:')
    print(metrics['confusion_matrix'])
    if 'sensitivity' in metrics:
        print(f'sensitivity: {format_metric_with_ci(metrics, "sensitivity")}')
        print(f'specificity: {format_metric_with_ci(metrics, "specificity")}')
        print(f'positive class: {metrics["positive_class"]}')
    elif 'binary_metrics_message' in metrics:
        print(metrics['binary_metrics_message'])
    auc_key = 'auc' if 'auc' in metrics else 'auc_macro'
    if auc_key in metrics:
        print(f'{auc_key}: {format_metric_with_ci(metrics, auc_key)}')
        if len(class_names) == 2:
            positive_class = metrics.get('positive_class', class_names[1])
            positive_label = class_names.index(positive_class)
            print(
                f'ROC/AUC positive class: {positive_class} '
                f'(class index {positive_label}).')
        print(f'ROC curve saved to: {roc_path}')
    print(f'classes: {class_names}')


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    model_name = 'davit-eye'
    default_checkpoint = str(repo_root / 'work_dirs' / 'davit-eye' /
                             'epoch_100.pth')
    default_image_dir = str(repo_root / 'data' / 'eye_area' / 'test')

    parser = ArgumentParser(
        description='Run DaViT eye classification inference on one or more images.')
    parser.add_argument(
        'images',
        nargs='*',
        help='Image files to test.')
    parser.add_argument(
        '--image-dir',
        type=str,
        default=default_image_dir,
        help='Directory to scan for test images.')
    parser.add_argument(
        '--model',
        default=str(script_dir / 'davit-eye.py'),
        help='Model config path.')
    parser.add_argument(
        '--checkpoint',
        default=default_checkpoint,
        help='Checkpoint path. Defaults to the latest/best checkpoint in work_dirs/davit-eye if found.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference, for example cpu or cuda:0.')
    parser.add_argument(
        '--batch-size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument(
        '--show-dir',
        type=str,
        help='Optional directory to save visualization results.')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='Base directory for timestamped inference outputs.')
    parser.add_argument(
        '--positive-class',
        default='abnormal',
        help='Positive class name for ROC/AUC computation.')
    parser.add_argument(
        '--roc-path',
        type=str,
        help='Path to save the ROC curve image.')
    parser.add_argument(
        '--metrics-out',
        type=str,
        help='Path to save the inference metrics text report.')
    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help='Disable default Grad-CAM heatmap generation.')
    parser.add_argument(
        '--heatmap-dir',
        type=str,
        help='Directory to save Grad-CAM heatmaps.')
    parser.add_argument(
        '--heatmap-method',
        default='GradCAM',
        help='Grad-CAM method name.')
    parser.add_argument(
        '--heatmap-target-layers',
        nargs='+',
        help='Explicit model layer names for Grad-CAM.')
    parser.add_argument(
        '--heatmap-eigen-smooth',
        action='store_true',
        help='Enable eigen smoothing for Grad-CAM.')
    parser.add_argument(
        '--heatmap-aug-smooth',
        action='store_true',
        help='Enable augmentation smoothing for Grad-CAM.')
    parser.add_argument(
        '--heatmap-vit-like',
        action='store_true',
        help='Treat Grad-CAM target features as ViT-like tokens.')
    parser.add_argument(
        '--heatmap-num-extra-tokens',
        type=int,
        help='Number of extra tokens for ViT-like Grad-CAM features.')
    parser.add_argument(
        '--raw-roc',
        action='store_true',
        help='Use the raw ROC curve. This is the default.')
    parser.add_argument(
        '--smooth-roc',
        action='store_true',
        help='Enable display-only ROC smoothing.')
    parser.add_argument(
        '--smooth-roc-points',
        type=int,
        default=400,
        help='Number of points used for display-only ROC smoothing.')
    parser.add_argument(
        '--no-ci',
        action='store_true',
        help='Disable bootstrap confidence intervals for labeled inference.')
    parser.add_argument(
        '--ci-bootstrap-samples',
        type=int,
        default=DEFAULT_CI_BOOTSTRAP_SAMPLES,
        help='Number of bootstrap resamples for confidence intervals.')
    parser.add_argument(
        '--ci-seed',
        type=int,
        default=DEFAULT_CI_SEED,
        help='Random seed for bootstrap confidence intervals.')
    parser.add_argument(
        '--ci-level',
        type=float,
        default=DEFAULT_CI_LEVEL,
        help='Confidence level for bootstrap intervals.')
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError(
            'No checkpoint found. Pass --checkpoint /path/to/your_checkpoint.pth.')
    if args.ci_bootstrap_samples <= 0:
        raise ValueError('--ci-bootstrap-samples must be positive.')
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError('--ci-level must be between 0 and 1.')

    output_paths = build_inference_output_paths(
        repo_root=repo_root,
        model_name=model_name,
        out_dir=args.out_dir,
        roc_path=args.roc_path,
        show_dir=args.show_dir,
        heatmap_dir=args.heatmap_dir,
        metrics_out=args.metrics_out)
    args.roc_path = str(output_paths['roc_path'])
    args.show_dir = str(output_paths['show_dir'])
    args.heatmap_dir = str(output_paths['heatmap_dir'])
    args.metrics_out = str(output_paths['metrics_path'])
    print(f'Inference output directory: {output_paths["output_dir"]}')
    if not args.no_heatmap:
        validate_gradcam_method(args.heatmap_method)
        args.heatmap_vit_like = True
        if args.heatmap_num_extra_tokens is None:
            args.heatmap_num_extra_tokens = 0

    image_paths = collect_images(args.images, args.image_dir)
    inferencer = ImageClassificationInferencer(
        model=args.model, pretrained=args.checkpoint, device=args.device)
    results = inferencer(
        image_paths,
        batch_size=args.batch_size,
        show_dir=args.show_dir,
        draw_pred=False,
        draw_score=False)
    class_to_idx = build_class_to_idx(args.image_dir)

    for image_path, result in zip(image_paths, results):
        print(
            f'{image_path}\n'
            f'  pred_label: {result["pred_label"]}\n'
            f'  pred_class: {result.get("pred_class", "<unknown>")}\n'
            f'  pred_score: {result["pred_score"]:.6f}')

    heatmap_paths = []
    heatmap_errors = []
    if not args.no_heatmap:
        heatmap_result = generate_gradcam_heatmaps(
            inferencer.model,
            inferencer.config,
            image_paths,
            [result['pred_label'] for result in results],
            output_paths['heatmap_dir'],
            method=args.heatmap_method,
            target_layers=args.heatmap_target_layers,
            device=args.device,
            eigen_smooth=args.heatmap_eigen_smooth,
            aug_smooth=args.heatmap_aug_smooth,
            vit_like=args.heatmap_vit_like,
            num_extra_tokens=args.heatmap_num_extra_tokens)
        heatmap_paths = heatmap_result['paths']
        heatmap_errors = heatmap_result['errors']
        print(f'Heatmaps saved to: {args.heatmap_dir}')
        if heatmap_errors:
            print(f'Heatmap errors: {len(heatmap_errors)}')

    pred_scores, gt_labels = build_eval_tensors(image_paths, results,
                                                class_to_idx)
    metrics = None
    class_names = []
    if pred_scores is not None:
        class_names = [name for name, _ in sorted(
            class_to_idx.items(), key=lambda item: item[1])]
        positive_label = resolve_binary_positive_label(class_names,
                                                       args.positive_class)
        if len(torch.unique(gt_labels)) < 2:
            print('ROC/AUC skipped: need both positive and negative samples.')
            metrics = evaluate_with_mmpretrain(pred_scores, gt_labels,
                                               args.roc_path,
                                               with_auc=False,
                                               positive_label=positive_label)
            add_binary_metrics(metrics, class_names, args.positive_class)
            if not args.no_ci:
                add_confidence_intervals(
                    metrics,
                    pred_scores,
                    gt_labels,
                    class_names,
                    args.positive_class,
                    args.ci_level,
                    args.ci_bootstrap_samples,
                    args.ci_seed,
                    with_auc=False)
            print_metrics(metrics, class_names, args.roc_path)
        else:
            metrics = evaluate_with_mmpretrain(pred_scores, gt_labels,
                                               args.roc_path,
                                               positive_label=positive_label,
                                               smooth_roc=args.smooth_roc,
                                               smooth_roc_points=args
                                               .smooth_roc_points)
            add_binary_metrics(metrics, class_names, args.positive_class)
            if not args.no_ci:
                add_confidence_intervals(
                    metrics,
                    pred_scores,
                    gt_labels,
                    class_names,
                    args.positive_class,
                    args.ci_level,
                    args.ci_bootstrap_samples,
                    args.ci_seed,
                    with_auc=True)
            print_metrics(metrics, class_names, args.roc_path)

    save_inference_metrics_report(
        output_paths['metrics_path'],
        metrics,
        classes=class_names,
        roc_path=args.roc_path,
        show_dir=args.show_dir,
        heatmap_dir=args.heatmap_dir,
        image_dir=args.image_dir,
        checkpoint=args.checkpoint,
        model=args.model,
        num_images=len(image_paths),
        output_dir=output_paths['output_dir'])
    print(f'Metrics report saved to: {args.metrics_out}')


if __name__ == '__main__':
    main()
