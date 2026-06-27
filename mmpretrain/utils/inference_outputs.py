# Copyright (c) OpenMMLab. All rights reserved.
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

_IGNORED_REPORT_METADATA = {'heatmap_paths', 'heatmap_errors'}


def _as_path(path: Optional[str]) -> Optional[Path]:
    return Path(path).expanduser() if path else None


def _make_unique_timestamp_dir(base_dir: Path,
                               timestamp: Optional[str] = None) -> Path:
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f'{timestamp}_{suffix:03d}'
        suffix += 1
    return run_dir


def build_inference_output_paths(repo_root: Path,
                                 model_name: str,
                                 out_dir: Optional[str] = None,
                                 roc_path: Optional[str] = None,
                                 show_dir: Optional[str] = None,
                                 heatmap_dir: Optional[str] = None,
                                 metrics_out: Optional[str] = None,
                                 timestamp: Optional[str] = None) -> dict:
    """Build and create timestamped inference output paths."""
    base_dir = _as_path(out_dir)
    if base_dir is None:
        base_dir = Path(repo_root) / 'inference_work_dir' / model_name

    output_dir = _make_unique_timestamp_dir(base_dir, timestamp=timestamp)
    resolved_roc_path = _as_path(roc_path) or output_dir / 'roc_curve_test.png'
    resolved_show_dir = _as_path(show_dir) or output_dir / 'visualizations'
    resolved_heatmap_dir = _as_path(heatmap_dir) or output_dir / 'heatmap'
    resolved_metrics_path = (
        _as_path(metrics_out) or output_dir / 'metrics_test.txt')

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_show_dir.mkdir(parents=True, exist_ok=True)
    resolved_heatmap_dir.mkdir(parents=True, exist_ok=True)
    resolved_roc_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    return dict(
        output_dir=output_dir,
        roc_path=resolved_roc_path,
        show_dir=resolved_show_dir,
        heatmap_dir=resolved_heatmap_dir,
        metrics_path=resolved_metrics_path)


def to_jsonable(value: Any) -> Any:
    """Convert common numeric containers to JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_inference_metrics_report(metrics: Optional[Mapping[str, Any]],
                                   **metadata) -> dict:
    """Build a normalized inference metrics report."""
    metrics = metrics or {}
    report = {
        'accuracy_top1': metrics.get('top1'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1_score': metrics.get('f1-score'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'sensitivity': metrics.get('sensitivity'),
        'specificity': metrics.get('specificity'),
        'positive_class': metrics.get('positive_class'),
        'binary_metrics_message': metrics.get('binary_metrics_message'),
        'confidence_intervals': metrics.get('confidence_intervals'),
    }

    for auc_key in ('auc', 'auc_macro', 'auc_weighted'):
        if auc_key in metrics:
            report[auc_key] = metrics[auc_key]

    report.update({
        key: value
        for key, value in metadata.items()
        if key not in _IGNORED_REPORT_METADATA
    })
    return to_jsonable({k: v for k, v in report.items() if v is not None})


def _format_sequence(value: list) -> str:
    return ', '.join(str(item) for item in value)


def _format_confusion_matrix(matrix) -> str:
    rows = [[str(item) for item in row] for row in matrix]
    if not rows:
        return ''
    column_count = max(len(row) for row in rows)
    widths = [
        max(len(row[index]) for row in rows if index < len(row))
        for index in range(column_count)
    ]
    formatted_rows = []
    for row in rows:
        cells = [
            row[index].rjust(widths[index])
            for index in range(len(row))
        ]
        formatted_rows.append(f'  [{" ".join(cells)}]')
    return '\n'.join(formatted_rows)


def _format_report_value(value: Any) -> str:
    if isinstance(value, dict) and 'metrics' in value:
        return _format_confidence_intervals(value)
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return '\n' + _format_confusion_matrix(value)
        return _format_sequence(value)
    return str(value)


def _format_confidence_intervals(confidence_intervals: Mapping[str,
                                                              Any]) -> str:
    metric_name_map = {
        'top1': 'accuracy_top1',
        'f1-score': 'f1_score',
    }
    preferred_order = [
        'top1', 'precision', 'recall', 'f1-score', 'auc', 'auc_macro',
        'auc_weighted', 'sensitivity', 'specificity'
    ]
    intervals = confidence_intervals.get('metrics', {})
    level = confidence_intervals.get('level')
    bootstrap_samples = confidence_intervals.get('bootstrap_samples')
    seed = confidence_intervals.get('seed')

    ordered_metric_names = [
        metric_name for metric_name in preferred_order
        if metric_name in intervals
    ]
    ordered_metric_names.extend(
        metric_name for metric_name in sorted(intervals)
        if metric_name not in preferred_order)

    lines = ['']
    if level is not None:
        lines.append(f'  level: {level}')
    if bootstrap_samples is not None:
        lines.append(f'  bootstrap_samples: {bootstrap_samples}')
    if seed is not None:
        lines.append(f'  seed: {seed}')

    for metric_name in ordered_metric_names:
        interval = intervals[metric_name]
        report_metric_name = metric_name_map.get(metric_name, metric_name)
        valid_samples = interval.get('valid_samples')
        sample_text = (
            f' valid_samples={valid_samples}'
            if valid_samples is not None else '')
        lines.append(
            f'  {report_metric_name}: [{interval["low"]}, '
            f'{interval["high"]}]{sample_text}')

    return '\n'.join(lines)


def format_inference_metrics_report(report: Mapping[str, Any]) -> str:
    """Format an inference metrics report as plain text."""
    metric_keys = [
        'accuracy_top1', 'precision', 'recall', 'f1_score', 'auc',
        'auc_macro', 'auc_weighted', 'sensitivity', 'specificity',
        'positive_class', 'binary_metrics_message', 'confidence_intervals',
        'confusion_matrix'
    ]
    lines = ['Inference Metrics Report', '', 'Metrics:']
    has_metric = False
    for key in metric_keys:
        if key not in report:
            continue
        has_metric = True
        lines.append(f'{key}: {_format_report_value(report[key])}')
    if not has_metric:
        lines.append('No metrics available.')

    metadata = {
        key: value
        for key, value in report.items()
        if key not in metric_keys and key not in _IGNORED_REPORT_METADATA
    }
    if metadata:
        lines.extend(['', 'Run:'])
        for key in sorted(metadata):
            lines.append(f'{key}: {_format_report_value(metadata[key])}')

    return '\n'.join(lines) + '\n'


def save_inference_metrics_report(metrics_path: Path,
                                  metrics: Optional[Mapping[str, Any]],
                                  **metadata) -> dict:
    """Save an inference metrics report and return the normalized data."""
    report = build_inference_metrics_report(metrics, **metadata)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', encoding='utf-8') as file:
        file.write(format_inference_metrics_report(report))
    return report
