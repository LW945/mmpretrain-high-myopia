# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from sklearn.metrics import roc_auc_score, roc_curve, auc

from mmpretrain.registry import METRICS


@METRICS.register_module()
class AUC(BaseMetric):
    """AUC evaluation metric.
    
    Calculate the Area Under the ROC Curve (AUC) for binary or multi-class
    classification tasks.
    
    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        average (str, optional): For multi-class, the averaging strategy.
            Options are 'macro', 'micro', 'weighted', or None. 
            Defaults to 'macro'.
        plot_roc (bool): Whether to plot and save ROC curve. Defaults to False.
        roc_save_path (str): Path to save ROC curve image. 
            Defaults to 'roc_curve.png'.
        save_per_eval (bool): Whether to save ROC curves with a unique
            validation or testing suffix for each evaluation run. Defaults
            to False.
        smooth_display (bool): Whether to smooth the displayed ROC line.
            This only affects visualization and does not change the computed
            AUC or the raw ROC thresholds. Defaults to False.
        smooth_display_points (int): Number of points used for the smoothed
            display curve. Defaults to 400.
        positive_label (int): Positive class label for binary AUC/ROC.
            Defaults to 1.
    """
    default_prefix: Optional[str] = 'auc'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 average: str = 'macro',
                 plot_roc: bool = False,
                 roc_save_path: str = 'roc_curve.png',
                 save_per_eval: bool = False,
                 smooth_display: bool = False,
                 smooth_display_points: int = 400,
                 positive_label: int = 1) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.average = average
        self.plot_roc = plot_roc
        self.roc_save_path = roc_save_path
        self.save_per_eval = save_per_eval
        self.smooth_display = smooth_display
        self.smooth_display_points = smooth_display_points
        self.positive_label = int(positive_label)
        self._roc_eval_index = 0

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples."""
        for data_sample in data_samples:
            result = dict()
            pred_score = data_sample['pred_score']
            gt_label = data_sample['gt_label']
            
            # Convert to numpy if needed
            if isinstance(pred_score, torch.Tensor):
                pred_score = pred_score.cpu().numpy()
            if isinstance(gt_label, torch.Tensor):
                gt_label = gt_label.cpu().item()
            
            result['pred_score'] = pred_score
            result['gt_label'] = gt_label
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results."""
        # Gather all predictions and labels
        pred_scores = np.vstack([res['pred_score'] for res in results])
        gt_labels = np.array([res['gt_label'] for res in results])
        
        # Calculate AUC
        num_classes = pred_scores.shape[1]
        
        if num_classes == 2:
            if self.positive_label < 0 or self.positive_label >= num_classes:
                raise ValueError(
                    f'The positive label index {self.positive_label} is out '
                    f'of range for {num_classes} classes.')

            binary_gt_labels = (gt_labels == self.positive_label).astype(int)
            positive_scores = pred_scores[:, self.positive_label]
            auc_score = roc_auc_score(binary_gt_labels, positive_scores)
            metrics = {'auc': auc_score}
            
            # Plot ROC curve if enabled
            if self.plot_roc:
                self._plot_roc_curve_binary(binary_gt_labels, positive_scores,
                                            auc_score)
        else:
            # Multi-class classification
            if self.average == 'macro':
                auc_score = roc_auc_score(gt_labels, pred_scores, 
                                   multi_class='ovr', average='macro')
                metrics = {'auc_macro': auc_score}
            elif self.average == 'weighted':
                auc_score = roc_auc_score(gt_labels, pred_scores, 
                                   multi_class='ovr', average='weighted')
                metrics = {'auc_weighted': auc_score}
            elif self.average is None:
                auc_per_class = roc_auc_score(gt_labels, pred_scores, 
                                             multi_class='ovr', average=None)
                metrics = {f'auc_class{i}': auc_val for i, auc_val in enumerate(auc_per_class)}
            else:
                raise ValueError(f'Unsupported average type: {self.average}')
            
            # Plot ROC curve if enabled
            if self.plot_roc:
                self._plot_roc_curve_multiclass(gt_labels, pred_scores, num_classes)
        
        return metrics

    def _plot_roc_curve_binary(self, gt_labels, pred_scores, auc_score):
        """Plot ROC curve for binary classification."""
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            print('matplotlib is required for plotting ROC curve. '
                  'Install it with: pip install matplotlib')
            return
        
        fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
        plot_fpr, plot_tpr = self._get_display_curve(fpr, tpr)

        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        ax.plot(plot_fpr, plot_tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_score:.4f})')
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        ax.scatter([optimal_fpr], [optimal_tpr], marker='o', color='red',
                   s=100, label=f'Optimal threshold = {optimal_threshold:.4f}\n'
                   f'(FPR={optimal_fpr:.4f}, TPR={optimal_tpr:.4f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random (AUC = 0.5)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        figure.tight_layout()
        self._save_roc_curve(figure)

    def _plot_roc_curve_multiclass(self, gt_labels, pred_scores, num_classes):
        """Plot ROC curve for multi-class classification."""
        try:
            from matplotlib import colormaps
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from sklearn.preprocessing import label_binarize
        except ImportError:
            print('matplotlib and sklearn are required for plotting ROC curve.')
            return
        
        # Binarize the labels
        gt_labels_bin = label_binarize(gt_labels, classes=range(num_classes))
        
        figure = Figure(figsize=(10, 8))
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        colors = colormaps['tab10'](np.linspace(0, 1, num_classes))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(gt_labels_bin[:, i], pred_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plot_fpr, plot_tpr = self._get_display_curve(fpr, tpr)
            ax.plot(plot_fpr, plot_tpr, color=colors[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.4f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random (AUC = 0.5)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        figure.tight_layout()
        self._save_roc_curve(figure)

    def _get_display_curve(self, fpr: np.ndarray,
                           tpr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optionally smooth the displayed ROC line without changing metrics."""
        if not self.smooth_display or len(fpr) < 3:
            return fpr, tpr

        unique_fpr = np.unique(fpr)
        if len(unique_fpr) < 3:
            return fpr, tpr

        # ROC may contain repeated FPR values from vertical steps. Collapse
        # them to the upper envelope so interpolation has a strictly
        # increasing x-axis.
        unique_tpr = np.array([tpr[fpr == x].max() for x in unique_fpr])
        dense_fpr = np.linspace(unique_fpr[0], unique_fpr[-1],
                                self.smooth_display_points)

        try:
            from scipy.interpolate import PchipInterpolator
            dense_tpr = PchipInterpolator(unique_fpr, unique_tpr)(dense_fpr)
        except ImportError:
            dense_tpr = np.interp(dense_fpr, unique_fpr, unique_tpr)

        dense_tpr = np.clip(np.maximum.accumulate(dense_tpr), 0.0, 1.0)
        return dense_fpr, dense_tpr

    def _save_roc_curve(self, figure) -> None:
        """Save ROC figure to a unique path when requested."""
        save_path = self._resolve_roc_save_path()
        save_dir = osp.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self._log_roc_save_path(save_path)

    def _resolve_roc_save_path(self) -> str:
        """Resolve ROC path for the current evaluation context."""
        if not self.save_per_eval:
            return self.roc_save_path

        self._roc_eval_index += 1
        context = self._get_roc_context()
        save_path = self._build_roc_save_path(context)

        if osp.exists(save_path):
            save_path = self._build_roc_save_path(
                context, suffix=f'eval_{self._roc_eval_index:03d}')

        return save_path

    def _build_roc_save_path(self,
                             context: Dict,
                             suffix: Optional[str] = None) -> str:
        base_dir = osp.dirname(self.roc_save_path)
        base_name = osp.basename(self.roc_save_path)
        stem, ext = osp.splitext(base_name)
        if not ext:
            ext = '.png'

        stem_tokens = set(filter(None, stem.split('_')))
        tokens = []

        phase = context.get('phase')
        if isinstance(phase, str) and phase and phase not in stem_tokens:
            tokens.append(phase)

        step_name = context.get('step_name')
        step = context.get('step')
        if isinstance(step_name, str) and step_name and step is not None:
            tokens.append(f'{step_name}_{step}')
        else:
            tokens.append(f'eval_{self._roc_eval_index:03d}')

        if suffix is not None:
            tokens.append(suffix)

        save_name = '_'.join([stem, *tokens]) + ext
        return osp.join(base_dir, save_name) if base_dir else save_name

    def _get_roc_context(self) -> Dict:
        """Read ROC naming context populated by the runtime hook."""
        try:
            from mmengine.logging import MessageHub
        except ImportError:
            return {}

        try:
            message_hub = MessageHub.get_current_instance()
            context = message_hub.get_info('roc_curve_context')
        except Exception:
            return {}

        return context if isinstance(context, dict) else {}

    def _log_roc_save_path(self, save_path: str) -> None:
        """Log the final ROC save path."""
        try:
            from mmengine.logging import MMLogger
            MMLogger.get_current_instance().info(
                f'ROC curve saved to {save_path}')
        except Exception:
            print(f'ROC curve saved to {save_path}')
