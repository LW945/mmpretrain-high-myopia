# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence

import mmcv
import numpy as np
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.dataset import default_collate
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils import is_norm

from mmpretrain import digit_version
from mmpretrain.registry import TRANSFORMS


def _import_grad_cam():
    try:
        import pkg_resources
        import pytorch_grad_cam as cam
        from pytorch_grad_cam.activations_and_gradients import \
            ActivationsAndGradients
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise ImportError(
            'pytorch-grad-cam is required for heatmap generation. '
            'Install it with: pip install "grad-cam>=1.3.6"') from exc

    return cam, ActivationsAndGradients, show_cam_on_image, pkg_resources


def _method_map():
    cam, _, _, _ = _import_grad_cam()
    method_map = {'gradcam++': cam.GradCAMPlusPlus}
    method_map.update({
        cam_class.__name__.lower(): cam_class
        for cam_class in cam.base_cam.BaseCAM.__subclasses__()
    })
    return method_map


def validate_gradcam_method(method: str = 'GradCAM') -> None:
    """Validate Grad-CAM dependency availability and method name."""
    method_map = _method_map()
    if method.lower() not in method_map:
        raise ValueError(
            f'Invalid heatmap method {method}; supports '
            f'{", ".join(sorted(method_map.keys()))}.')


def _reshape_transform(tensor, model, vit_like=False, num_extra_tokens=None):
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f'Unsupported tensor shape {tensor.shape}.')
    if not vit_like:
        raise ValueError(
            f'The tensor shape is {tensor.shape}; pass --heatmap-vit-like '
            'for ViT-like backbones.')

    num_extra_tokens = num_extra_tokens
    if num_extra_tokens is None:
        num_extra_tokens = getattr(model.backbone, 'num_extra_tokens', 1)
    tensor = tensor[:, num_extra_tokens:, :]
    heat_map_area = tensor.size()[1]
    height, width = to_2tuple(int(math.sqrt(heat_map_area)))
    if height * width != heat_map_area:
        raise ValueError(
            f'The feature token count {heat_map_area} is not a square. '
            'Please pass explicit --heatmap-target-layers or '
            '--heatmap-num-extra-tokens.')
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def _get_layer(layer_str: str, model: nn.Module):
    for name, layer in model.named_modules():
        if name == layer_str:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{layer_str}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))


def _default_target_layers(model: nn.Module,
                           vit_like: bool = False,
                           num_extra_tokens: Optional[int] = None):
    norm_layers = [
        (name, layer)
        for name, layer in model.backbone.named_modules(prefix='backbone')
        if is_norm(layer)
    ]
    if not norm_layers:
        raise ValueError('Cannot find a default norm layer for Grad-CAM.')

    if vit_like:
        num_extra_tokens = num_extra_tokens
        if num_extra_tokens is None:
            num_extra_tokens = getattr(model.backbone, 'num_extra_tokens', 1)
        out_type = getattr(model.backbone, 'out_type', 'avg_featmap')
        if (out_type == 'cls_token' or num_extra_tokens > 0) and \
                len(norm_layers) >= 3:
            return [norm_layers[-3][1]]

    return [norm_layers[-1][1]]


def _safe_heatmap_path(image_path: str, heatmap_dir: Path,
                       used_names: set) -> Path:
    path = Path(image_path)
    stem = path.stem or 'image'
    candidate = heatmap_dir / f'{stem}.png'
    suffix = 1
    while candidate.name in used_names or candidate.exists():
        candidate = heatmap_dir / f'{stem}_{suffix:03d}.png'
        suffix += 1
    used_names.add(candidate.name)
    return candidate


def generate_gradcam_heatmaps(model: nn.Module,
                              config,
                              image_paths: Sequence[str],
                              pred_labels: Sequence[int],
                              heatmap_dir: Path,
                              method: str = 'GradCAM',
                              target_layers: Optional[Sequence[str]] = None,
                              device: str = 'cpu',
                              eigen_smooth: bool = False,
                              aug_smooth: bool = False,
                              vit_like: bool = False,
                              num_extra_tokens: Optional[int] = None) -> dict:
    """Generate Grad-CAM heatmap overlays for image classification results."""
    validate_gradcam_method(method)
    method_map = _method_map()
    method_key = method.lower()

    _, ActivationsAndGradients, show_cam_on_image, pkg_resources = \
        _import_grad_cam()

    heatmap_dir = Path(heatmap_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    transforms = Compose(
        [TRANSFORMS.build(t) for t in config.test_dataloader.dataset.pipeline])
    if target_layers:
        cam_target_layers = [_get_layer(layer, model) for layer in target_layers]
    else:
        cam_target_layers = _default_target_layers(
            model, vit_like=vit_like, num_extra_tokens=num_extra_tokens)

    gradcam = method_map[method_key](
        model=model,
        target_layers=cam_target_layers,
        use_cuda=('cuda' in str(device)),
        reshape_transform=partial(
            _reshape_transform,
            model=model,
            vit_like=vit_like,
            num_extra_tokens=num_extra_tokens))
    gradcam.activations_and_grads.release()
    gradcam.activations_and_grads = ActivationsAndGradients(
        gradcam.model, gradcam.target_layers, gradcam.reshape_transform)

    grad_cam_version = pkg_resources.get_distribution('grad_cam').version
    use_output_target = digit_version(grad_cam_version) >= digit_version(
        '1.3.7')
    if use_output_target:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    heatmap_paths = []
    heatmap_errors = []
    used_names = set()
    try:
        for image_path, pred_label in zip(image_paths, pred_labels):
            out_path = _safe_heatmap_path(image_path, heatmap_dir, used_names)
            try:
                data = transforms({'img_path': image_path})
                src_img = copy.deepcopy(data['inputs']).numpy().transpose(
                    1, 2, 0)
                model_inputs = model.data_preprocessor(
                    default_collate([data]), False)
                targets = [ClassifierOutputTarget(int(pred_label))] \
                    if use_output_target else [int(pred_label)]
                grayscale_cam = gradcam(
                    model_inputs['inputs'],
                    targets,
                    eigen_smooth=eigen_smooth,
                    aug_smooth=aug_smooth)
                src_img = np.float32(src_img) / 255
                visualization_img = show_cam_on_image(
                    src_img, grayscale_cam[0, :], use_rgb=False)
                mmcv.imwrite(visualization_img, str(out_path))
                heatmap_paths.append(str(out_path))
            except Exception as exc:
                heatmap_errors.append({
                    'image': str(image_path),
                    'error': str(exc),
                })
    finally:
        gradcam.activations_and_grads.release()

    return dict(paths=heatmap_paths, errors=heatmap_errors)
