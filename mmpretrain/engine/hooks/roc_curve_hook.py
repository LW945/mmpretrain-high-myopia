# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class ROCCurveHook(Hook):
    """Record validation and testing context for ROC file naming."""

    def _build_context(self, runner: Runner, phase: str) -> dict:
        context = dict(phase=phase)

        if isinstance(getattr(runner, 'train_loop', None), EpochBasedTrainLoop):
            context.update(step_name='epoch', step=runner.epoch)
        elif hasattr(runner, 'iter'):
            context.update(step_name='iter', step=runner.iter)

        return context

    def before_val_epoch(self, runner: Runner) -> None:
        runner.message_hub.update_info(
            'roc_curve_context', self._build_context(runner, 'val'))

    def before_test_epoch(self, runner: Runner) -> None:
        runner.message_hub.update_info(
            'roc_curve_context', self._build_context(runner, 'test'))
