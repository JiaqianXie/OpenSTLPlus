import warnings
from .hooks import Hook
from openstl.utils import weights_to_cpu
import torch
import os.path as osp
class SaveBestHook(Hook):
    def __init__(self, metrics=["mse", "mae"], save_lowest=True):
        self.metrics = metrics
        self.save_lowest = save_lowest
        self.metric_value = 100000

    def after_val_epoch(self, runner):
        val_result = runner.val_result
        current_metric_value = 0
        for metric in self.metrics:
            current_metric_value += val_result[metric].mean()
        if current_metric_value < self.metric_value:
            checkpoint = {
                'optimizer': runner.method.model_optim.state_dict(),
                'state_dict': weights_to_cpu(runner.method.model.state_dict()),
                'scheduler': runner.method.scheduler.state_dict()}
            torch.save(checkpoint, osp.join(runner.checkpoints_path, 'best_model.pth'))
            self.metric_value = current_metric_value
            print("\nBest model saved.")
        self.after_epoch(runner)