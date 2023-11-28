import warnings
from .hooks import Hook
import wandb


class WandBHook(Hook):
    def __init__(self):
        self.run = None

    def init_wandb(self, name, config):
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project="video-prediction",
            name=name,
            # track hyperparameters and run metadata
            config=config
        )
        pass

    def after_train_epoch(self, runner):
        if self.run is None:
            self.init_wandb(runner.config['ex_name'], runner.config)
        train_loss = runner.train_loss.avg
        wandb.log({"train_loss": train_loss})


    def after_val_epoch(self, runner):
        if self.run is None:
            self.init_wandb(runner.config['ex_name'], runner.config)
        val_result = runner.val_result
        log = dict()
        for key, value in val_result.items():
            if key == "loss":
                log["val_loss"] = value.mean()
            else:
                log[key] = value.mean()
        wandb.log(log)
