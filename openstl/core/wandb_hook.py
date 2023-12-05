from .hooks import Hook
import wandb


class WandBHook(Hook):
    def __init__(self, metric_config={"mse":"min", "mae":"min"}):
        self.metric_config = metric_config
        self.run = None

    def init_wandb(self, name, config, val_result):
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=config['project'],
            name=name,
            # track hyperparameters and run metadata
            config=config
        )
        wandb.define_metric("epoch")
        for metric in val_result.keys():
            if metric in self.metric_config:
                wandb.define_metric(metric, step_metric="epoch", summary=self.metric_config[metric])
            else:
                wandb.define_metric(metric, step_metric="epoch")

    def after_train_epoch(self, runner):
        train_loss = runner.train_loss.avg
        if self.run is not None:
            wandb.log({"train_loss": train_loss})

    def after_val_epoch(self, runner):
        val_result = runner.val_result
        if self.run is None:
            self.init_wandb(runner.config['ex_name'], runner.config, val_result)
        log = {"epoch": runner.epoch}
        for key, value in val_result.items():
            if key == "loss":
                log["val_loss"] = value.mean()
            else:
                log[key] = value.mean()
        # print(log)
        wandb.log(log)
