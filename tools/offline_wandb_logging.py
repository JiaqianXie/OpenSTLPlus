import os
import re
import wandb
import json

project = "video-prediction"
log_dir = "../logs"

for exp_name in os.listdir(log_dir):
    log_dict = dict()
    config = json.load(open(os.path.join(log_dir, exp_name, "model_param.json")))

    run = wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name=exp_name,
        # track hyperparameters and run metadata
        config=config
    )

    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("mae", step_metric="epoch", summary="min")
    wandb.define_metric("mse", step_metric="epoch", summary="min")

    if os.path.isdir(os.path.join(log_dir, exp_name)):
        for filename in os.listdir(os.path.join(log_dir, exp_name)):
            if ".log" in filename:
                with open(os.path.join(log_dir, exp_name, filename)) as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        line = lines[i]
                        print(line)
                        if "Intermediate result" in line:
                            val_result_str = lines[i-1]
                            loss_result_str = lines[i+1]
                            mse_mae_values = re.findall(r"mse:(\d+\.\d+), mae:(\d+\.\d+)", val_result_str)
                            if mse_mae_values:
                                mse, mae = mse_mae_values[0]
                                mse, mae = float(mse), float(mae)
                            else:
                                print("failed to find mse and mae")
                                mse, mae = None, None
                            extracted_loss_values = re.findall(r"Epoch: (\d+).*Train Loss: ([\d\.]+).*Vali Loss: ([\d\.]+)",
                                                          loss_result_str)
                            if extracted_loss_values:
                                epoch, train_loss, vali_loss = extracted_loss_values[0]
                                epoch = int(epoch)
                                train_loss, vali_loss = float(train_loss), float(vali_loss)
                            else:
                                print("failed to find loss")
                                epoch, train_loss, vali_loss = None, None, None

                            log_dict[epoch] = {
                                "epoch": epoch,
                                "val_loss": vali_loss,
                                "train_loss": train_loss,
                                "mae": mae,
                                "mse": mse
                            }
    sorted_dict = dict(sorted(log_dict.items(), key=lambda item: int(item[0])))
    for epoch, log in sorted_dict.items():
        wandb.log(log)
    wandb.finish()
