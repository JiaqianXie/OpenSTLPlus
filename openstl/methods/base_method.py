import os

import numpy as np
import torch.nn as nn
import os.path as osp
import pytorch_lightning as pl
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler
from openstl.core import metric
import pickle
import glob
import shutil
import sys
class Base_method(pl.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }

    def forward(self, batch):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y) * batch_x.shape[-2] * batch_x.shape[-1]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        if self.hparams.mem_efficient_test:
            tmp_dir = osp.join(self.hparams.save_dir, "intermediate_test_results")
            pickle.dump(outputs, open(osp.join(tmp_dir, f"output_{batch_idx}.pkl"), "wb"))
        else:
            self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        if self.hparams.mem_efficient_test:
            tmp_dir = osp.join(self.hparams.save_dir, "intermediate_test_results")
            pickle.dump(self.metric_list, open(osp.join(tmp_dir, "metric_list.pkl"), "wb"))
            pickle.dump(self.channel_names, open(osp.join(tmp_dir, "channel_names.pkl"), "wb"))
            pickle.dump(self.spatial_norm, open(osp.join(tmp_dir, "spatial_norm.pkl"), "wb"))
            pickle.dump(self.hparams, open(osp.join(tmp_dir, "hparams.pkl"), "wb"))
            print("Entering memory efficient test mode. Please run tools/mem_efficient_test.py")
            sys.exit(0)
        else:
            for k in self.test_outputs[0].keys():
                results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
            eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
                self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
                channel_names=self.channel_names, spatial_norm=self.spatial_norm,
                threshold=self.hparams.get('metric_threshold', None))

            results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

            if self.trainer.is_global_zero:
                print_log(eval_log)
                folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

                for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

        return results_all