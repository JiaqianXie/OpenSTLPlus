import argparse
import os.path as osp
import pickle
import glob
import numpy as np
import shutil
from openstl.core import metric
from openstl.utils import print_log, check_dir

parser = argparse.ArgumentParser(description="Memory efficient test script.")
parser.add_argument('ex_name', type=str, help='The experiment name')

args = parser.parse_args()

# Access the ex_name argument
ex_name = args.ex_name

assert osp.exists(f"work_dirs/{ex_name}/intermediate_test_results")

intermediate_test_dir = osp.join(f"work_dirs/{ex_name}/intermediate_test_results")

hparams = pickle.load(open(osp.join(intermediate_test_dir, 'hparams.pkl'), "rb"))
metric_list = pickle.load(open(osp.join(intermediate_test_dir, 'metric_list.pkl'), "rb"))
channel_names = pickle.load(open(osp.join(intermediate_test_dir, 'channel_names.pkl'), "rb"))
spatial_norm = pickle.load(open(osp.join(intermediate_test_dir, 'spatial_norm.pkl'), "rb"))


results_all = {}
for filename in glob.glob(osp.join(intermediate_test_dir, 'output_*.pkl'), recursive=True):
    batch = pickle.load(open(filename, "rb"))
    keys = batch.keys()
    for k in keys:
        if k not in results_all:
            results_all[k] = []
        else:
            results_all[k].append(batch[k])
for k, v in results_all.items():
    results_all[k] = np.concatenate(v, axis=0)

eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
    hparams.test_mean, hparams.test_std, metrics=metric_list,
    channel_names=channel_names, spatial_norm=spatial_norm,
    threshold=hparams.get('metric_threshold', None))

results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

print_log(eval_log)
folder_path = check_dir(osp.join(hparams.save_dir, 'saved'))

for np_data in ['metrics', 'inputs', 'trues', 'preds']:
    np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

print("Removing the intermediate test results")
shutil.rmtree(intermediate_test_dir)
