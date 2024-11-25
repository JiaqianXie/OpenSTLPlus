#!/bin/bash
# conda create -n OpenSTLPlus -y python=3.10
# conda activate OpenSTLPlus
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y lightning -c conda-forge
conda install -y -c conda-forge xarray dask netCDF4 bottleneck
python -m pip install timm==0.6.11 --quiet
python -m pip install scikit-image hickle decord fvcore lpips nni einops pandas tqdm wandb dill optuna  --quiet
python -m pip install -e .
