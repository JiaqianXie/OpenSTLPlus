#!/bin/bash
conda create -n openstlplus -y python=3.9
conda activate openstlplus
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install hickle decord fvcore lpips nni pandas tqdm wandb --quiet
pip3 install -e .
