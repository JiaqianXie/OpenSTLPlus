#!/bin/bash

# List of folder names
folders=(
  "kitti_prednet"
  "kitti_SimVP_gSTA_L-nt24_weight_sharing_h100"
  "kitti_SimVP_IncepU-nt24"
  "kitti_SimVP_IncepU-nt8"
  "kitti_SimVP_gSTA_L-nt24"
  "kitti_SimVP_gSTA-nt8"
  "kitti_SimVP_IncepU-nt24_weight_sharing_h100"
  #"mmnist_TAU_nt24"
)

# List of indices
indices=(500 1000 1500 2000 2500 3000)

# Base command
base_command="python tools/visualizations/vis_video.py -d kitticaltech"

# Traverse folders and indices
for folder in "${folders[@]}"; do
  for index in "${indices[@]}"; do
    work_dir="work_dirs/${folder}"
    save_dir="vis_dirs/${folder}"
    command="${base_command} --work_dirs ${work_dir} --save_dirs ${save_dir} --index ${index}"
    echo "Executing: ${command}"
    eval ${command}
  done
done
