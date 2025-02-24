#!/bin/bash

# Define the range and step
START=1
END=100
STEP=10

# Loop through the range with the specified step
for (( INDEX=START; INDEX<=END; INDEX+=STEP ))
do
  echo "Running with index: $INDEX"
  
  docker run --rm --gpus '"device=0"' \
    -e WANDB_API_KEY=44403bf061ca0ba79e9ee1e3a52cc179eea7218c \
    -v ./work_dirs:/app/work_dirs \
    -v ./data:/app/data \
    -v ./configs:/app/configs \
    -v ./openstl:/app/openstl \
    -v ./tools:/app/tools \
    -v ./vis_dirs:/app/vis_dirs \
    --name openstl-vis1 \
    --shm-size=50gb \
    openstlplus \
    python3 tools/visualizations/vis_video.py -d kth -r 10 \
    --work_dirs work_dirs/kth_simvp_incepu \
    --save_dirs vis_dirs/kth_simvp_incepu \
    --index $INDEX
done

echo "All iterations completed."