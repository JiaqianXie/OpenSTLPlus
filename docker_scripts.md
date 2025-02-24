Test
docker run --rm --gpus '"device=0"' -e WANDB_API_KEY=44403bf061ca0ba79e9ee1e3a52cc179eea7218c -v ./work_dirs:/app/work_dirs -v ./data:/app/data -v ./configs:/app/configs -v ./openstl:/app/openstl --name openstl-test --shm-size=50gb openstlplus python3 tools/test.py kth work_dirs/kth_vivian_mamba/SimVP_Mamba.py kth_vivian_mamba --checkpoint_name best-epoch=42-val_loss=43.632.ckpt --no_display_method_info --project_name openstl

Train
docker run --rm --gpus '"device=1"' -e WANDB_API_KEY=44403bf061ca0ba79e9ee1e3a52cc179eea7218c -v ./work_dirs:/app/work_dirs -v ./data:/app/data -v ./configs:/app/configs -v ./openstl:/app/openstl --name openstl-train --shm-size=50gb openstlplus python3 tools/train.py kth configs/kth/PhyDNet.py kth_PhyDNet --no_display_method_info --project_name openstl

Visualization
docker run --rm --gpus '"device=0"' -e WANDB_API_KEY=44403bf061ca0ba79e9ee1e3a52cc179eea7218c -v ./work_dirs:/app/work_dirs -v ./data:/app/data -v ./configs:/app/configs -v ./openstl:/app/openstl -v ./tools:/app/tools -v ./vis_dirs:/app/vis_dirs --name openstl-vis --shm-size=50gb openstlplus python3 tools/visualizations/vis_video.py -d kth --index 0 -r 10 --work_dirs work_dirs/kth_vivian_mamba --save_dirs vis_dirs/kth_BiMambaVP


docker run --rm --gpus '"device=0"' -e WANDB_API_KEY=44403bf061ca0ba79e9ee1e3a52cc179eea7218c -v ./work_dirs:/app/work_dirs -v ./data:/app/data -v ./configs:/app/configs -v ./openstl:/app/openstl --name openstl-test --shm-size=50gb openstlplus python3 tools/test.py kth configs/kth/simvp/SimVP_IncepU.py kth_simvp_incepu --checkpoint_name latest.pth --no_display_method_info --project_name openstl