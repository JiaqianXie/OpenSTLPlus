method = 'PhyDNet'
project='video-prediction'
# model
patch_size = 4
# training
lr = 1e-3
batch_size = 16
sched = 'onecycle'
save_best_hook = dict()
wandb_hook = dict()