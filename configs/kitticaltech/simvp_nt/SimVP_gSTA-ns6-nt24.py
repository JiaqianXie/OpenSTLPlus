method = 'SimVP'
project='video-prediction'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_T = 24
N_S = 6
# training
lr = 1e-3
drop_path = 0.2
batch_size = 8
sched = 'onecycle'
save_best_hook = dict()
wandb_hook = dict()