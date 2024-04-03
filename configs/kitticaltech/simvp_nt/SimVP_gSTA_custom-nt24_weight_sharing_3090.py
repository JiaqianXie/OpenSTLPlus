method = 'SimVP'
project='video-prediction'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA-weight-sharing'
hid_S = 64
hid_T = 512
N_T = 24
N_S = 2
# training
lr = 5e-4
batch_size = 1
val_batch_size = 1
drop_path = 0.2
sched = 'onecycle'
epoch = 200
save_best_hook = dict()
wandb_hook = dict()
