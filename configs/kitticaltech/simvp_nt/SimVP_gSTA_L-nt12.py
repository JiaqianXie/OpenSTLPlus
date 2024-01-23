method = 'SimVP'
project='video-prediction'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
# hid_S = 128
# hid_T = 1024
# N_T = 24
# N_S = 4

hid_S = 128
hid_T = 512
N_T = 12
N_S = 4

# training
lr = 5e-3
batch_size = 4
drop_path = 0.2
# sched = 'cosine'
# warmup_lr = 1e-5
# warmup_epoch = 20
# min_lr = 1e-5
sched = 'onecycle'
epoch = 200
save_best_hook = dict()
wandb_hook = dict()