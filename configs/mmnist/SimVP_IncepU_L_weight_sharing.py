method = 'SimVP'
# model
model_type = 'IncepU-weight-sharing'
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
# training
lr = 1e-3
sched = 'onecycle'
save_best_hook = dict()
wandb_hook = dict()
batch_size = 8
