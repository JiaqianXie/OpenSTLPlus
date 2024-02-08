method = 'SimVP'
# model
project='openstl'
model_type = 'incepU-weight-sharing'
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 128
hid_T = 512
N_T = 24
N_S = 4
# training
lr = 1e-4
sched = 'onecycle'
batch_size = 4
save_best_hook = dict()
wandb_hook = dict()