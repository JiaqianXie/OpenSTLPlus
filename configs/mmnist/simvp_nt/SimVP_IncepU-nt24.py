method = 'SimVP'
project="video-prediction"
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
# training
lr = 1e-3
batch_size = 16
sched = 'onecycle'
epoch = 200
save_best_hook = dict()
wandb_hook = dict()

