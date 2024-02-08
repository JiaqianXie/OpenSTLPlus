method = 'SimVP'
project='openstl'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 128
hid_T = 512
N_T = 12
N_S = 2
# training
lr = 1e-4
drop_path = 0.2
batch_size = 2  # bs = 4 x 4GPUs
sched = 'onecycle'
epoch = 200
save_best_hook = dict()
wandb_hook = dict()