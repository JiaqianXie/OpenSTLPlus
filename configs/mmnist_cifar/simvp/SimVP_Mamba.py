method = 'SimVP'
project ='openstl'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'mamba'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'
epoch = 400
save_best_hook = dict()
wandb_hook = dict()