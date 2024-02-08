method = 'SimVP'
project='openstl'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
# training
lr = 5e-4
batch_size = 8
drop_path = 0
sched = 'cosine'
warmup_lr = 0.0
warmup_epoch = 20
min_lr = 1e-5
epoch = 200
save_best_hook = dict()
wandb_hook = dict()