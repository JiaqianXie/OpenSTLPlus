method = 'TAU'
project='video-prediction'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
#####
model_type = 'tau-dynamic-routing'
#####
hid_S = 64
hid_T = 256
# N_T = 6
N_T = 24
N_S = 2
alpha = 0.1
# training
lr = 1e-2
drop_path = 0.1
batch_size = 2
val_batch_size = 1
#####
routing_beta = 0.5
gamma = 0.8
coef = 0.5
routing_out_channels = 64
#####
sched = 'onecycle'
save_best_hook = dict()
wandb_hook = dict()