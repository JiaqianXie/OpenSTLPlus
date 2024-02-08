method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 64
hid_T = 256
N_T = 12
N_S = 2
# training
lr = 1e-3
drop_path = 0.1
batch_size = 4  # bs = 2 x 8GPUs
sched = 'onecycle'

# ema_hook = dict(momentum=1e-4, priority='ABOVE_NORMAL')
save_best_hook = dict(priority='ABOVE_NORMAL')
wandb_hook = dict(priority='ABOVE_NORMAL')