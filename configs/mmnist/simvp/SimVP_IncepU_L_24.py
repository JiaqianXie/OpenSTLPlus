method = 'SimVP'
project='openstl'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
# training
lr = 1e-4
batch_size = 16
sched = 'onecycle'

# ema_hook = dict(momentum=1e-4, priority='ABOVE_NORMAL')
save_best_hook = dict(priority='ABOVE_NORMAL')
wandb_hook = dict(priority='ABOVE_NORMAL')