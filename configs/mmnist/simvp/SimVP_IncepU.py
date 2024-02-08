method = 'SimVP'
project='openstl'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 16
sched = 'onecycle'

# ema_hook = dict(momentum=1e-4, priority='ABOVE_NORMAL')
save_best_hook = dict(priority='ABOVE_NORMAL')
wandb_hook = dict(priority='ABOVE_NORMAL')