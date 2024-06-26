method = 'SimVP'
project_name="video-prediction"
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU-weight-sharing'  # SimVP.V1
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 2
# training
lr = 5e-4
drop_path = 0.1
batch_size = 1
val_batch_size = 1
sched = 'onecycle'
epoch = 200
save_best_hook = dict()
wandb_hook = dict()
