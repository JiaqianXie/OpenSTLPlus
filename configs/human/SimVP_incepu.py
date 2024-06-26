method = 'SimVP'
project_name = 'video-prediction'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
model_type = 'incepu'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 16
val_batch_size = 16
sched = 'onecycle'
warmup_epoch = 0
epoch = 50