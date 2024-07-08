method = 'SwinLSTM_D'
# model
project_name = 'video-prediction'
depths_downsample = '2,6'
depths_upsample = '6,2'
num_heads = '4,8'
patch_size = 2
window_size = 4
embed_dim = 128
# training
lr = 1e-4
batch_size = 48
val_batch_size = 48
sched = 'onecycle'
epoch = 200