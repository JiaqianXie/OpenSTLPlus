method = 'SwinLSTM_B'
project_name = 'video-prediction'
# model
depths = 12
num_heads = 4
patch_size = 2
window_size = 4
embed_dim = 128
# training
lr = 1e-4
batch_size = 6
val_batch_size = 6
sched = 'onecycle'
epoch = 50