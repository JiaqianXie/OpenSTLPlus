method = 'SwinLSTM_B'
project_name = 'video-prediction'
num_nodes = 8
# model
depths = 12
num_heads = 4
patch_size = 2
window_size = 4
embed_dim = 128
# training
lr = 1e-4
batch_size = 1 * num_nodes
val_batch_size = 1 * num_nodes
sched = 'onecycle'
epoch = 50