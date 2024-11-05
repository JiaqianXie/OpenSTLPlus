method = 'SwinLSTM_D'
# model
project_name = 'openstl'
depths_downsample = '2,6'
depths_upsample = '6,2'
num_heads = '4,8'
patch_size = 2
window_size = 6
embed_dim = 128
# training
lr = 5e-5
batch_size = 2
sched = 'onecycle'
epoch = 200