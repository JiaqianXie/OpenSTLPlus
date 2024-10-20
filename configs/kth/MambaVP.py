method = 'MambaVP'
# model
img_size=128
patch_size=16
embed_dim=576
depth=32
rms_norm=True
residual_in_fp32=True
fused_add_norm=True
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'
epoch = 200
