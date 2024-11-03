method = 'MambaVP'
# model
img_size=64
patch_size=4
embed_dim=192
depth=8
rms_norm=True
residual_in_fp32=True
fused_add_norm=True
num_frames=10
clip_decoder_embed_dim=192
dec_dim=192
# training
lr = 1e-4
batch_size = 6
val_batch_size = 6
drop_path = 0
sched = 'onecycle'
epoch = 200