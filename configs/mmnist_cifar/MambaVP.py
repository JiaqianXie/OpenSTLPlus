method = 'MambaVP'
# model
img_size=64
patch_size=4
embed_dim=576
depth=32
rms_norm=True
residual_in_fp32=True
fused_add_norm=True
num_frames=10
clip_decoder_embed_dim=576
dec_dim=576
# training
lr = 1e-3
batch_size = 6
val_batch_size = 6
drop_path = 0
sched = 'onecycle'
epoch = 200