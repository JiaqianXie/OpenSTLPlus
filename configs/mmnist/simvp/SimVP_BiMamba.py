method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'mamba'
bimamba = True
bimamba_strategy = "add"
hid_S = 64
hid_T = 256
N_T = 4
N_M = 6
N_S = 4
# training
lr = 5e-4
batch_size = 16
drop_path = 0.1
sched = 'cosine'
opt = 'adamw'
epoch = 500
use_augment=False
augment_params={
    "use_mask": False,
    "use_flip": False,
    "use_crop": False,
    "mask_prob": 0.5,
    "max_mask_ratio": 0.05,
    "max_num_masks": 3
}
clip_grad= 0.2
clip_mode="norm"
visualize_data=False