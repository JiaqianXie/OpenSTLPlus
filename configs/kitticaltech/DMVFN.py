method = 'DMVFN'
# model
routing_out_channels = 32
in_planes = 4 * 3 + 1 + 4 # the first 1: data channel, the second 1: mask channel, the third 4: flow channel
num_block = 9
num_features = [160, 160, 160, 80, 80, 80, 44, 44, 44]
scale = [4, 4, 4, 2, 2, 2, 1, 1, 1]
training = True
# loss
beta = 0.5
gamma = 0.8
coef = 0.5
# training
lr = 5e-4
batch_size = 16
sched = 'onecycle'
epoch=250
# ema_hook = dict(momentum=1e-4, priority='ABOVE_NORMAL')
save_best_hook = dict(priority='ABOVE_NORMAL')
wandb_hook = dict(priority='ABOVE_NORMAL')