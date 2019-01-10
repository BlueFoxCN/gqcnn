from easydict import EasyDict as edict
import numpy as np

cfg = edict()

# augmentation noise
cfg.gaussian_process_rate = 0.5
cfg.gp_sigma = 0.005
cfg.noise_height = 8
cfg.noise_width = 8

# datapoints
cfg.filename = 'dexnet2.hdf5'
cfg.data_dir = '/home/user/DexNet/gqcnn/dexnet_09_13_17/tensors/'
cfg.train_pct = 0.8
cfg.tot_datapoints = 2759 * 1000 + 60
# cfg.tot_datapoints = 100 * 1000
cfg.train_num = int(cfg.tot_datapoints * cfg.train_pct)
cfg.val_num = cfg.tot_datapoints - cfg.train_num

# image_wise params
np.random.seed(1028)
cfg.train_idxs = np.random.choice(cfg.tot_datapoints, cfg.train_num, replace=False)         # 1033461 / 5383080 = 0.191983
cfg.val_idxs = np.array(list(set(np.arange(cfg.tot_datapoints)) - set(cfg.train_idxs)))     # 259589  / 1345770 = 0.192893

cfg.im_mean = 0.700
cfg.im_std = 0.032
cfg.depth_mean = 0.679
cfg.depth_std = 0.032

cfg.im_height = 32
cfg.im_width = 32

# weight_decay
cfg.weight_decay = 0.0005
# cfg.weight_decay = 0

# lr_decay
cfg.base_lr = 0.001
cfg.decay_step_multiplier = 0.2
cfg.decay_step = cfg.decay_step_multiplier * cfg.train_num
cfg.decay_rate = 0.95
# optimizer
cfg.momentum_rate = 0.9

# train
cfg.epoch_num = 500

# LocalNorm params
cfg.radius = 2
cfg.alpha = 2e-5
cfg.beta = 0.75
cfg.bias = 1.0

# model architecture
cfg.drop_fc3 = False
cfg.fc3_drop_rate = 0
