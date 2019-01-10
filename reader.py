import collections
from scipy import misc
import scipy.stats as ss
import tensorflow as tf

import os, sys, shutil
import numpy as np
import random
import copy
import logging
import cv2
import uuid
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

from tensorpack import *

import h5py

SAVE_DIR = 'input_images'

class Data(RNGDataFlow):
    def __init__(self,
                 data_dir,
                 save_data=False,
                 test_set=False,
                 image_wise=False):

        self.test_set = test_set
        self.save_data = save_data
        self.image_wise = image_wise

        self.data_dir = data_dir

    def size(self):
        return cfg.train_num if not self.test_set else cfg.val_num

    def generate_sample(self, idx):
        '''
        with h5py.File(self.filename, 'r') as f:
            depth_im = f['depth_im'][idx]
            depth = f['hand_depth'][idx]
            label = f['label'][idx]            
        '''
        
        num_1 = idx // 1000
        num_2 = idx % 1000

        depth_im = np.load(self.data_dir + 'depth_ims_tf_table_%05d.npz' % num_1)['arr_0'][num_2,...]
        depth = np.load(self.data_dir + 'hand_poses_%05d.npz' % num_1)['arr_0'][num_2,...][2]
        angle = np.load(self.data_dir + 'hand_poses_%05d.npz' % num_1)['arr_0'][num_2,...][3]
        depth = np.array([depth, angle])
        label = np.load(self.data_dir + 'robust_suction_wrench_resistance_%05d.npz' % num_1)['arr_0'][num_2,...]
        label = int(label > 0.2)

        add_noise = (not self.test_set) and np.random.rand() < cfg.gaussian_process_rate
            
        if add_noise:
            gp_noise = ss.norm.rvs(scale=cfg.gp_sigma, size=cfg.noise_height * cfg.noise_width).reshape(cfg.noise_height, cfg.noise_width)
            gp_noise = misc.imresize(gp_noise, (cfg.im_height, cfg.im_width), interp='bicubic', mode='F')
            gp_noise = np.expand_dims(gp_noise, -1)
            depth_im[depth_im > 0] += gp_noise[depth_im > 0]

        if self.save_data:
            misc.imsave(os.path.join(SAVE_DIR, '%d_%d_depth_im_%d.jpg' % (label, idx, int(add_noise))), depth_im[:,:,0])

        return [depth_im, depth, label]

    def get_data(self):
        if not self.image_wise:
            if not self.test_set:
                idxs = np.arange(cfg.train_num)
                self.rng.shuffle(idxs)
            else:
                idxs = np.arange(cfg.train_num, cfg.tot_datapoints)
        else:
            if not self.test_set:
                idxs = cfg.train_idxs 
                self.rng.shuffle(idxs)
            else:
                idxs = cfg.val_idxs

        for k in idxs:
            retval = self.generate_sample(k)
            if retval == None:
                continue
            yield retval

    def reset_state(self):
        super(Data, self).reset_state() 

if __name__ == '__main__':

    ds = Data(cfg.data_dir, save_data=True)       
    ds.reset_state()

    g = ds.get_data()
    for i in range(100):
        data = next(g)
        import pdb
        pdb.set_trace()
