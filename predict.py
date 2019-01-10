#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import cv2
import pdb
from PIL import Image
import tensorflow as tf

from grasp_sampling import *
from camera_recv import camera_recv

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

# try:
#     from .cfgs.config import cfg
# except Exception:
#     from cfgs.config import cfg

try:
    from .train import GQCNN
except Exception:
    from train import GQCNN


def predict_image(model_path, images, depths):

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    sess_init = SaverRestore(model_path)
    model = GQCNN()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "pose"],
                                   output_names=["Softmax"])

    predict_func = OfflinePredictor(predict_config)

    # image = cv2.imread('resize_image.png', 0)
    # image = np.expand_dims(image, 0)
    # image = np.expand_dims(image, -1)
    # depth = np.array([500], np.float32)
    # import pdb

    prediction = predict_func(images, depths)
    # pdb.set_trace()

    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--save_dir', help='dir to save imgs', default='debug_predict')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    depth_img = np.load('100.pkl')
    img_mean = np.mean(depth_img)
    # img_mean = np.mean(depth_img[120:360, 160:480])
    samples, binary_img = grasp_sample(depth_img,
                                       grasp_num=10,
                                       save_dir=args.save_dir)
    visualize_grasp(samples, binary_img, save_dir=args.save_dir)
    gqcnn_imgs, gqcnn_depths = align(depth_img,
                                     samples,
                                     table_height=560,
                                     z_num=3, save_dir=args.save_dir)
    # gqcnn_imgs = (gqcnn_imgs - img_mean) / 0.032
    # gqcnn_depths = (gqcnn_depths - np.mean(gqcnn_depths)) / 0.032
    pdb.set_trace()
    prediction = predict_image(args.model_path, gqcnn_imgs, gqcnn_depths)
    pdb.set_trace()
    a = prediction

