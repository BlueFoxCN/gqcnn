#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import cv2
import pdb
from PIL import Image
import tensorflow as tf

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

# try:
#     from .cfgs.config import cfg
# except Exception:
#     from cfgs.config import cfg

try:
    from .train import GQCNN
    from .reader import *
except Exception:
    from train import GQCNN
    from reader import *

def predict_image(model_path, images, depths):

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    sess_init = SaverRestore(model_path)
    model = GQCNN()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "pose"],
                                   output_names=["Softmax"])

    predict_func = OfflinePredictor(predict_config)
    prediction = predict_func(images, depths)

    return prediction

def get_metric(p_prob, n_prob, threshold):
    p_prob = np.array(p_prob)
    n_prob = np.array(n_prob)
    p_pred = (p_prob >= threshold).astype(np.int32)
    n_pred = (n_prob >= threshold).astype(np.int32)

    P  = np.shape(p_pred)[0]
    N  = np.shape(n_prob)[0]
    TP = np.sum(p_pred)
    TN = np.sum(1 - n_pred)
    FP = N - TN
    FN = P - TP
    accuracy  = np.divide((TP + TN), (P + N))
    precision = np.divide(TP, (TP + FP + 1e-7))
    recall    = np.divide(TP, P)
    sensitive = np.divide(TN, N)
    f1_score  = np.divide((2 * recall * precision), (recall + precision + 1e-7))
    metric = np.around(np.array([accuracy, precision, recall, sensitive, f1_score]), 3)
    return p_pred, n_pred, metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.', default='train_log/train_2.8m/model-68976')
    parser.add_argument('--save_dir', help='dir to save imgs', default='debug_predict')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    ds_valid = Data(cfg.data_dir, save_data=True, test_set=True)
    g = ds_valid.get_data()
    num_p = 0
    num_n = 0
    # print(ds_valid.size())

    p_prob, n_prob = [], []
    for j in range(10):
        p_img, p_pose = [], []
        n_img, n_pose = [], []
        for i in range(1000):
            # if (i + 1) % 5000 == 0:
            #     print("%-7d samples" % (5000 * i))
            data = next(g)
            if data[2]:
                p_img.append(data[0])
                p_pose.append(data[1])
            else:
                n_img.append(data[0])
                n_pose.append(data[1])

        num_p += len(p_pose)
        num_n += len(n_pose)
        p_prob_b = predict_image(args.model_path, p_img, p_pose)[0][:,1]
        n_prob_b = predict_image(args.model_path, n_img, n_pose)[0][:,1]
        p_prob = np.append(p_prob, p_prob_b)
        n_prob = np.append(n_prob, n_prob_b)
        print("### %d ###" %j)


    prob  = np.concatenate([p_prob, n_prob])
    label = np.concatenate([np.ones(num_p), np.zeros(num_n)])
    # import pdb
    # pdb.set_trace()
    personr = stats.pearsonr(prob, label)

    print("P: %d\tN: %d\tPersonr:  " % (num_p, num_n), personr)
    print("### metric: [accuracy, precision, recall, sensitive, f1_score]")
    TPR, FPR = [], []
    for threshold in range(0, 105, 5):
        threshold /= 100
        p_pred, n_pred, metric = get_metric(p_prob, n_prob, threshold)
        TPR.append(metric[2])
        FPR.append(1 - metric[3])
        print("threshold = %-5.3f:  "%threshold, metric)
    plt.figure()
    plt.scatter(FPR, TPR, marker='x')
    plt.plot(FPR, TPR)
    plt.savefig('ROC.jpg')


