#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import os
import sys
import numpy as np
import argparse
from model import KNetModel
import h5py

"""
parser = argparse.ArgumentParser(description='CNN parameters')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--filter_size1', type=int, default=5)
parser.add_argument('--num_filters1', type=int, default=16)
parser.add_argument('--filter_size2', type=int, default=5)
parser.add_argument('--num_filters2', type=int, default=36)
parser.add_argument('--fc_size', type=int, default=128)
parser.add_argument('--max_step', type=int, default=1000)
FLAGS = parser.parse_args()
"""

def read_data(path):
    f = h5py.File(path, 'r')
    I_hkl = f['I_hkl'][:]
    band_gap = f['band_gap'][:]
    band_gap = band_gap.reshape(-1, 1)
    return I_hkl, band_gap


class Trainer(object):
    def __init__(self, **kwargs):
        self.KNet_model = KNetModel(**kwargs)

    def train(self):
        I_hkl, band_gap = read_data(path="data_test/dataset0.h5")
        # x shape (None, 744, 4)
        # y shape (None, 1) 
        max_points = 744 
        num_channels = 4
        with self.KNet_model.g_train.as_default():
            x_train, y_true = self.KNet_model.placeholder_inputs(max_points, num_channels)
    
            y_pred = self.KNet_model.get_model(x_train, max_points, num_channels)
    
            loss = self.KNet_model.get_loss(y_pred, y_true)
    
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for iter in range(1):
                    x_batch = I_hkl[10:]
                    x_batch = x_batch.reshape(x_batch.shape[0], -1)
                    y_true_batch = band_gap[10:]
                    feed_dict_train = { x_train: x_batch,
                                        y_true: y_true_batch}
                    sess.run(optimizer, feed_dict=feed_dict_train)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()


