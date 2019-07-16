#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import os
import sys
import numpy as np
import argparse
from model import KNetModel
import h5py

parser = argparse.ArgumentParser(description='CNN parameters')
parser.add_argument('--max_npoint', type=int, default=744)
parser.add_argument('--num_channels', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
FLAGS = parser.parse_args()

learning_rate = FLAGS.learning_rate
max_epoch = FLAGS.max_epoch

def read_data(path):
    f = h5py.File(path, 'r')
    I_hkl = f['I_hkl'][:]
    band_gap = f['band_gap'][:]
    band_gap = band_gap.reshape(-1, 1)
    return I_hkl, band_gap


class Trainer(object):
    def __init__(self, **kwargs):
        self.KNet_model = KNetModel(**kwargs)
        self.max_npoint = FLAGS.max_npoint
        self.num_channels = FLAGS.num_channels
        self.batch_size = FLAGS.batch_size

    def train(self):
        with self.KNet_model.g_train.as_default():
            x_train, y_true = self.KNet_model.placeholder_inputs(self.max_npoint, self.num_channels)
    
            y_pred = self.KNet_model.get_model(x_train, self.max_npoint, self.num_channels)
    
            loss = self.KNet_model.get_loss(y_pred, y_true)
    
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                ops = {
                    "x_train": x_train,
                    "y_true": y_true,
                    "optimizer": optimizer
                }
                
                for epoch in range(max_epoch):
                    self.train_one_epoch(sess, ops)
                    if (epoch % 10 == 0):
                        print("Current epoch: {}".format(epoch))

    def train_one_epoch(self, sess, ops):                    
        I_hkl, band_gap = read_data(path="data_test/dataset0.h5")

        num_batches = I_hkl.shape[0] // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx+1) * self.batch_size

            x_batch = I_hkl[start_idx:end_idx, :, :]
            y_true_batch = band_gap[start_idx:end_idx]
            
            feed_dict_train = { ops["x_train"]: x_batch,
                            ops["y_true"]: y_true_batch}
            sess.run(ops["optimizer"], feed_dict=feed_dict_train)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()


