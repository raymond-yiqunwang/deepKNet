#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import os
import sys
import numpy as np
import argparse
from model import KNetModel
import h5py
import glob

parser = argparse.ArgumentParser(description='CNN parameters')
parser.add_argument('--max_npoint', type=int, default=-1)
parser.add_argument('--num_channels', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 1e-3]')
FLAGS = parser.parse_args()


class Trainer(object):
    def __init__(self, data, **kwargs):
        self.KNet_model = KNetModel(**kwargs)
        self.num_channels = FLAGS.num_channels
        self.batch_size = FLAGS.batch_size
        self.max_epoch = FLAGS.max_epoch
        self.learning_rate = FLAGS.learning_rate
        self.max_npoint = FLAGS.max_npoint
        if self.max_npoint == -1:
            print('--max_npoint must be set according to the input pointcloud data! Exit..')
            sys.exit(1)
        

    
    def train(self):
        with self.KNet_model.g_train.as_default():
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pointcloud_pl, y_true_pl = self.KNet_model.placeholder_inputs(self.batch_size, self.max_npoint, self.num_channels)

            # normalizer
    
            y_pred = self.KNet_model.get_model(pointcloud_pl, is_training_pl)
    
            MAELoss = self.KNet_model.get_loss(y_pred, y_true_pl)
    
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(MAELoss)

            # add gradient clipping

            # check nan
    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer(), feed_dict={is_training_pl: True})
                
                ops = {
                    "is_training_pl": is_training_pl,
                    "pointcloud_pl": pointcloud_pl,
                    "y_true_pl": y_true_pl,
                    "y_pred": y_pred,
                    "MAELoss": MAELoss,
                    "optimizer": optimizer
                }
                
                for epoch in range(self.max_epoch):
                    
                    self.train_one_epoch(sess, ops)
                    self.eval_one_epoch(sess, ops)

                    if (epoch % 10 == 0):
                        print("Current epoch: {}".format(epoch))

    
    def train_one_epoch(self, sess, ops):                    
        is_training = True

        I_hkl, band_gap = read_data(path="data_test/dataset0.h5")

        # shuffle input

        num_batches = I_hkl.shape[0] // self.batch_size

        # Summary writer

        loss_sum = 0.
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx+1) * self.batch_size

            pointcloud_batch = I_hkl[start_idx:end_idx, :, :]
            y_true_batch = band_gap[start_idx:end_idx]
            
            feed_dict_train = { ops["pointcloud_pl"]: pointcloud_batch,
                                ops["y_true_pl"]: y_true_batch,
                                ops["is_training_pl"]: is_training
                              }
            _, loss_val, pred_val = sess.run([ops["optimizer"], ops["MAELoss"], ops["y_pred"]], feed_dict=feed_dict_train)
            
            loss_sum += loss_val

        print("train mean loss: {0:.1%}".format(loss_sum))


    def eval_one_epoch(self, sess, ops):
        is_training = False

        I_hkl, band_gap = read_data(path="data_test/dataset1.h5")

        num_batches = I_hkl.shape[0] // self.batch_size

        loss_sum = 0.
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx+1) * self.batch_size

            pointcloud_batch = I_hkl[start_idx:end_idx, :, :]
            y_true_batch = band_gap[start_idx:end_idx]
            
            feed_dict_train = { ops["pointcloud_pl"]: pointcloud_batch,
                                ops["y_true_pl"]: y_true_batch,
                                ops["is_training_pl"]: is_training
                              }
            _, loss_val, pred_val = sess.run([ops["optimizer"], ops["MAELoss"], ops["y_pred"]], feed_dict=feed_dict_train)

            loss_sum += loss_val
            
        print("eval mean loss: {0:.1%}".format(loss_sum))

            
if __name__ == "__main__":
    # load data 
    with h5py.File("data/pointcloud_set0.h5", 'r') as f:
        x0 = tf.data.Dataset.from_tensor_slices(f['I_hkl'][:100])
    with h5py.File("data/pointcloud_set1.h5", 'r') as f:
        x1 = tf.data.Dataset.from_tensor_slices(f['I_hkl'][:100])
#        band_gap = tf.data.Dataset.from_tensor_slices(f['band_gap'][:100])
    x = x0.concatenate(x1)
    x = x.shuffle()
    print(x)



            

    # train
#    trainer = Trainer(data)
#    trainer.train()

    # test
    # trainer.KNetModel


