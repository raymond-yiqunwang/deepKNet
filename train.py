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
parser.add_argument('--num_channels', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 2]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 1e-3]')
FLAGS = parser.parse_args()


class Trainer(object):
    def __init__(self, **kwargs):
        self.KNet_model = KNetModel(**kwargs)
        self.num_channels = FLAGS.num_channels
        self.batch_size = FLAGS.batch_size
        self.max_epoch = FLAGS.max_epoch
        self.learning_rate = FLAGS.learning_rate
        

    
    def train(self):
        with self.KNet_model.g_train.as_default():
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pointcloud_pl, y_true_pl = self.KNet_model.placeholder_inputs(self.batch_size, self.num_channels)

            # normalizer
            # TODO
    
            y_pred = self.KNet_model.get_model(pointcloud_pl, is_training_pl)
    
            MAELoss = self.KNet_model.get_loss(y_pred, y_true_pl)
    
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(MAELoss)

            # add gradient clipping
            # TODO
            # check nan
            # TODO
    
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
#                    self.eval_one_epoch(sess, ops)

                    if (epoch % 10 == 0):
                        print("Current epoch: {}".format(epoch))

    
    def train_one_epoch(self, sess, ops):                    
        is_training = True

        dataset = load_tfrecords("data/pointcloud_dataset0.tfrecords").batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        idata = iterator.get_next()

        # shuffle input
        # TODO
        # Summary writer
        # TODO

        loss_sum = 0.
        while True:
            try:
                # pointcloud_raw = idata['pointcloud_raw']
                # If you use tf.dataset, it is a part of the graph, so you need to first session run it to get the real data
                # But it is not the best practice.
                # Best one is directly feed it into your train function, and eval the loss
                pointcloud_raw = sess.run(idata['pointcloud_raw'])
                # NEED HELP WITH THIS, HOW TO FEED THIS TO SESS.RUN?
                pointcloud_batch = np.frombuffer(pointcloud_raw, dtype=float).reshape((-1, self.num_channles))
                y_true_batch = ...

                feed_dict_train = { ops["pointcloud_pl"]: pointcloud_batch,
                                    ops["y_true_pl"]: y_true_batch,
                                    ops["is_training_pl"]: is_training
                                  }
                _, loss_val, pred_val = sess.run([ops["optimizer"], ops["MAELoss"], ops["y_pred"]], feed_dict=feed_dict_train)
           
                loss_sum += loss_val

                break # DEBUGGING
            
            except tf.errors.OutOfRangeError:
                print("Training finished...")
                break

#        print("train mean loss: {0:.1%}".format(loss_sum))

    """
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
    """

def load_tfrecords(record_file):
    raw_dataset = tf.data.TFRecordDataset(record_file)

    feature_description = {
        'pointcloud_raw': tf.FixedLenFeature([], tf.string),
        'num_channels': tf.FixedLenFeature([], tf.int64),
        'my_id': tf.FixedLenFeature([], tf.int64),
        'band_gap': tf.FixedLenFeature([], tf.float32),
        'formation_energy_per_atom': tf.FixedLenFeature([], tf.float32),
        'nsites': tf.FixedLenFeature([], tf.int64),
    }

    def _parse_image_function(example_proto):
        return tf.parse_single_example(example_proto, feature_description)

    parsed_data = raw_dataset.map(_parse_image_function)
    
    return parsed_data
    

if __name__ == "__main__":

    # train
    trainer = Trainer()
    trainer.train()

    # test
    # trainer.KNetModel


