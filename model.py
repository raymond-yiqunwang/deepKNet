#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import numpy as np
import glob
import os
import sys
sys.path.append("utils")
import tf_util
from transform_nets import input_transform_net, feature_transform_net

class KNetModel(object):
    def __init__(self):
        self._init_graph()
    
    def _init_graph(self):
        self.g_train_ = tf.Graph()
        self.g_valid_ = tf.Graph()
    
    @property
    def g_train(self):
        return self.g_train_

    @property
    def g_valid(self):
        return self.g_valid_
    

    def gap_func(self, feat, bn_decay=None, is_training=True):

        net = tf.keras.layers.Conv1D(128, 1, activation=tf.nn.relu)(feat)

        # bs x maxlen x 128
#        net = tf.keras.layers.LSTM(128, return_sequences=True)(net)

        net = tf.keras.layers.Conv1D(256, 3, padding="same", activation=tf.nn.relu)(net)
        for i in range(0, 5):
            intmp = net
            net = tf.keras.layers.Conv1D(256, 3, padding="same", dilation_rate=2**i, activation=None)(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net += intmp # residue connection
            net = tf.nn.relu(net)
        
        net = tf.keras.layers.Conv1D(512, 3, padding="same", activation=tf.nn.relu)(net)
        for i in range(0, 5):
            intmp = net
            net = tf.keras.layers.Conv1D(512, 3, padding="same", dilation_rate=2**i, activation=None)(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net += intmp # residue connection
            net = tf.nn.relu(net)

        net = tf.keras.layers.Conv1D(1024, 3, padding="same", activation=tf.nn.relu)(net)
        for i in range(0, 5):
            intmp = net
            net = tf.keras.layers.Conv1D(1024, 3, padding="same", dilation_rate=2**i, activation=None)(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net += intmp # residue connection
            net = tf.nn.relu(net)

        net = tf.math.reduce_max(net, axis=1)
        net = tf.reshape(net, [-1, 1024])
        
        net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 64, bn=False, is_training=is_training,
                                      scope='fc3', bn_decay=bn_decay)
        y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc4')

        return y_pred

    def train_graph(self, pointcloud, band_gap):
        with self.g_train_.as_default():
            bn_decay = None

            y_pred = self.gap_func(pointcloud, bn_decay, True)

            return tf.losses.huber_loss(band_gap, y_pred, delta=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def valid_graph(self, pointcloud, band_gap):
        with self.g_valid_.as_default():
            bn_decay = None

            y_pred = self.gap_func(pointcloud, bn_decay, False)

#            return tf.losses.mean_squared_error(band_gap, y_pred, weights=1.0, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            return tf.losses.absolute_difference(band_gap, y_pred, weights=1.0, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


