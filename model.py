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

    def train_graph(self, pointcloud, band_gap):
        with self.g_train_.as_default():
            # pointnet shape: (batch_size, npoint, num_channels)
            batch_size = pointcloud.get_shape()[0].value
            is_training = tf.constant(True, dtype=bool)
            bn_decay = None
            pointcloud = tf.expand_dims(pointcloud, -1)
            
            net = tf_util.conv2d(pointcloud, 64, [1,123],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
    
            net = tf_util.conv2d(net, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv3', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv4', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 1024, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv5', bn_decay=bn_decay)
        
            net = tf.math.reduce_max(net, axis=1, keepdims=True)
            net = tf.reshape(net, [batch_size, 1024])
            
            net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                          scope='fc1', bn_decay=bn_decay)
    #        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                              scope='dp1')
            net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                          scope='fc2', bn_decay=bn_decay)
    #        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                              scope='dp2')
            net = tf_util.fully_connected(net, 64, bn=False, is_training=is_training,
                                          scope='fc3', bn_decay=bn_decay)
            y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc4')
    
            return tf.losses.huber_loss(band_gap, y_pred, delta=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


    def valid_graph(self, pointcloud, band_gap):
        with self.g_valid_.as_default():
            # pointnet shape: (batch_size, npoint, num_channels)
            batch_size = pointcloud.get_shape()[0].value
            is_training = tf.constant(False, dtype=bool)
            bn_decay = None
            pointcloud = tf.expand_dims(pointcloud, -1)
            
            net = tf_util.conv2d(pointcloud, 64, [1,123],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
    
            net = tf_util.conv2d(net, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv3', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv4', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 1024, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv5', bn_decay=bn_decay)
            #net = tf.keras.layers.BatchNormalization()(net)
        
            net = tf.math.reduce_max(net, axis=1, keepdims=True)
            net = tf.reshape(net, [batch_size, 1024])
            
            net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                          scope='fc1', bn_decay=bn_decay)
    #        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                              scope='dp1')
            net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                          scope='fc2', bn_decay=bn_decay)
    #        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                              scope='dp2')
            net = tf_util.fully_connected(net, 64, bn=False, is_training=is_training,
                                          scope='fc3', bn_decay=bn_decay)
            y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc4')
    
            return tf.losses.mean_squared_error(band_gap, y_pred, weights=1.0, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


