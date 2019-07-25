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
        self.g_val_ = tf.Graph()
    
    @property
    def g_train(self):
        return self.g_train_

    @property
    def g_val(self):
        return self.g_val_
    
    def train_graph(self, pointcloud, band_gap):
        # pointnet shape: (batch_size, npoint, num_channel)
        batch_size = pointcloud.get_shape()[0].value
        is_training = tf.constant(True, dtype=bool)
        bn_decay = None
        
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(pointcloud, is_training, bn_decay, K=4)
        pointcloud_transformed = tf.matmul(pointcloud, transform)
        input_point = tf.expand_dims(pointcloud_transformed, -1)
        # (batch_size, npoint, num_channel, 1)

        net = tf_util.conv2d(input_point, 64, [1,4],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
        # (batch_size, npoint, 1, 64)
        net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
        # (batch_size, npoint, 1, 64)

        with tf.variable_scope('transform_net2') as sc: 
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = tf_util.conv2d(net_transformed, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        # (batch_size, npoint, 1, 128)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        # (batch_size, npoint, 1, 1024)
    
        # Symmetric function: max pooling
        net = tf.math.reduce_max(net, axis=1, keepdims=True)
        # (batch_size, 1, 1, 1024)
        net = tf.reshape(net, [batch_size, 1024])
        # (batch_size, 1024)
        
        net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        # (batch_size, 512)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp1')
        # (batch_size, 512)
        net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        # (batch_size, 256)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp2')
        net = tf_util.fully_connected(net, 64, bn=False, is_training=is_training,
                                      scope='fc3', bn_decay=bn_decay)
        # (batch_size, 64)
        y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc4')
        # (batch_size, 1)


        #return tf.reduce_mean(tf.losses.absolute_difference(y_pred, band_gap))
        #return tf.sqrt(tf.reduce_mean((y_pred - band_gap)**2))
        return tf.losses.huber_loss(band_gap, y_pred, delta=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


