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
    def __init__(self, batch_size=1, num_channels=3):
        self._init_graph()
        self.batch_size = batch_size
        self.num_channels = num_channels
    
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
        # pointnet batch x 10 x 3
        num_point = pointcloud.get_shape()[1].value
        is_training = True
        bn_decay = None

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(pointcloud, is_training, bn_decay, K=3)
        pointcloud_transformed = tf.matmul(pointcloud, transform)
        input_image = tf.expand_dims(pointcloud_transformed, -1)
        
        net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

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
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
    
        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
    
        net = tf.reshape(net, [self.batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp1')
#        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                      scope='fc2', bn_decay=bn_decay)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp2')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')

        #### linear model for debugging
#        initializer = tf.contrib.layers.xavier_initializer()
#        weights = tf.Variable(initializer([5, self.num_channels]))
#        biases = tf.Variable(0.0)
#        y_pred = np.zeros((self.batch_size)) + biases
#        y_pred = tf.tensordot(pointcloud, weights, axes=[[1,2], [0,1]]) + biases
        #### end of linear model

        return tf.reduce_mean(tf.losses.absolute_difference(y_pred, band_gap))


    def val_graph(self, pointcloud, band_gap):
        # pointnet batch x 10 x 3
        num_point = pointcloud.get_shape()[1].value
        is_training = False
        bn_decay = None

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(pointcloud, is_training, bn_decay, K=3)
        pointcloud_transformed = tf.matmul(pointcloud, transform)
        input_image = tf.expand_dims(pointcloud_transformed, -1)
        
        net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

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
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
    
        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
    
        net = tf.reshape(net, [self.batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp1')
#        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                      scope='fc2', bn_decay=bn_decay)
#        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                              scope='dp2')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        y_pred = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')

        #### linear model for debugging
#        initializer = tf.contrib.layers.xavier_initializer()
#        weights = tf.Variable(initializer([5, self.num_channels]))
#        biases = tf.Variable(0.0)
#        y_pred = np.zeros((self.batch_size)) + biases
#        y_pred = tf.tensordot(pointcloud, weights, axes=[[1,2], [0,1]]) + biases
        #### end of linear model

        return tf.reduce_mean(tf.losses.absolute_difference(y_pred, band_gap))


