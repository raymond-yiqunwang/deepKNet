#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import numpy as np
#import tf_util

class KNetModel(object):
    def __init__(self):
        self._init_graph()
    
    def _init_graph(self):
        self.g_train_ = tf.Graph()
    
    @property
    def g_train(self):
        return self.g_train_
    
    def placeholder_inputs(self, batch_size, max_npoint, num_channels):
        pointcloud_pl = tf.placeholder(tf.float32, shape=(batch_size, max_npoint, num_channels))
        y_true_pl = tf.placeholder(tf.float32, shape=(batch_size))
        return pointcloud_pl, y_true_pl
    
    def get_model(self, pointcloud, is_training):
        # input  -- (BATCH_SIZE, MAX_NPOINT, NUM_CHANNELS)
        # output -- (BATCH_SIZE, 1)
        batch_size = pointcloud.get_shape()[0].value
        max_npoint = pointcloud.get_shape()[1].value
        num_channels = pointcloud.get_shape()[2].value

        """
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
    end_points['transform'] = transform
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

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        """

        #### linear model for debugging
        weights = tf.Variable(tf.zeros([max_npoint, num_channels]))
        biases = tf.Variable(0.0)
        y_pred = tf.tensordot(pointcloud, weights, axes=[[1,2], [0,1]]) + biases
        #### end of linear model

        return y_pred

    def get_loss(self, y_pred, y_true):
        MSELoss = tf.reduce_mean(tf.square(y_pred - y_true))
        return MSELoss


