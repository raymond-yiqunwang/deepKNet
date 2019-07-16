#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import numpy as np

###kinit_func = lambda: tf.initializers.glorot_normal()

class KNetModel(object):
    def __init__(self):
        self._init_graph()

    def _init_graph(self):
        self.g_train_ = tf.Graph()

    @property
    def g_train(self):
        return self.g_train_

    def placeholder_inputs(self, max_npoint, num_channels):
        x_train = tf.placeholder(tf.float32, [None, max_npoint, num_channels])
        y_true = tf.placeholder(tf.float32, [None, 1])
        return x_train, y_true

    def get_model(self, x_train, max_npoint, num_channels):
        # input  -- (BATCH_SIZE, MAX_NPOINT, 4)
        # output -- (BATCH_SIZE, 1)
        weights = tf.Variable(tf.zeros([max_npoint, num_channels]))
        biases = tf.Variable(0.0)
        y_pred = tf.tensordot(x_train, weights, axes=[[1,2], [0,1]]) + biases
        return y_pred

    def get_loss(self, y_pred, y_true):
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return loss


