import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

f = h5py.File("data_test/dataset0.h5")
I_hkl = f['I_hkl'][:10]
band_gap = f['band_gap'][:10]

I_hkl = I_hkl.reshape(I_hkl.shape[0], -1)
band_gap = band_gap.reshape(-1, 1)
print(I_hkl.shape) # (10, -1)
print(band_gap.shape) # (10, 1)

x = tf.placeholder(tf.float32, [None, 744*4])
y_true = tf.placeholder(tf.float32, [None, 1])
weights = tf.Variable(tf.zeros(744*4))
biases = tf.Variable(0.0)

y_pred = tf.multiply(x, weights) + biases
loss = tf.reduce_mean(tf.square(y_pred - y_true))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)


session = tf.Session()
session.run(tf.global_variables_initializer())

#x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
feed_dict_train = {x: I_hkl, y_true: band_gap}
session.run(optimizer, feed_dict=feed_dict_train)
