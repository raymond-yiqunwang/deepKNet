#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import argparse
from model import KNetModel

parser = argparse.ArgumentParser(description='KNet parameters')
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 1e-3]')
FLAGS = parser.parse_args()


class Trainer(object):
    def __init__(self, batch_size, num_channels, learning_rate, max_epoch):
        self.KNet_model = KNetModel(batch_size, num_channels)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.train_data = "data/pointcloud_train.tfrecords"

    
    def train(self):
        with self.KNet_model.g_train.as_default():
            # input dataset
            dataset = self.load_tfrecords(self.train_data)
            dataset = dataset.repeat(1).batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, 10, 3])

            band_gap = tf.reshape(features["band_gap"], [1,1])

            # define loss
            MAELoss = self.KNet_model.train_graph(pointcloud, band_gap)
            tf.summary.scalar('MAELoss', MAELoss)

            # TODO learning rate decay
            learning_rate = self.learning_rate
            tf.summary.scalar('learning_rate', learning_rate)

            # define optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(MAELoss)

            merged = tf.summary.merge_all()

            # summary writer
            train_writer = tf.summary.FileWriter("./logs", self.KNet_model.g_train)
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch in range(self.max_epoch):
                     
                     _, loss_val, mgd = sess.run([optimizer, MAELoss, merged])
                     train_writer.add_summary(mgd)

                     if (epoch % 5 == 0):
                        print("Current epoch: {}".format(epoch))


    def load_tfrecords(self, record_files):
        raw_dataset = tf.data.TFRecordDataset(record_files)
    
        feature_description = {
            'pointcloud': tf.VarLenFeature(tf.float32),
            'my_id': tf.FixedLenFeature([], tf.int64),
            'band_gap': tf.FixedLenFeature([], tf.float32),
            'formation_energy_per_atom': tf.FixedLenFeature([], tf.float32),
            'nsites': tf.FixedLenFeature([], tf.int64),
        }
    
        def _parse_pointcloud_function(example_proto):
            return tf.parse_single_example(example_proto, feature_description)
    
        parsed_data = raw_dataset.map(_parse_pointcloud_function)
    
        return parsed_data


if __name__ == "__main__":

    # train
    trainer = Trainer(batch_size=FLAGS.batch_size, num_channels=FLAGS.num_channels,
                      learning_rate=FLAGS.learning_rate, max_epoch=FLAGS.max_epoch)
    trainer.train()

    # test
    # trainer.KNetModel


