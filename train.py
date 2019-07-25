#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import argparse
import sys
from model import KNetModel

parser = argparse.ArgumentParser(description='KNet parameters')
parser.add_argument('--num_channels', type=int, default=4)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--continue_training', type=bool, default=False)
FLAGS = parser.parse_args()

    
def load_tfrecords(record_files):
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


class Trainer(object):
    def __init__(self, batch_size, num_channels, learning_rate, train_epoch, model_path=None, continue_training=False):
        self.KNet_model = KNetModel()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch
        self.model_path = model_path
        self.continue_training = continue_training
        self.train_data = "./data/pointcloud_train.tfrecords"
        self.val_data = "./data/pointcloud_val.tfrecords"

    
    def train(self):
        with self.KNet_model.g_train.as_default():
            # input dataset
            dataset = load_tfrecords(self.train_data)
            dataset = dataset.batch(self.batch_size).repeat(self.train_epoch).shuffle(buffer_size=500)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, -1, self.num_channels])

            # TODO this is ugly but works for the moment
            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            # define loss
            MSELoss = self.KNet_model.train_graph(pointcloud, band_gap)
            tf.summary.scalar('MSELoss', MSELoss)

            # TODO learning rate decay, lr_scheduling
            learning_rate = self.learning_rate
            #tf.summary.scalar('learning_rate', learning_rate)

            # TODO batch normalization

            # tf.summary.histo to visualize weight distribution

            global_step = tf.Variable(0, name='global_step',trainable=False)

            # define optimizer
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(MSELoss, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(MSELoss, global_step=global_step)

            # summary writer
            train_writer = tf.summary.FileWriter("./logs_train", self.KNet_model.g_train)
            merged = tf.summary.merge_all()

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if (self.continue_training and self.model_path):
                    print("Loading model at:", self.model_path)
                    saver.restore(sess, self.model_path)

                while True:
                    try:
                        step, _, loss_val, mgd = sess.run([global_step, optimizer, MSELoss, merged])
                        train_writer.add_summary(mgd, step)
                        if step % 5000 == 0:
                            print(">> Current step: {}".format(step))
                            print(">> Save model at: {}".format(saver.save(sess, self.model_path)))
                            print("")

                    except tf.errors.OutOfRangeError:
                        print("Fininshed training...")
                        break


if __name__ == "__main__":

    trainer = Trainer(batch_size=FLAGS.batch_size, num_channels=FLAGS.num_channels,
                      learning_rate=FLAGS.learning_rate, train_epoch=FLAGS.train_epoch,
                      model_path="./save_model/model.ckpt", continue_training=FLAGS.continue_training)
    
    # train
    trainer.train()
    

