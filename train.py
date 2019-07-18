#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import argparse
from model import KNetModel

parser = argparse.ArgumentParser(description='KNet parameters')
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--train_epoch', type=int, default=2, help='Epoch to run [default: 2]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 1e-3]')
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
        self.KNet_model = KNetModel(batch_size, num_channels)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch
        self.model_path = model_path
        self.continue_training = continue_training
        self.train_data = "data/pointcloud_train.tfrecords"
        self.val_data = "data/pointcloud_val.tfrecords"

    
    def train(self):
        with self.KNet_model.g_train.as_default():
            # input dataset
            dataset = load_tfrecords(self.train_data)
            dataset = dataset.batch(self.batch_size).repeat(self.train_epoch).shuffle(buffer_size=2000)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            # TODO will take care of the reshape problem later
            pointcloud = tf.reshape(pointcloud, [self.batch_size, 800, 3])

            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            # define loss
            MAELoss = self.KNet_model.train_graph(pointcloud, band_gap)
            tf.summary.scalar('MAELoss', MAELoss)

            # TODO learning rate decay
            learning_rate = self.learning_rate
            tf.summary.scalar('learning_rate', learning_rate)

            global_step = tf.Variable(0, name='global_step',trainable=False)

            # define optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(MAELoss, global_step=global_step)

            merged = tf.summary.merge_all()

            # summary writer
            train_writer = tf.summary.FileWriter("./logs_train", self.KNet_model.g_train)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if (self.continue_training and self.model_path):
                    print("Loading model at:", self.model_path)
                    saver.restore(sess, self.model_path)

                while True:
                    try:
                        step, _, loss_val, mgd = sess.run([global_step, optimizer, MAELoss, merged])
                        train_writer.add_summary(mgd, step)
                        if step % 100 == 0:
                            print(">> Current step: {}".format(step))
                            if step % 1000 == 0:
                                print(">> Save model at: {}".format(saver.save(sess, self.model_path)))

                    except tf.errors.OutOfRangeError:
                        print("Fininshed training...")
                        break


    def validate(self):
        with self.KNet_model.g_val.as_default():
            # input dataset
            dataset = load_tfrecords(self.val_data)
            dataset = dataset.batch(self.batch_size).repeat(1).shuffile(buffer_size=2000)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, 10, 3])

            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            # define loss
            MAELoss = self.KNet_model.val_graph(pointcloud, band_gap)
            tf.summary.scalar('MAELoss', MAELoss)
            
            merged = tf.summary.merge_all()

            # summary writer
            val_writer = tf.summary.FileWriter("./logs_val", self.KNet_model.g_val)
            
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if self.model_path:
                    print("Loading model at:", self.model_path)
                    saver.restore(sess, self.model_path)

                while True:
                    try:
                        mgd = sess.run(merged)
                        val_writer.add_summary(mgd)
                    except tf.errors.OutOfRangeError:
                        print("Fininshed validating...")
                        break


if __name__ == "__main__":

    trainer = Trainer(batch_size=FLAGS.batch_size, num_channels=FLAGS.num_channels,
                      learning_rate=FLAGS.learning_rate, train_epoch=FLAGS.train_epoch,
                      model_path="save_model/model.ckpt", continue_training=FLAGS.continue_training)
    
    # train
    trainer.train()
    
    # validate
#    trainer.validate()

    # test


