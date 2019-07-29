#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import argparse
import sys
from model import KNetModel

parser = argparse.ArgumentParser(description='KNet parameters')
parser.add_argument('--num_channels', type=int, default=123)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--continue_training', type=bool, default=False)
FLAGS = parser.parse_args()

    
def load_tfrecords(record_files):
    raw_dataset = tf.data.TFRecordDataset(record_files)

    feature_description = {
        'pointcloud': tf.VarLenFeature(tf.float32),
        'band_gap': tf.FixedLenFeature([], tf.float32)
    }
    def _parse_pointcloud_function(example_proto):
        return tf.parse_single_example(example_proto, feature_description)

    parsed_data = raw_dataset.map(_parse_pointcloud_function)
    return parsed_data


class Trainer(object):
    def __init__(self, batch_size, num_channels, learning_rate, max_epoch, model_path=None, continue_training=False):
        self.KNet_model = KNetModel()
        self.num_channels = num_channels

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        
        self.continue_training = continue_training
        self.model_path = model_path
        self.train_data = "./data/pointcloud_train.tfrecords"
        self.valid_data = "./data/pointcloud_valid.tfrecords"
        
        self.train_ops = self._get_train_ops()
        self.valid_ops = self._get_valid_ops()

        # init tf sessions
        self.train_session = tf.Session(graph=self.KNet_model.g_train)
        sess = self.train_session
        sess.run(self.train_init)
        
#        self.valid_session = tf.Session(graph=self.KNet_model.g_valid)
        
    
    def _get_valid_ops(self):
        with self.KNet_model.g_valid.as_default():
            dataset = load_tfrecords(self.valid_data)
            dataset = dataset.batch(self.batch_size).repeat(1)
            iterator = dataset.make_initializable_iterator()
            self.valid_iterator = iterator
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, -1, self.num_channels])

            # TODO this is ugly but works for the moment
            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            # define loss
            loss = self.KNet_model.valid_graph(pointcloud, band_gap)
            tf.summary.scalar('loss', loss)

            # TODO learning rate decay, lr_scheduling
            learning_rate = self.learning_rate
            #tf.summary.scalar('learning_rate', learning_rate)

            # TODO batch normalization

            # tf.summary.histo to visualize weight distribution

            global_step = tf.Variable(0, name='global_step',trainable=False)

            # summary writer
            self.valid_writer = tf.summary.FileWriter("./logs/valid", self.KNet_model.g_valid)
            merged = tf.summary.merge_all()

            self.valid_init = tf.global_variables_initializer()
            self.valid_saver = tf.train.Saver()
            
            return global_step, loss, merged
    
    def valid(self):
        writer = self.valid_writer
        saver = self.valid_saver

        sess = self.valid_session
        sess.run(self.valid_iterator.initializer)

        saver.restore(sess, self.model_path)

        pdata = 0.0
        cnt = 0

        while True:
            try:
                step, loss, mgd = sess.run(self.train_ops)
                writer.add_summary(mgd, step)
                pdata += loss.sum() # TODO average?
                cnt += 1

            except tf.errors.OutOfRangeError as e:
                print("Average batch loss: {.3f}\n".format(pdata/cnt))
                break
            

    def _get_train_ops(self):
        with self.KNet_model.g_train.as_default():
            # input dataset
            dataset = load_tfrecords(self.train_data)
            dataset = dataset.batch(self.batch_size).repeat(1).shuffle(buffer_size=300)
            iterator = dataset.make_initializable_iterator()
            self.train_iterator = iterator
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, -1, self.num_channels])

            # TODO this is ugly but works for the moment
            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            # define loss
            loss = self.KNet_model.train_graph(pointcloud, band_gap)
            tf.summary.scalar('Huber loss', loss)

            # TODO learning rate decay, lr_scheduling
            learning_rate = self.learning_rate
            #tf.summary.scalar('learning_rate', learning_rate)

            # TODO batch normalization

            # tf.summary.histo to visualize weight distribution

            global_step = tf.Variable(0, name='global_step',trainable=False)

            # define optimizer
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # summary writer
            self.train_writer = tf.summary.FileWriter("./logs/train", self.KNet_model.g_train)
            merged = tf.summary.merge_all()

            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
            
            return global_step, optimizer, loss, merged
    
    def train(self):
        
        writer = self.train_writer
        saver = self.train_saver

        sess = self.train_session
        sess.run(self.train_iterator.initializer)

        while True:
            try:
                step, _, loss, mgd = sess.run(self.train_ops)
                writer.add_summary(mgd, step)

            except tf.errors.OutOfRangeError as e:
                print("")
                break
            
        print(">> Save model at: {}".format(saver.save(sess, self.model_path)))
        print("")


    def train_and_eval(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        print('###')

        for i in range(self.max_epoch):
            print("In epoch #{}, ".format(i+1), end="\n")
            self.train()
#            self.valid()

        self.train_session.close()
#        self.valid_session.close()


if __name__ == "__main__":

    trainer = Trainer(batch_size        = FLAGS.batch_size,
                      num_channels      = FLAGS.num_channels,
                      learning_rate     = FLAGS.learning_rate,
                      max_epoch       = FLAGS.max_epoch,
                      model_path        = "./save_model/model.ckpt",
                      continue_training = FLAGS.continue_training
                      )
    
    # train
    trainer.train_and_eval()
    

