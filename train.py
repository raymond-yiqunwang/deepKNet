#! /home/raymondw/.conda/envs/tf-cpu/bin/python

import tensorflow as tf
import argparse
import sys
from model import KNetModel

DEBUG = True

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
        # initialize model
        self.KNet_model = KNetModel()
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.continue_training = continue_training
        
        # data and saving directories
        self.train_data = "./data/pointcloud_train.tfrecords"
        self.valid_data = "./data/pointcloud_valid.tfrecords"
        if (DEBUG):
            self.train_data = "./data/pointcloud_debug.tfrecords"
            self.valid_data = "./data/pointcloud_debug.tfrecords"
        self.model_path = model_path
        
        # init tf sessions and get train valid ops
        self.train_session = tf.Session(graph=self.KNet_model.g_train)
        self.train_ops = self._get_train_ops()
        
        self.valid_session = tf.Session(graph=self.KNet_model.g_valid)
        self.valid_ops = self._get_valid_ops()
        
    
    def _get_valid_ops(self):
        with self.KNet_model.g_valid.as_default():
            dataset = load_tfrecords(self.valid_data)
            dataset = dataset.batch(self.batch_size).repeat(1)
            iterator = dataset.make_initializable_iterator()
            self.valid_iterator = iterator
            features = iterator.get_next()

            pointcloud = tf.sparse_tensor_to_dense(features["pointcloud"])
            pointcloud = tf.reshape(pointcloud, [self.batch_size, -1, self.num_channels])

            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            loss = self.KNet_model.valid_graph(pointcloud, band_gap)
            tf.summary.scalar('loss', loss)

            # TODO batch normalization

            global_step = tf.Variable(0, name='global_step',trainable=False)

            merged = tf.summary.merge_all()
            self.valid_init = tf.global_variables_initializer()
            self.valid_saver = tf.train.Saver()
            self.valid_writer = tf.summary.FileWriter("./logs/valid", self.KNet_model.g_valid)
            
            return global_step, loss, merged
    
    def valid(self):
        sess = self.valid_session
        sess.run(self.valid_init)
        sess.run(self.valid_iterator.initializer)

        writer = self.valid_writer
        saver = self.valid_saver
        saver.restore(sess, self.model_path)

        pdata = 0.0
        cnt = 0
        while True:
            try:
                step, loss, mgd = sess.run(self.valid_ops)
                writer.add_summary(mgd, step)
                pdata += loss 
                cnt += 1

            except tf.errors.OutOfRangeError as e:
                print(">> Average batch loss: {:.3f}\n".format(pdata/cnt))
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

            band_gap = tf.reshape(features["band_gap"], [self.batch_size, 1])

            loss = self.KNet_model.train_graph(pointcloud, band_gap)
            tf.summary.scalar('Huber loss', loss)

            # TODO learning rate decay, lr_scheduling
            learning_rate = self.learning_rate
            #tf.summary.scalar('learning_rate', learning_rate)

            # TODO batch normalization

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            global_step = tf.Variable(0, name='global_step',trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=1.0)
                optim = optimizer.minimize(loss, global_step=global_step)

            merged = tf.summary.merge_all()
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
            self.train_writer = tf.summary.FileWriter("./logs/train", self.KNet_model.g_train)
            
            return global_step, optim, loss, merged
    
    def train(self):
        
        writer = self.train_writer
        saver = self.train_saver

        sess = self.train_session
        sess.run(self.train_init)
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
        print('\n###\n')

        for i in range(self.max_epoch):
            print("In epoch #{}, ".format(i+1), end="\n")
            self.train()
            self.valid()

        self.train_session.close()
        self.valid_session.close()


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
    

