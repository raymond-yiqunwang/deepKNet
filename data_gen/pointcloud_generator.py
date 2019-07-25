#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
from pymatgen.core.structure import Structure
import xrd_calculator as xrd
import tensorflow as tf
from collections import Iterable
import random


class RecordWriter(object):
    def __init__(self, input_data, save_dir, wavelength):
        self.input_data = input_data
        self.save_dir = save_dir
        self.wavelength = wavelength

    # wrapper functions for TFRecord
    def _float_feature(self, value):
        if not isinstance(value, Iterable):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, np.ndarray):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
        else:
            print(value)
    
    def _int64_feature(self, value):
        if not isinstance(value, Iterable):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))
        else:
            print(value)

    def write_feature(self, writer, pointcloud, my_id, band_gap, formation_energy_per_atom, nsites):
            feature = {
                'pointcloud': self._float_feature(pointcloud),
                'my_id': self._int64_feature(my_id),
                'band_gap': self._float_feature(band_gap),
                'formation_energy_per_atom': self._float_feature(formation_energy_per_atom),
                'nsites': self._int64_feature(nsites),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    def write_pointnet(self):
        # read data
        data_origin = pd.read_csv(self.input_data, sep=';', header=0, index_col=None)
    
        # split data into train, val, and test
        # TODO currently use all for training, update ratio later
        num_instance = data_origin.shape[0]
        rand_index = list(np.arange(num_instance))
        random.shuffle(rand_index)
        train_ratio = 0.9
        num_train = int(num_instance * train_ratio)
        val_ratio = 0.05
        num_val = int(num_instance * val_ratio)

        # init XRD calculator
        xrdcalc = xrd.XRDCalculator(wavelength=self.wavelength)
        
        # write train val test records
        for mode in ['train', 'val', 'test']:
            print("writing {} records..".format(mode))

            if (mode == 'train'): index = rand_index[:num_train]
            elif (mode == 'val'): index = rand_index[num_train: num_train+num_val]
            else: index = rand_index[num_train+num_val:]

            record_file = self.save_dir + 'pointcloud_{}.tfrecords'.format(mode)
            with tf.io.TFRecordWriter(record_file) as writer:
                cnt = 0
                for idx in index:
                    if (cnt % 1000 == 0): print(">> checkpoint {}".format(cnt))
                    irow = data_origin.iloc[idx]
                    struct = Structure.from_str(irow['cif'], fmt="cif")
                    pointcloud = xrdcalc.get_pattern(struct)
                    pointcloud = np.asarray(pointcloud).flatten()
        
                    my_id = irow['my_id']
                    band_gap = irow['band_gap']
                    formation_energy_per_atom = irow['formation_energy_per_atom']
                    nsites = irow['nsites']

                    self.write_feature(writer, pointcloud, my_id, band_gap,
                                        formation_energy_per_atom, nsites)
                    cnt += 1

                writer.close()


if __name__ == "__main__":
    
    input_data = "../data/MPdata.csv"
    save_dir = "../data/"
    wavelength = "CuKa"

    rw = RecordWriter(input_data=input_data, save_dir=save_dir, wavelength=wavelength)
    rw.write_pointnet()


