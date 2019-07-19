#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd
import tensorflow as tf
from collections import Iterable
import random

# XRD wavelengths in angstroms
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}

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

    def pointcloud_normalizer(self, pointcloud_raw):
        pointcloud = np.asarray(pointcloud_raw, dtype=float)
        radius = 2. / WAVELENGTHS[self.wavelength]
        pointcloud[:,:3] /= radius
        pointcloud[:,-1] = np.where(pointcloud[:,-1]>1e-6, np.log(pointcloud[:,-1]), np.log(1e-6))
        return pointcloud

    def write_pointnet(self):
        # read data
        data_origin = pd.read_csv(self.input_data, sep=';', header=0, index_col=None)
    
        # split data into train, val, and test
        num_instance = data_origin.shape[0]
        rand_index = list(np.arange(num_instance))
        random.shuffle(rand_index)
        train_ratio = 0.7
        num_train = int(num_instance * train_ratio)
        val_ratio = 0.15
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
                    if (cnt % 5000 == 0): print(">> checkpoint {}".format(cnt))
                    irow = data_origin.iloc[idx]
                    struct = Structure.from_str(irow['cif'], fmt="cif")
                    pointcloud = xrdcalc.get_pattern(struct)
                    pointcloud = self.pointcloud_normalizer(pointcloud)
                    pointcloud = pointcloud.flatten()
        
                    my_id = irow['my_id']
                    band_gap = irow['band_gap']
                    formation_energy_per_atom = irow['formation_energy_per_atom']
                    nsites = irow['nsites']

                    self.write_feature(writer, pointcloud, my_id, band_gap,
                                        formation_energy_per_atom, nsites)
                    cnt += 1

                writer.close()


def check_max_npoint(source_file, wavelength):
    # read data
    data = pd.read_csv(source_file, sep=';', header=0, index_col=None)
            
    print("checking maximum number of points...")

    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    max_npoint = 0
    for index, irow in data.iterrows():
        if (index % 1000 == 1): 
            print(">> checkpoint {}, current max: {}".format(index, max_npoint))
            print("")
        struct = Structure.from_str(irow['cif'], fmt="cif")
        hkl = xrdcalc.get_pattern(struct)
        if (len(hkl) > 3*max_npoint):
            max_npoint = int(len(hkl) / 3)
    return max_npoint


if __name__ == "__main__":
    
    input_data = "../data/MPdata.csv"
    save_dir = "../data/"
    wavelength = "CuKa"

#    print("max npoint : {}".format(check_max_npoint(input_data, wavelength)))
    
    rw = RecordWriter(input_data=input_data, save_dir=save_dir, wavelength=wavelength)
    rw.write_pointnet()
