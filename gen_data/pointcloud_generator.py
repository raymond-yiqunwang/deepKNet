#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd
import tensorflow as tf


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
        if (len(hkl) > max_npoint):
            max_npoint = len(hkl)
    return max_npoint


# wrapper functions for TFRecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(pointcloud_string, max_npoint, my_id, band_gap, formation_energy_per_atom, nsites):
        feature = {
            'pointcloud_raw': _bytes_feature(pointcloud_string),
            'max_npoint': _int64_feature(max_npoint),
            'my_id': _int64_feature(my_id),
            'band_gap': _float_feature(band_gap),
            'formation_energy_per_atom': _float_feature(formation_energy_per_atom),
            'nsites': _int64_feature(nsites),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_pointnet(source_file, save_dir, wavelength, max_npoint):
    # read data
    data_origin = pd.read_csv(source_file, sep=';', header=0, index_col=None)

    # split data and store in multiple files
    num_instance = data_origin.shape[0]
    instance_per_file = 8000
    num_files = int(np.ceil(num_instance / instance_per_file))
    
    # init XRD calculator
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    
    start = 0
    end = instance_per_file
    for iter in range(num_files):
        print("processing range: {} - {}".format(start, end))
        
        idata = data_origin[start:end]
        
        # TFRecord file
        record_file = save_dir + 'pointcloud_dataset{}.tfrecords'.format(iter)

        with tf.io.TFRecordWriter(record_file) as writer:
            for index, irow in idata.iterrows():
                struct = Structure.from_str(irow['cif'], fmt="cif")
                hkl = xrdcalc.get_pattern(struct)
                while(len(hkl) < max_npoint): hkl.append([0.]*4)
                hkl = [val for sublist in hkl for val in sublist]
                hkl = np.asarray(hkl, dtype=float)
                pointcloud_string = hkl.tobytes()
    
                my_id = irow['my_id']
                band_gap = irow['band_gap']
                formation_energy_per_atom = irow['formation_energy_per_atom']
                nsites = irow['nsites']
    
                tf_example = make_example(pointcloud_string, max_npoint, my_id, 
                                          band_gap, formation_energy_per_atom, nsites)
                writer.write(tf_example.SerializeToString())
    
        start += instance_per_file
        end += instance_per_file
        if (end > num_instance): end = num_instance
        
       


if __name__ == "__main__":
    
    input_data = "../data/MPdata.csv"
    wavelength = "CuKa"
    
    if (1):
        max_npoint = check_max_npoint(source_file=input_data, wavelength=wavelength)
        print("max npoint: {}".format(max_npoint))
    else:
        max_npoint = 482

    generate_pointnet(source_file=input_data, save_dir="../data/", \
                      wavelength=wavelength, max_npoint=max_npoint)

"""

def read_record():
    raw_data = tf.data.TFRecordDataset('test.tfrecords')
    feature_description = {
        'max_npoint': tf.FixedLenFeature([], tf.int64),
        'pointcloud_raw': tf.FixedLenFeature([], tf.string),
        'band_gap': tf.FixedLenFeature([], tf.float32),
    }
    
    def _parse_image_function(example_proto):
        return tf.parse_single_example(example_proto, feature_description)
    
    parsed_data = raw_data.map(_parse_image_function)

    iterator = parsed_data.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    for i in range(3):
        item = sess.run(next_element)
        pointcloud_raw = item['pointcloud_raw']
        pointcloud = np.frombuffer(pointcloud_raw, dtype=float)
        print(pointcloud.shape)

"""
