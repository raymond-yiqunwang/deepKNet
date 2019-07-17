#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd
import tensorflow as tf


"""
def check_max_npoint(source_file, wavelength, num_channels):
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
        if (len(hkl) > num_channels*max_npoint):
            max_npoint = int(len(hkl) / num_channels)
    return max_npoint
"""


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


def make_example(pointcloud_string, num_channels, my_id, band_gap, formation_energy_per_atom, nsites):
        feature = {
            'pointcloud_raw': _bytes_feature(pointcloud_string),
            'num_channels': _int64_feature(num_channels),
            'my_id': _int64_feature(my_id),
            'band_gap': _float_feature(band_gap),
            'formation_energy_per_atom': _float_feature(formation_energy_per_atom),
            'nsites': _int64_feature(nsites),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_pointnet(source_file, save_dir, wavelength, num_channels):
    # read data
    data_origin = pd.read_csv(source_file, sep=';', header=0, index_col=None)

    # split data and store in multiple files
    num_instance = data_origin.shape[0]
    instance_per_file = 4000
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
                hkl = np.asarray(hkl, dtype=float)
                pointcloud_string = hkl.tobytes()
    
                my_id = irow['my_id']
                band_gap = irow['band_gap']
                formation_energy_per_atom = irow['formation_energy_per_atom']
                nsites = irow['nsites']
    
                tf_example = make_example(pointcloud_string, num_channels, my_id, 
                                          band_gap, formation_energy_per_atom, nsites)
                writer.write(tf_example.SerializeToString())
    
        start += instance_per_file
        end += instance_per_file
        if (end > num_instance): end = num_instance
        
       


if __name__ == "__main__":
    
    input_data = "../data/MPdata.csv"
    wavelength = "CuKa"
    num_channels = 4
    
    """
    if (1):
        max_npoint = check_max_npoint(source_file=input_data, wavelength=wavelength, num_channels=num_channels)
        print("max npoint: {}".format(max_npoint))
    else:
        max_npoint = 940
    """
    generate_pointnet(source_file=input_data, save_dir="../data/", \
                      wavelength=wavelength, num_channels=num_channels)

