#! /home/raymondw/.conda/envs/tf-cpu/bin/python
from pymatgen import MPRester
from tfrecord_writer import tfRecordWriter
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from collections import defaultdict
import tensorflow as tf
import xrd_calculator as xrd
import pandas as pd
import os
import numpy as np
import random


def fetch_materials_data():
    # properties of interest
    properties = [ 
        "material_id", "pretty_formula", "band_gap",
        "elements", "nsites", "cif", "volume"
    ]
    
    # MaterialsProject API settings
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query data with computed band structures and no warning sign
    mp_data = m.query(criteria={"band_structure": { "$ne" : None }, "warnings": []}, properties=properties)
    
    data_origin = []
    for entry in mp_data:
        plist = []
        for _, val in entry.items():
            plist.append(val)
        data_origin.append(plist)
    
    data_origin = pd.DataFrame(data_origin, index=None, columns=properties)
    return data_origin


def customize_data(data_origin):
    
    # show statistics of original data
    print("Data distribution before customization:")
    print("Number of instances: {}".format(data_origin.shape))
    band_gap = data_origin['band_gap']
    print(" band gap: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(band_gap.mean(), band_gap.median(), band_gap.std(), band_gap.min(), band_gap.max()))
    nsites = data_origin['nsites']
    print(" nsites: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(nsites.mean(), nsites.median(), nsites.std(), nsites.min(), nsites.max()))
    volume = data_origin['volume']
    print(" volume: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(volume.mean(), volume.median(), volume.std(), volume.min(), volume.max()))
    # rare elements
    rare_elements = []
    elem_dict = defaultdict(int)
    for elements in data_origin['elements']:
        for elem in elements:
            elem_dict[Element(elem).Z] += 1
    for elem, count in elem_dict.items():
        if (count < int(0.01*data_origin.shape[0])):
            rare_elements.append(elem)
            print("Element No. {} has a count of {}.".format(elem, count))

    # customize data
    data_custom = data_origin[['band_gap', 'nsites', 'volume', 'cif', 'elements']].copy()
    drop_instance = []
    for idx, irow in data_custom.iterrows():
        if (True in [ (Element(elem).Z in rare_elements) for elem in irow['elements']]) \
            or (irow['nsites'] > 50) or (irow['volume'] > 800.):
            drop_instance.append(idx)
    data_custom = data_custom.drop(drop_instance)

    # show statistics of customized data
    print("Data distribution after customizaion:")
    print("Number of instances: {}".format(data_custom.shape))
    band_gap = data_custom['band_gap']
    print(" band gap: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(band_gap.mean(), band_gap.median(), band_gap.std(), band_gap.min(), band_gap.max()))
    nsites = data_custom['nsites']
    print(" nsites: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(nsites.mean(), nsites.median(), nsites.std(), nsites.min(), nsites.max()))
    volume = data_custom['volume']
    print(" volume: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(volume.mean(), volume.median(), volume.std(), volume.min(), volume.max()))
    # rare elements
    rare_elements = []
    elem_dict2 = defaultdict(int)
    for idx, irow in data_custom.iterrows():
        for elem in irow['elements']:
            elem_dict2[Element(elem).Z] += 1
    for elem, count in elem_dict2.items():
        if (count < int(0.01*data_custom.shape[0])):
            rare_elements.append(elem)
            print("Element No. {} has a count of {}.".format(elem, count))

    return data_custom


def write_tfRecord(data, save_dir, wavelength):
    
    # split data into train, val, and test
    num_instance = data.shape[0]
    rand_index = list(np.arange(num_instance))
    random.shuffle(rand_index)
    train_ratio = 0.7
    num_train = int(num_instance * train_ratio)
    val_ratio = 0.15
    num_val = int(num_instance * val_ratio)

    # init XRD calculator
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    
    # tfRecord writer
    rw = tfRecordWriter()

    # write train val test records
    for mode in ['train', 'val', 'test']:
        print("writing {} records..".format(mode))

        if (mode == 'train'): index = rand_index[:num_train]
        elif (mode == 'val'): index = rand_index[num_train: num_train+num_val]
        else: index = rand_index[num_train+num_val:]

        record_file = save_dir + 'pointcloud_{}.tfrecords'.format(mode)
        with tf.io.TFRecordWriter(record_file) as writer:
            checkpoint = 0
            for idx in index:
                if (checkpoint % 1000 == 0): print(">> checkpoint {}".format(checkpoint))
                irow = data.iloc[idx]

                # generate pointnet
                struct = Structure.from_str(irow['cif'], fmt="cif")
                #pointcloud = np.asarray(xrdcalc.get_intensity(struct)).flatten()
                pointcloud = np.asarray(xrdcalc.get_atomic_form_factor(struct)).flatten()

                band_gap = irow['band_gap']

                rw.write_feature(writer, pointcloud, band_gap)
                checkpoint += 1
            
            writer.close()


if __name__ == "__main__":
    # get data from MaterialsProject
    data_origin = fetch_materials_data()
    
    # modify data
    data_custom = customize_data(data_origin)

    # compute XRD pattern and write to tfRecord
    write_tfRecord(data_custom, save_dir="../data/", wavelength="CuKa")


