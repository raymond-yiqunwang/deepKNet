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
    #mp_data = m.query(criteria={"volume": { "$lt" : 50 }, "band_structure": { "$ne" : None }, "warnings": []}, properties=properties)
    
    data_origin = []
    for entry in mp_data:
        plist = []
        for _, val in entry.items():
            plist.append(val)
        data_origin.append(plist)
    
    data_origin = pd.DataFrame(data_origin, index=None, columns=properties)
    return data_origin


def customize_data(data_origin, wavelength):
    
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
    for _, irow in data_custom.iterrows():
        for elem in irow['elements']:
            elem_dict2[Element(elem).Z] += 1
    for elem, count in elem_dict2.items():
        if (count < int(0.01*data_custom.shape[0])):
            rare_elements.append(elem)
            print("Element No. {} has a count of {}.".format(elem, count))

    # number of points
    npoint_list = []
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    for _, irow in data_custom.iterrows():
        struct = Structure.from_str(irow['cif'], fmt="cif")
        npoint_list.append(xrdcalc.get_npoint(struct))
    
    npoint_array = np.asarray(npoint_list)
    print(" npoint: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(np.mean(npoint_array), np.median(npoint_array), np.std(npoint_array), np.min(npoint_array), np.max(npoint_array)))
    """
    npoint: mean = 2590.45, median = 2319.00, standard deviation = 1623.81, min = 87.00, max = 7339.00
    """

    return data_custom


def write_tfRecord(data, save_dir, wavelength, npoint_cutoff):
    
    # split data into train, valid, and test
    rand_index = list(data.index)
    random.shuffle(rand_index)
    train_ratio = 0.97
    num_train = int(len(rand_index) * train_ratio)

    # init XRD calculator
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    
    # tfRecord writer
    rw = tfRecordWriter()

    # write train valid test records
    for mode in ['train', 'valid']:
        print("writing {} records..".format(mode))

        if (mode == 'train'): index = rand_index[:num_train]
        else: index = rand_index[num_train:]

        record_file = save_dir + 'pointcloud_{}.tfrecords'.format(mode)
        with tf.io.TFRecordWriter(record_file) as writer:
            checkpoint = 0
            for idx in index:
                if (checkpoint % 500 == 0): print(">> checkpoint {}".format(checkpoint))
                irow = data.loc[idx]

                # generate pointnet
                struct = Structure.from_str(irow['cif'], fmt="cif")
                #pointcloud = np.asarray(xrdcalc.get_intensity(struct)).flatten()
                pointcloud = xrdcalc.get_atomic_form_factor(struct)
                # tailoring and padding
                while (len(pointcloud) < npoint_cutoff):
                    pointcloud.extend(pointcloud)
                if (len(pointcloud) > npoint_cutoff):
                    pointcloud = pointcloud[:npoint_cutoff]
                assert (len(pointcloud) == npoint_cutoff)
                # output
                pointcloud = np.asarray(pointcloud).flatten()

                band_gap = irow['band_gap']

                rw.write_feature(writer, pointcloud, band_gap)
                checkpoint += 1
            
            writer.close()


if __name__ == "__main__":
    
    wavelength = "CuKa"
    
    # get data from MaterialsProject
    data_origin = fetch_materials_data()
    
    # customize data and print statistics
    data_custom = customize_data(data_origin, wavelength)
    
    # compute XRD pattern and write to tfRecord
    write_tfRecord(data_custom, save_dir="../data/", wavelength=wavelength, npoint_cutoff = 4000)


