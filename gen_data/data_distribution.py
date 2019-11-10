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
        "band_gap"
    ]
    
    # MaterialsProject API settings
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query data with computed band structures and no warning sign
#    mp_data = m.query(criteria={"band_structure": { "$ne" : None }, "warnings": []}, properties=properties)
    mp_data = m.query(criteria={"volume": { "$lt" : 800 }, "band_structure": { "$ne" : None }, "warnings": []}, properties=properties)
    
    data_origin = []
    for entry in mp_data:
        plist = []
        for _, val in entry.items():
            plist.append(val)
        data_origin.append(plist)
    
    data_origin = pd.DataFrame(data_origin, index=None, columns=properties)
    print(data_origin.mean(axis=0))
    print(data_origin.median(axis=0))
    print(data_origin.std(axis=0))
    return data_origin



if __name__ == "__main__":
    
    # get data from MaterialsProject
    data_origin = fetch_materials_data()
    
