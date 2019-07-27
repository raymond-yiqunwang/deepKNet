#! /home/raymondw/.conda/envs/tf-cpu/bin/python
from pymatgen import MPRester
from tfrecord_generator import tfRecordWriter
import pandas as pd
import os


def fetch_materials_data(data_path):
    # properties of interest
    prop_list = [ 
        "material_id", "pretty_formula", "band_gap", "band_structure",
        "elements",  "nelements", "nsites", "spacegroup", "cif", "volume",
        "energy", "energy_per_atom", "e_above_hull", "formation_energy_per_atom",
        "density", "elasticity", "piezo",  "diel", "total_magnetization"
    ]
    
    # fetch data
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query data with computed band structures and no warning sign
    mp_data = m.query(criteria={"band_structure": { "$ne" : None }, "warnings": []}, properties=prop_list)
    
    index = 0 #my_id
    data = []
    for entry in mp_data:
        plist = [index]
        for _, val in entry.items():
            plist.append(val)
        data.append(plist)
        index += 1
    
    prop_list.insert(0, "my_id")
    data = pd.DataFrame(data, index=None, columns=None)
    data.to_csv(data_path, sep=';', columns=None, header=prop_list, index=None)


def show_statistics(data_path):
    data_origin = pd.read_csv(data_path, sep=';', header=0, index_col=None)
    band_gap = data_origin['band_gap']
    print(" band gap: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(band_gap.mean(), band_gap.median(), band_gap.std(), band_gap.min(), band_gap.max()))
    nsites = data_origin['nsites']
    print(" nsites: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(nsites.mean(), nsites.median(), nsites.std(), nsites.min(), nsites.max()))
    volume = data_origin['volume']
    print(" volume: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(volume.mean(), volume.median(), volume.std(), volume.min(), volume.max()))


if __name__ == "__main__":
    data_path = "../data/MPdata_all.csv"

    # get all data available from MaterialsProject
    if (0): 
        fetch_materials_data(data_path)
    
    # compute mean, median, var, etc.
    if (0): 
        show_statistics(data_path)
        """ current output
        band gap: mean = 1.41, median = 0.72, standard deviation = 1.67, min = 0.00, max = 9.33
        nsites: mean = 26.55, median = 20.00, standard deviation = 24.37, min = 1.00, max = 200.00
        volume: mean = 423.60, median = 297.47, standard deviation = 425.53, min = 7.81, max = 5718.62
        """
    
    # customize data
    data_all = pd.read_csv(data_path, sep=';', header=0, index_col=None)
    data = data_all[['my_id', 'nsites', 'volume', 'band_gap', 'cif', 'formation_energy_per_atom']]
    # nsites filter
    data = data.loc[data['nsites'] < 50]
    # volume filter
    data = data.loc[data['volume'] < 1000.]

    # compute XRD pattern and write to tfRecord
    save_dir = "../data/"
    wavelength = "CuKa"
    rw = tfRecordWriter(data=data, save_dir=save_dir, wavelength=wavelength)
    rw.write_pointcloud()


