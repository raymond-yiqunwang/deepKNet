#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
import h5py
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd


def check_max_npoint(path, wavelength):
    # read data
    data = pd.read_csv(path, sep=';', header=0, index_col=None)
            
    print("checking maximum number of points...")

    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    max_npoint = 0
    for index, irow in data.iterrows():
        if (index % 500 == 1): 
            print(">> checkpoint {}, current max: {}".format(index, max_npoint))
            print("")
        struct = Structure.from_str(irow['cif'], fmt="cif")
        hkl = xrdcalc.get_pattern(struct)
        if (len(hkl) > max_npoint):
            max_npoint = len(hkl)
    return max_npoint


def generate_pointnet(source_file, save_dir, wavelength="CuKa"):
    # read data
    data_origin = pd.read_csv(source_file, sep=';', header=0, index_col=None)
    # split data and store in multiple files
    num_instance = data_origin.shape[0]
    instance_per_file = 900
    num_files = int(np.ceil(num_instance / instance_per_file))
    
    # init XRD calculator
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    
    # determine maximum number of points in k-space
    max_npoint = check_max_npoint(source_file, wavelength)
    print("\nMax number of points: {}".format(max_npoint))
    
    start = 0
    end = instance_per_file
    for iter in range(num_files):
        print("processing range: {} - {}".format(start, end))
        
        idata = data_origin[start:end]
        
        # output file
        h5f = h5py.File(save_dir+"pointcloud_set{}.h5".format(iter), 'w')
        
        # create dataset for my_id
        h5f.create_dataset('my_id', data=idata['my_id'])
        
        # compute XRD diffraction intensity
        hkl_data = []
        for index, irow in idata.iterrows():
            # compute structure factor F_{hkl}, data format -- [[bx, by, bz, I_{hkl}], ...] (N x 4 array)
            #   where N is the number of valid hkl points within the limiting sphere with radius lambda/2
            struct = Structure.from_str(irow['cif'], fmt="cif")
            hkl = xrdcalc.get_pattern(struct)
            # zero padding (points are permutation invariant)
            while (len(hkl) < max_npoint):
                hkl.append([0.]*4)
            hkl_data.append(hkl)
        hkl_data = np.array(hkl_data, dtype=np.float32)
        h5f.create_dataset('I_hkl', data=hkl_data)
        
        # create dataset for band gap
        h5f.create_dataset('band_gap', data=idata['band_gap'])

        # create dataset for formation_energy_per_atom
        h5f.create_dataset('formation_energy_per_atom', data=idata['formation_energy_per_atom'])
        
        # create dataset for nsites
        h5f.create_dataset('nsites', data=idata['nsites'])

        start += instance_per_file
        end += instance_per_file
        if (end > num_instance): end = -1


if __name__ == "__main__":

    generate_pointnet(source_file="../data/MPdata.csv", 
                 save_dir="../data/", wavelength="CuKa")

