#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
import h5py
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd
import util


def process_data(source_file, save_dir, wavelength="CuKa"):
    # read data
    data = pd.read_csv(source_file, sep=';', header=0, index_col=None)
    
    save_dir = save_dir + "X-" + str(wavelength) + "/"
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir, 0o755)

    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    chuck_size = 850
    npoint_max = util.check_npoint_max(source_file, wavelength)
    print("Max number of points: {}".format(npoint_max))
    start = 0
    end = chuck_size
    for iter in range(int(np.ceil(data.shape[0] / chuck_size))):
        print("processing range: {} - {}".format(start, end))
        h5f = h5py.File(save_dir+"dataset{}.h5".format(iter), 'w')
        idata = data[start:end]
        
        # create dataset for band gap
        h5f.create_dataset('band_gap', data=idata['band_gap'])
        
        # create dataset for structure factor
        hkl_data = []
        for index, irow in idata.iterrows():
            # compute structure factor F_{hkl}, data format -- [[bx, by, bz, I_{hkl}], ...] (N x 4 array)
            #   where N is the number of valid hkl points within the limiting sphere with radius lambda/2
            struct = Structure.from_str(irow['cif'], fmt="cif")
            hkl = xrdcalc.get_pattern(struct)
            # padding
            while (len(hkl) < npoint_max):
                hkl.append([0.]*4)
            hkl_data.append(hkl)
        hkl_data = np.array(hkl_data, dtype=np.float32)
        h5f.create_dataset('I_hkl', data=hkl_data)

        start += chuck_size
        end += chuck_size
        if (end > data.shape[0]): end = -1


if __name__ == "__main__":

    process_data(source_file="/data_deep/deepKNet_data/MPdata_bandstruct_no_warning.csv", 
                 save_dir="/data_deep/deepKNet_data/xrd_bandstruct_no_warning/", wavelength="CuKa")

#    process_data("./data/MPdata_bandgap.csv")
#    process_data("./data/MPdata_bandgap_no_warning.csv")


