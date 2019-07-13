#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
import os
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd


def process_data(source_file, save_dir, wavelength="CuKa"):
    # read data
    data = pd.read_csv(source_file, sep=';', header=0, index_col=None)
    
    out_dir = save_dir + wavelength + "/"
    if not (os.path.isdir(out_dir)):
        os.mkdir(out_dir, 0o755)

    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    for index, irow in data.iterrows():
        # print checkpoint
        if (index%5000 == 0): print("check point -- index: "+str(index))
        
        # unique identifier
        material_id = irow['material_id']
        
        # compute structure factor F_{hkl}, data format -- [[bx, by, bz, I_{hkl}], ...] (N x 4 array)
        #   where N is the number of valid hkl points within the limiting sphere with radius lambda/2
        struct = Structure.from_str(irow['cif'], fmt="cif")
        hkl_data = xrdcalc.get_pattern(struct)
        hkl_data = pd.DataFrame(data=hkl_data)
        hkl_data.to_csv(out_dir+material_id+"_attrib.csv", sep=';', index=None, header=['bx', 'by', 'bz', 'I_hkl'])

        # save properties
        prop_list = [ "material_id", "pretty_formula", "band_gap", "elements",  "nelements", "nsites",
            "spacegroup", "volume", "energy", "energy_per_atom", "e_above_hull", "density",
            "formation_energy_per_atom", "elasticity", "piezo",  "diel", "total_magnetization" ]
        properties = irow[prop_list]
        properties.to_csv(out_dir+material_id+"_prop.csv", sep=';', index=prop_list, header=False)
    


if __name__ == "__main__":

    process_data(source_file="/data_deep/deepKNet_data/MPdata_bandstruct_no_warning.csv", 
                 save_dir="/data_deep/deepKNet_data/xrd_bandstruct_no_warning/", wavelength="CuKa")

#    process_data("./data/MPdata_bandgap.csv")
#    process_data("./data/MPdata_bandgap_no_warning.csv")


