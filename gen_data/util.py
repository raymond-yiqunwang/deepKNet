#! /home/raymondw/.conda/envs/pymatgen/bin/python
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
import pymatgen.analysis.diffraction.xrd_mod as xrd


def check_npoint_max(path, wavelength):
    # read data
    data = pd.read_csv(path, sep=';', header=0, index_col=None)

    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    npoint_max = 0
    for index, irow in data.iterrows():
        if (index % 500 == 1): 
            print("checkpoint {}".format(index))
            print("current max: {}".format(npoint_max))
            print("")
        struct = Structure.from_str(irow['cif'], fmt="cif")
        hkl = xrdcalc.get_pattern(struct)
        if (len(hkl) > npoint_max):
            npoint_max = len(hkl)
    return npoint_max

