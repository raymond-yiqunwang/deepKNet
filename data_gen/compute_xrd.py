import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import xrd_simulator.xrd_simulator as xrd
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def compute_xrd(root_dir, fnames, wavelength, sym_thresh):
    xrd_simulator = xrd.XRDSimulator(wavelength=wavelength)
    for filename in fnames:
        with open(os.path.join(root_dir, filename+".cif")) as f:
            cif_struct = Structure.from_str(f.read(), fmt="cif")
        # conventional cell
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        conventional_struct = sga.get_conventional_standard_structure()
        _, recip_latt, features = xrd_simulator.get_pattern(structure=conventional_struct)
        break


if __name__ == "__main__":
    # read all MPdata filenames
    root_dir = "./MPdata_all/"
    MPdata = pd.read_csv(root_dir+"MPdata_all.csv", sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    
    # parameters
    wavelength = 'CuKa' # CuKa by default
    sym_thresh = 0.1
#    nworkers = max(multiprocessing.cpu_count(), 1)
    nworkers = 1

    # initialize pool of workers
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, fnames, wavelength, sym_thresh) for fnames in file_split]
    pool.starmap(compute_xrd, args)
    pool.close()
    pool.join()


