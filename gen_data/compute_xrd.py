import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import XRD_simulator.xrd_simulator as xrd
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure

parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

""" properties
        "material_id", "icsd_ids",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "cif",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "density", "total_magnetization", "elasticity",
        "is_hubbard", "hubbards",
        "warnings", "tags",
"""

def compute_xrd(data_raw, wavelength):
    xrd_data_batch = []
    xrd_simulator = xrd.XRDSimulator(wavelength=wavelength)
    for idx, irow in data_raw.iterrows():
        # obtain xrd features
        struct = Structure.from_str(irow['cif'], fmt="cif")
        _, recip_latt, features = xrd_simulator.get_pattern(structure=struct)
        """
          features: nrow = number of reciprocal k-points
          features: ncol = (hkl, intensity_hkl)
        """
        # regroup features
        hkl = [ipoint[0] for ipoint in features]
        intensity_hkl = [ipoint[1] for ipoint in features]
#        aff_phase = [ipoint[2] for ipoint in features]

        # properties of interest
        material_id = irow['material_id']
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']

        # property list
        ifeat = [material_id, band_gap, energy_per_atom, formation_energy_per_atom] 
        # features for post-processing
        ifeat.append(recip_latt.tolist())
        # point-specific features
#        ifeat.extend([hkl, f_hkl, aff_phase])
        ifeat.extend([hkl, intensity_hkl])
        xrd_data_batch.append(ifeat)
    
    return pd.DataFrame(xrd_data_batch)


def parallel_computing(df_in, wavelength, nworkers=1):
    # initialize pool of workers
    pool = Pool(processes=nworkers)
    df_split = np.array_split(df_in, nworkers)
    args = [(data, wavelength) for data in df_split]
    df_out = pd.concat(pool.starmap(compute_xrd, args), axis=0)
    pool.close()
    pool.join()
    return df_out


def main():
    global args

    filename = "./data_raw/custom_MPdata.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(filename, sep=';', header=0, index_col=None)
    
    if args.debug:
        # random subsample in debug mode
        subsample_size = 1000
        MP_data = MP_data.sample(n=subsample_size, replace=False, random_state=1, axis=0)

    # specify output
    if not args.debug:
        out_file = "./data_raw/compute_xrd.csv"
    else:
        out_dir = "./data_raw/debug_data/"
        if not os.path.exists(out_dir):
            print("{} folder does not exist, making directory..".format(out_dir))
            os.mkdir(out_dir)
        out_file = out_dir + "debug_compute_xrd.csv"
    # output safeguard
    if os.path.exists(out_file):
        _ = input("Attention, the existing xrd data will be deleted and regenerated.. \
            \n>> Hit Enter to continue, Ctrl+c to terminate..")
    header = [['material_id', 'band_gap', 'energy_per_atom', 'formation_energy_per_atom', \
               'recip_latt', 'hkl', 'intensity_hkl']]
    df = pd.DataFrame(header)
    df.to_csv(out_file, sep=';', header=None, index=False, mode='w')
    
    # parameters
    wavelength = 'CuKa' # X-ray wavelength
    nworkers = max(multiprocessing.cpu_count()-4, 1)
    n_slices = MP_data.shape[0] // (20*nworkers) # number of batches to split into

    # parallel processing
    MP_data_chunk = np.array_split(MP_data, n_slices)
    print("start computing xrd..")
    # 'serial parallel' processing
    for idx, chunk in enumerate(MP_data_chunk):
        # generate xrd point cloud representations
        xrd_data = parallel_computing(chunk, wavelength, nworkers)
        # write to file
        xrd_data.to_csv(out_file, sep=';', header=None, index=False, mode='a')
        print('finished processing chunk {}/{}'.format(idx+1, n_slices)) 


if __name__ == "__main__":
    main()


