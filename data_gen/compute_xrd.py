import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import xrd_simulator.xrd_simulator as xrd
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./', metavar='DATA_DIR')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--wavelength', default='CuKa', metavar='X-ray wavelength')
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

def compute_xrd(raw_data, wavelength):
    xrd_data_batch = []
    xrd_simulator = xrd.XRDSimulator(wavelength=wavelength)
    for idx, irow in raw_data.iterrows():
        # obtain xrd features
        struct = Structure.from_str(irow['cif'], fmt="cif")
        _, recip_latt, features = xrd_simulator.get_pattern(structure=struct)

        # properties of interest
        material_id = irow['material_id']
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']
        MIT = float(band_gap > 0)

        # property list
        ifeat = [material_id, recip_latt.tolist(), features,
                 band_gap, energy_per_atom, formation_energy_per_atom, MIT]
        # append to dataset
        xrd_data_batch.append(ifeat)
    
    return pd.DataFrame(xrd_data_batch)


def parallel_computing(df_in, wavelength, nworkers=1):
    # initialize pool of workers
    pool = Pool(processes=nworkers)
    df_split = np.array_split(df_in, nworkers)
    pargs = [(data, wavelength) for data in df_split]
    df_out = pd.concat(pool.starmap(compute_xrd, pargs), axis=0)
    pool.close()
    pool.join()
    return df_out


def main():
    global args

    filename = os.path.join(args.root, "raw_data/custom_MPdata.csv")
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(filename, sep=';', header=0, index_col=None)
    
    if args.debug:
        # random subsample in debug mode
        subsample_size = 800
        MP_data = MP_data.sample(n=subsample_size, replace=False, random_state=1, axis=0)

    # specify output
    if not args.debug:
        out_file = os.path.join(args.root, "raw_data/compute_xrd_"+args.wavelength+".csv")
    else:
        out_dir = os.path.join(args.root, "raw_data/debug_data/")
        if not os.path.exists(out_dir):
            print("{} folder does not exist, making directory..".format(out_dir))
            os.mkdir(out_dir)
        out_file = os.path.join(out_dir, "debug_compute_xrd_"+args.wavelength+".csv")
    
    # output safeguard
    if os.path.exists(out_file):
        print("Attention, the existing xrd data will be deleted and regenerated..")
    header = [['material_id', 'recip_latt', 'features',
               'band_gap', 'energy_per_atom', 'formation_energy_per_atom', 'MIT']]
    df = pd.DataFrame(header)
    df.to_csv(out_file, sep=';', header=None, index=False, mode='w')
    
    # parameters
    nworkers = max(multiprocessing.cpu_count(), 1)
    n_slices = MP_data.shape[0] // (20*nworkers) # number of batches to split into

    # parallel processing
    MP_data_chunk = np.array_split(MP_data, n_slices)
    print("start computing xrd..")
    for idx, chunk in enumerate(MP_data_chunk):
        # generate xrd point cloud representations
        xrd_data = parallel_computing(chunk, args.wavelength, nworkers)
        # write to file
        xrd_data.to_csv(out_file, sep=';', header=None, index=False, mode='a')
        print('finished processing chunk {}/{}'.format(idx+1, n_slices)) 


if __name__ == "__main__":
    main()


