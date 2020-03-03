import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import XRD_simulator.xrd_simulator as xrd
from multiprocessing import Pool
from pymatgen.core.structure import Structure


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
        _, features, recip_latt = xrd_simulator.get_pattern(structure=struct)
        """
          features: nrow = number of reciprocal k-points
          features: ncol = (hkl  , lorentz_factor, i_hkl , atomic_form_factor)
                            [1x3], scalar,       , scalar, [1x94]
        """
        # regroup features
        hkl = [ipoint[0] for ipoint in features]
        lorentz_factor = [ipoint[1] for ipoint in features]
        i_hkl = [ipoint[2] for ipoint in features]
        atomic_form_factor = [ipoint[3] for ipoint in features]

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
        ifeat.extend([hkl, lorentz_factor, i_hkl, atomic_form_factor])
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
    filename = "./data_raw/custom_MPdata.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(filename, sep=';', header=0, index_col=None)

    # specify output
    out_file = "./data_raw/compute_xrd.csv"
    # safeguard
    if os.path.exists(out_file):
        _ = input("Attention, the existing xrd data will be deleted and regenerated.. \
            \n>> Hit Enter to continue, Ctrl+c to terminate..")
        print("Started recomputing xrd..")
    header = [['material_id', 'band_gap', 'energy_per_atom', 'formation_energy_per_atom', \
              'recip_latt', 'hkl', 'lorentz_factor', 'i_hkl', 'atomic_form_factor']]
    df = pd.DataFrame(header)
    df.to_csv(out_file, sep=';', header=None, index=False, mode='w')
    
    # parameters
    n_slices = MP_data.shape[0] // 240 + 1 # number of batches to split
    wavelength = 'CuKa' # X-ray wavelength
    nworkers = 12

    # parallel processing
    MP_data_chunk = np.array_split(MP_data, n_slices)
    # 'serial parallel' processing
    for chunk in MP_data_chunk:
        # generate xrd point cloud representations
        xrd_data = parallel_computing(chunk, wavelength, nworkers)
        # write to file
        xrd_data.to_csv(out_file, sep=';', header=None, index=False, mode='a')


if __name__ == "__main__":
    main()


