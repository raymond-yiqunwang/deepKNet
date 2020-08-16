import os
import sys
import ast
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import xrd_simulator.xrd_simulator as xrd
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

""" properties
        "material_id", "icsd_ids",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "cif",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "density", "total_magnetization", "elasticity",
        "is_hubbard", "hubbards",
        "warnings", "tags", "crystal_system"
"""

def compute_xrd(raw_data, wavelength):
    xrd_data_batch = []
    xrd_simulator = xrd.XRDSimulator(wavelength=wavelength)
    for idx, irow in raw_data.iterrows():
        # obtain xrd features
        cif_struct = Structure.from_str(irow['cif'], fmt="cif")
        # conventional cell
        sga = SpacegroupAnalyzer(cif_struct, symprec=0.1)
        assert(sga.get_crystal_system() == irow['crystal_system'])
        struct = sga.get_conventional_standard_structure()
        _, recip_latt, features = xrd_simulator.get_pattern(structure=struct)

        # properties of interest
        material_id = irow['material_id']
        crystal_system = irow['crystal_system']
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']
        e_above_hull = irow['e_above_hull']
        elastic = irow['elasticity']
        try:
            elastic_dict = ast.literal_eval(elastic)
            shear_mod = elastic_dict['G_Voigt_Reuss_Hill']
            bulk_mod = elastic_dict['K_Voigt_Reuss_Hill']
            poisson_ratio = elastic_dict['poisson_ratio']
        except:
            shear_mod, bulk_mod, poisson_ratio = 'UNK', 'UNK', 'UNK'

        # property list
        ifeat = [material_id, recip_latt.tolist(), features, crystal_system,
                 band_gap, energy_per_atom, formation_energy_per_atom, e_above_hull,
                 shear_mod, bulk_mod, poisson_ratio]
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
    #input_file = "./raw_data/custom_Xsys_data.csv"
    #out_file = "./raw_data/compute_xrd_Xsys_P343.csv"
    input_file = "./raw_data/custom_MIC_data.csv"
    out_file = "./raw_data/compute_xrd_MIC_P3.csv"

    if not os.path.isfile(input_file):
        print("{} file does not exist, please generate it first..".format(input_file))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(input_file, sep=';', header=0, index_col=None)
    
    # output safeguard
    if os.path.exists(out_file):
        print("Attention, the existing xrd data will be deleted and regenerated..")
    header = [['material_id', 'recip_latt', 'features', 'crystal_system',
               'band_gap', 'energy_per_atom', 'formation_energy_per_atom', 'e_above_hull',
               'shear_mod', 'bulk_mod', 'poisson_ratio']]
    df = pd.DataFrame(header)
    df.to_csv(out_file, sep=';', header=None, index=False, mode='w')
    
    # parameters
    wavelength = 'CuKa' # CuKa by default
    nworkers = max(multiprocessing.cpu_count(), 1)
    n_slices = MP_data.shape[0] // (20*nworkers) # number of batches to split into

    # parallel processing
    MP_data_chunk = np.array_split(MP_data, n_slices)
    print("start computing xrd on {} workers and {} slices".format(nworkers, n_slices))
    for idx, chunk in enumerate(MP_data_chunk):
        # generate xrd point cloud representations
        xrd_data = parallel_computing(chunk, wavelength, nworkers)
        # write to file
        xrd_data.to_csv(out_file, sep=';', header=None, index=False, mode='a')
        print('finished processing chunk {}/{}'.format(idx+1, n_slices)) 


if __name__ == "__main__":
    main()


