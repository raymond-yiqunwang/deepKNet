import os
import time
import random
import shutil
import numpy as np
import pandas as pd
from pymatgen import MPRester
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import xrd_simulator.xrd_simulator as xrd
import multiprocessing
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def fetch_materials_data(root_dir):
    # properties of interest
    properties = [ 
        "material_id", "icsd_ids", "cif",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "crystal_system",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", 
        "formation_energy_per_atom", "e_above_hull",
        "band_gap", "elasticity", "density", 
        "total_magnetization",
        "warnings", "tags"
    ]
    # MaterialsProject API settings
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query all materials data
    query_all = m.query(criteria={}, properties=properties)
    MPdata_all = pd.DataFrame(entry.values() for entry in query_all)
    MPdata_all.columns = properties

    # write cif to file
    for _, irow in MPdata_all[["material_id", "cif"]].iterrows():
        cif_file = os.path.join(root_dir, irow["material_id"] + ".cif")
        with open(cif_file, 'w') as f:
            f.write(irow["cif"])
    MPdata_all = MPdata_all.drop(columns=["cif"])

    # materials with calculated band structures
    query_band = m.query(criteria={"has": "bandstructure"},
                         properties=["material_id"])
    band_filenames = [list(entry.values())[0] for entry in query_band]
    MPdata_all['has_band_structure'] = MPdata_all["material_id"].isin(band_filenames)
    
    # write properties to file
    out_file = os.path.join(root_dir, "MPdata_all.csv")
    MPdata_all.to_csv(out_file, sep=';', index=False, header=MPdata_all.columns, mode='w')


def parallel_computing(root_dir, fnames, wavelength, sym_thresh):
    # initialize XRD simulator
    xrd_simulator = xrd.XRDSimulator(wavelength=wavelength)
    for idx, filename in enumerate(fnames):
        if (idx+1)%100 == 0:
            print('this worker finished processing {} materials..'.format(idx+1), flush=True)
        with open(os.path.join(root_dir, filename+".cif")) as f:
            cif_struct = Structure.from_str(f.read(), fmt="cif")
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        
        # conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        _, conventional_recip_latt, conventional_features = \
            xrd_simulator.get_pattern(structure=conventional_struct)
        # save conventional reciprocal lattice vector
        np.save(os.path.join(root_dir, filename+"_conventional_basis.npy"), \
                conventional_recip_latt)
        # save conventional diffraction pattern
        np.save(os.path.join(root_dir, filename+"_conventional.npy"), \
                np.array(conventional_features))
        
        # primitive cell
        primitive_struct = sga.get_primitive_standard_structure()
        _, primitive_recip_latt, primitive_features = \
            xrd_simulator.get_pattern(structure=primitive_struct)
        # save primitive reciprocal lattice vector
        np.save(os.path.join(root_dir, filename+"_primitive_basis.npy"), \
                primitive_recip_latt)
        # save primitive diffraction pattern
        np.save(os.path.join(root_dir, filename+"_primitive.npy"), \
                np.array(primitive_features))


def compute_xrd(root_dir):    
    # read all MPdata filenames
    assert(os.path.isfile(root_dir+"MPdata_all.csv"))
    MPdata = pd.read_csv(root_dir+"MPdata_all.csv", sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    random.shuffle(filenames)

    # parameters
    wavelength = 'CuKa' # CuKa by default
    sym_thresh = 0.1
    nworkers = multiprocessing.cpu_count()
    print('total size: {}, parallel computing on {} workers..'.format(len(filenames), nworkers))

    # initialize pool of workers
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, fnames, wavelength, sym_thresh) for fnames in file_split]
    pool.starmap_async(parallel_computing, args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    # output directory
    root_dir = "./MPdata_all/"
    if os.path.exists(root_dir):
        _ = input("MPdata_all dir already exists, deleting.. press enter to continue")
#        shutil.rmtree(root_dir)
#    os.mkdir(root_dir)

    # obtain data from MP database
#    print('fetching data from MP..', flush=True)
#    fetch_materials_data(root_dir)

    # simulate 3D XRD patterns
#    print('start computing XRD..', flush=True)
#    compute_xrd(root_dir)


