import os
import time
import random
import shutil
import numpy as np
import pandas as pd

from pymatgen import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import diffraction_simulator.XRD_simulator as XRD
import diffraction_simulator.ND_simulator as ND

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
    my_API_key = "YOUR_MP_API_KEY"
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


def parallel_XRD(root_dir, fnames, wavelength, sym_thresh):
    # initialize XRD simulator
    xrd_simulator = XRD.XRDSimulator(wavelength=wavelength)
    for idx, filename in enumerate(fnames):
        if (idx+1)%100 == 0:
            print('this worker finished processing {} materials..'.format(idx+1), flush=True)
        filepath = os.path.join(root_dir, filename+".cif")
        try:
            assert(os.path.isfile(filepath))
        except:
            print('{} file is missing, abort..'.format(filepath))
        cif_struct = Structure.from_file(filepath)
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        
        # conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        _, conventional_recip_latt, conventional_features = \
            xrd_simulator.get_pattern(structure=conventional_struct, two_theta_range=None)
        # save conventional reciprocal lattice vector
        np.save(os.path.join(root_dir, filename+"_conventional_basis.npy"), \
                conventional_recip_latt)
        # save conventional diffraction pattern
        np.save(os.path.join(root_dir, filename+"_XRD_conventional.npy"), \
                np.array(conventional_features))
        
    print('this process is done..', flush=True)


def compute_XRD(root_dir):    
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
    pool.starmap_async(parallel_XRD, args)
    pool.close()
    pool.join()
    
    print('all jobs done, pool closed..')


def parallel_ND(root_dir, fnames, sym_thresh):
    print('size of this batch:', len(fnames), flush=True)
    # initialize XRD simulator
    nd_simulator = ND.NDSimulator()
    for idx, filename in enumerate(fnames):
        if (idx+1)%100 == 0:
            print('this worker finished processing {} materials..'.format(idx+1), flush=True)
        filepath = os.path.join(root_dir, filename+".cif")
        try:
            assert(os.path.isfile(filepath))
        except:
            print('{} file is missing, abort..'.format(filepath))
        cif_struct = Structure.from_file(filepath)
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        # conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        # skip if this compound contains Ac or Pu (no neutron scattering length available)
        if (Element('Ac') in conventional_struct.species) or (Element('Pu') in conventional_struct.species):
            print('containing Ac or Pu, skipped {}'.format(filename), flush=True)
            continue
        _, conventional_recip_latt, conventional_nd_features = \
            nd_simulator.get_pattern(structure=conventional_struct, two_theta_range=None)
        # addtional safe guard in parallel processing
        try:
            assert(np.array_equal(conventional_recip_latt, 
                                  np.load(os.path.join(root_dir, filename+"_conventional_basis.npy"))))
        except:
            print('file {} reciprocal lattice mismatch'.format(filename), flush=True)

        # save conventional diffraction pattern
        np.save(os.path.join(root_dir, filename+"_ND_conventional.npy"), \
                np.array(conventional_nd_features))
    print('this process is done..', flush=True)
        

def compute_ND(root_dir):    
    # read all MPdata filenames
    assert(os.path.isfile(root_dir+"MPdata_all.csv"))
    MPdata = pd.read_csv(root_dir+"MPdata_all.csv", sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    random.shuffle(filenames)

    # parameters
    sym_thresh = 0.1
    nworkers = multiprocessing.cpu_count()
    print('total size: {}, parallel computing on {} workers..'.format(len(filenames), nworkers))

    # initialize pool of workers
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, fnames, sym_thresh) for fnames in file_split]
    pool.starmap_async(parallel_ND, args)
    pool.close()
    pool.join()

    print('all jobs done, pool closed..')


if __name__ == "__main__":
    # output directory
    root_dir = "./MPdata_all/"
    if os.path.exists(root_dir):
        _ = input("MPdata_all dir already exists, press Enter to continue, Ctrl+C to terminate..")
    else:
        os.mkdir(root_dir)

    # obtain data from MP database
    print('fetching data from MP..', flush=True)
    fetch_materials_data(root_dir)

    # simulate 3D X-ray diffraction patterns
    print('start computing XRD..', flush=True)
    compute_XRD(root_dir)

    # simulate 3D inelastic neutron scattering patterns
    print('start computing ND..', flush=True)
    compute_ND(root_dir)

