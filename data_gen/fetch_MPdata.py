import os
import sys
import random
import numpy as np
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import diffraction_simulator.XRD_simulator as XRD
import diffraction_simulator.ND_simulator as ND
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def fetch_materials_data(root_dir):
    # properties of interest
    properties = [ 
        # ID
        "material_id",
        "icsd_id",
        "icsd_ids",
        # structure
        "cif",
        "spacegroup",
        "crystal_system",
        "volume",
        "nsites", 
        # composition
        "unit_cell_formula",
        "pretty_formula",
        "elements", 
        "nelements",
        # energy
        "energy", 
        "energy_per_atom", 
        "formation_energy_per_atom",
        "e_above_hull",
        # electronic property
        "band_gap",
        "is_hubbard",
        "hubbards",
        "is_compatible",
        # mechanical property
        "elasticity",
        "piezo",
        'diel',
        # other property
        "density", 
        "total_magnetization",
        # additional info
        "oxide_type",
        "warnings",
        "task_ids",
        "tags"
    ]

    # MaterialsProject API settings
    my_API_key = None # put your API key here
    if not my_API_key:
        print("please specify your Materials Project API key here")
        sys.exit()
    m = MPRester(api_key=my_API_key)

    # query all materials data
    query_all = m.query(criteria={}, properties=properties)
    MPdata_all = pd.DataFrame(entry.values() for entry in query_all)
    MPdata_all.columns = properties

    # write cif to file
    cif_dir = os.path.join(root_dir, "MP_cifs")
    if os.path.exists(cif_dir):
        _ = input("MPdata_all/MP_cifs dir already exists, press Enter to continue, Ctrl+C to terminate..")
    else:
        os.mkdir(cif_dir)
    for _, irow in MPdata_all[["material_id", "cif"]].iterrows():
        cif_file = os.path.join(cif_dir, irow["material_id"] + ".cif")
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


def parallel_XRD(root_dir, save_dir, fnames, max_Miller, sym_thresh):
    process_name = multiprocessing.current_process().name
    # initialize XRD simulator
    xrd_simulator = XRD.XRDSimulator(max_Miller=max_Miller)
    for filename in tqdm(fnames):
        cifpath = os.path.join(root_dir, "MP_cifs", filename+".cif")
        assert os.path.isfile(cifpath)
        cif_struct = Structure.from_file(cifpath)
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        # primitive cell
        primitive_struct = sga.get_primitive_standard_structure()
        primitive_features = xrd_simulator.get_pattern(structure=primitive_struct)
        # save primitive diffraction pattern
        np.save(os.path.join(save_dir, filename+"_XRD_primitive.npy"), primitive_features)
        # conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        conventional_features = xrd_simulator.get_pattern(structure=conventional_struct)
        # save primitive diffraction pattern
        np.save(os.path.join(save_dir, filename+"_XRD_conventional.npy"), conventional_features)
    print('Process {} is done..'.format(process_name), flush=True)


def compute_XRD(root_dir):    
    # read all MPdata filenames
    assert os.path.isfile(root_dir+"MPdata_all.csv")
    MPdata = pd.read_csv(root_dir+"MPdata_all.csv", sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    # random shuffle in case of data clustering
    random.shuffle(filenames)
    # save dir
    save_dir = os.path.join(root_dir, "XRD_data")
    if os.path.exists(save_dir):
        _ = input("{} already exists, please check if you want to continue, Ctrl+C to terminate.."\
                  .format(save_dir))
    else:
        os.mkdir(save_dir)
    # parameters
    max_Miller = 4
    sym_thresh = 0.1
    nworkers = multiprocessing.cpu_count()
    print('total size: {}, parallel computing on {} workers..'.format(len(filenames), nworkers))
    # initialize pool of workers
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, save_dir, fnames, max_Miller, sym_thresh) for fnames in file_split]
    pool.starmap_async(parallel_XRD, args)
    pool.close()
    pool.join()
    print('all jobs done, pool closed..')


def parallel_ND(root_dir, save_dir, fnames, max_Miller, sym_thresh):
    process_name = multiprocessing.current_process().name
    # initialize ND simulator
    nd_simulator = ND.NDSimulator(max_Miller=max_Miller)
    for filename in tqdm(fnames):
        cifpath = os.path.join(root_dir, "MP_cifs", filename+".cif")
        assert os.path.isfile(cifpath)
        cif_struct = Structure.from_file(cifpath)
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        # primitive cell
        primitive_struct = sga.get_primitive_standard_structure()
        # skip if this compound contains Ac or Pu (no neutron scattering length available)
        if (Element('Ac') in primitive_struct.species) or (Element('Pu') in primitive_struct.species):
            print('containing Ac or Pu, skipped {}'.format(filename), flush=True)
            continue
        # compute ND patterns
        primitive_features = nd_simulator.get_pattern(structure=primitive_struct)
        # save primitive diffraction pattern
        np.save(os.path.join(save_dir, filename+"_ND_primitive.npy"), primitive_features)
        # conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        conventional_features = nd_simulator.get_pattern(structure=conventional_struct)
        # save primitive diffraction pattern
        np.save(os.path.join(save_dir, filename+"_ND_conventional.npy"), conventional_features)
    print('Process {} is done..'.format(process_name), flush=True)
        

def compute_ND(root_dir):    
    # read all MPdata filenames
    assert os.path.isfile(root_dir+"MPdata_all.csv")
    MPdata = pd.read_csv(root_dir+"MPdata_all.csv", sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    # random shuffle in case of data clustering
    random.shuffle(filenames)
    # save dir
    save_dir = os.path.join(root_dir, "ND_data")
    if os.path.exists(save_dir):
        _ = input("{} already exists, please check if you want to continue, Ctrl+C to terminate.."\
                  .format(save_dir))
    else:
        os.mkdir(save_dir)
    # parameters
    max_Miller = 4
    sym_thresh = 0.1
    nworkers = multiprocessing.cpu_count()
    print('total size: {}, parallel computing on {} workers..'.format(len(filenames), nworkers))
    # initialize pool of workers
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, save_dir, fnames, max_Miller, sym_thresh) for fnames in file_split]
    pool.starmap_async(parallel_ND, args)
    pool.close()
    pool.join()
    print('all jobs done, pool closed..')


if __name__ == "__main__":
    # output directory
    root_dir = "./MPdata_all/"
    
    # fetch data from the Materials Project database
    if os.path.exists(root_dir):
        _ = input("MPdata_all dir already exists, press Enter to continue, Ctrl+C to terminate..")
    else:
        os.mkdir(root_dir)

    print('fetching data from MP..', flush=True)
    fetch_materials_data(root_dir)

    # simulate 3D X-ray diffraction patterns
    print('start computing XRD..', flush=True)
    compute_XRD(root_dir)

    # simulate 3D inelastic neutron scattering patterns
    print('start computing ND..', flush=True)
    compute_ND(root_dir)


