import os
import math
import shutil
import ast
import numpy as np
import pandas as pd
from multiprocessing import Pool


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


# convert Cartesian coord to spherical
def cart2sphere(cart_coord):
    x, y, z = cart_coord[0], cart_coord[1], cart_coord[2]
    r = np.linalg.norm(cart_coord)
    theta = math.acos(cart_coord[2]/r) # [0, pi]
    theta = 2.*(theta/math.pi) - 1. # convert to [-1, 1]
    assert(-1. <= theta <= 1.)
    phi = math.atan2(y, x) # [-pi, pi]
    phi = phi / math.pi # convert to [-1, 1]
    assert(-1. <= phi <= 1.)
    r_scaled = r / 4. # scaling factor 4.0 is the approximated Ewald sphere radius
    assert(1e-3 < r_scaled < 1.)
    return [r_scaled, theta, phi]


def generate_point_cloud(xrd_data, features_dir, target_dir):
    # store point cloud representation for each material
    for _, irow in xrd_data.iterrows():
        # unique material ID
        material_id = irow['material_id']
        filename = str(material_id) + '.csv'
        if os.path.exists(features_dir+filename):
            print('duplicate material_id detected, check data source..')
            sys.exit(1)

        # all primitive features
        recip_latt = ast.literal_eval(irow['recip_latt'])
        hkl = ast.literal_eval(irow['hkl'])
        lorentz_factor = ast.literal_eval(irow['lorentz_factor'])
        i_hkl = ast.literal_eval(irow['i_hkl'])
        atomic_form_factor = ast.literal_eval(irow['atomic_form_factor'])
        
        """!!! please construct features wisely !!!"""
        # spherical coordinate
        recip_xyz = [np.dot(np.array(recip_latt).T, np.array(hkl[idx])) for idx in range(len(hkl))]
        recip_spherical = [cart2sphere(recip_xyz[idx]) for idx in range(len(recip_xyz))]
        # scaled atomic form factor
        aff = np.array(atomic_form_factor)
        lorentz_factor = np.array(lorentz_factor).reshape(-1, 1)
        aff_lorentz = aff * lorentz_factor
        atomic_form_factor_lorentz_scaled = (aff_lorentz/np.amax(aff_lorentz)).tolist()
        features = [recip_spherical[idx] + atomic_form_factor_lorentz_scaled[idx] for idx in range(len(hkl))]
        features = pd.DataFrame(features)
        # transpose features to accommodate PyTorch tensor style
        features_T = features.transpose()
        assert(features_T.shape[0] == 3+94)
        assert(features_T.shape[1] == 512)
        # write features_T
        features_T.to_csv(features_dir+filename, sep=';', header=None, index=False)

        # target properties
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']
        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom]]
        header_target = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom']
        properties = pd.DataFrame(properties)
        properties.to_csv(target_dir+filename, sep=';', header=header_target, index=False)


def main():
    # safeguard
    _ = input("Attention, all existing training data will be deleted and regenerated.. \
        \n>> Hit Enter to continue, Ctrl+c to terminate..")
    # remove existing csv files
    target_dir = '../data/target/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=False)
    os.mkdir(target_dir)
    features_dir = '../data/features/'
    if os.path.exists(features_dir):
        shutil.rmtree(features_dir, ignore_errors=False)
    os.mkdir(features_dir)

    # read xrd raw data
    print("Reading raw data..")
    xrd_data = pd.read_csv("./data_raw/compute_xrd.csv", sep=';', header=0, index_col=None)
    print("Finished reading raw data, start generating dataset..")

    # parameters
    nworkers = 12

    # parallel processing
    xrd_data_chunk = np.array_split(xrd_data, nworkers)
    pool = Pool(nworkers)
    args = [(data, features_dir, target_dir) for data in xrd_data_chunk]
    pool.starmap(generate_point_cloud, args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()


