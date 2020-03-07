import os
import sys
import math
import shutil
import ast
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

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

# convert Cartesian coord to spherical
def cart2sphere(cart_coord):
    x, y, z = cart_coord[0], cart_coord[1], cart_coord[2]
    r = np.linalg.norm(cart_coord) # [0, r_max]
    theta = math.acos(z/r) # [0, pi]
    phi = math.atan2(y, x) # [-pi, pi]
    ### scale r, theta, phi to [0, 1]
    r_scaled = r / (2. / 1.54184) # within the Ewald sphere
    theta_scaled = theta / math.pi 
    phi_scaled = (phi+math.pi) / (2*math.pi)
    assert(1e-3 < r_scaled <= 1.)
    assert(0. <= theta_scaled <= 1.)
    assert(0. <= phi_scaled <= 1.)
    return [r_scaled, theta_scaled, phi_scaled]


def extend_and_truncate(input_list, npoints):
    # extend list
    while len(input_list) < npoints:
        input_list.extend(input_list)
    # cutoff to a certain length
    return input_list[:npoints]
    

def generate_point_cloud(xrd_data, features_dir, target_dir, npoints):
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
        hkl = extend_and_truncate(hkl, npoints)
        intensity_hkl = ast.literal_eval(irow['intensity_hkl'])
        intensity_hkl = extend_and_truncate(intensity_hkl, npoints)
        atomic_form_factor = ast.literal_eval(irow['atomic_form_factor'])
        atomic_form_factor = extend_and_truncate(atomic_form_factor, npoints)
        
        """!!! please construct features wisely !!!"""
        # spherical coordinates
        recip_xyz = [np.dot(np.array(recip_latt).T, np.array(hkl[idx])) for idx in range(npoints)]
        recip_spherical = [cart2sphere(recip_xyz[idx]) for idx in range(npoints)]
        # total intensity
        intensity = np.array(intensity_hkl) / max(intensity_hkl)
        # atomic form factor
        aff = np.array(atomic_form_factor)
        aff /= np.maximum(1., aff.max(axis=1, keepdims=True))
        aff = aff.tolist()
        # build features
        features = [recip_spherical[idx] + [intensity[idx]] + aff[idx] \
                    for idx in range(npoints)]
        feat = np.array(features).flatten()
        assert(np.max(np.abs(feat)) < 1+1e-12)
        features = pd.DataFrame(features)
        
        # transpose features to accommodate PyTorch tensor style
        features_T = features.transpose()
        assert(features_T.shape[0] == 3+1+94)
        assert(features_T.shape[1] == npoints)
        # write features_T
        features_T.to_csv(features_dir+filename, sep=';', header=None, index=False, mode='w')

        # target properties
        band_gap = irow['band_gap'] 
        energy_per_atom = irow['energy_per_atom'] 
        formation_energy_per_atom = irow['formation_energy_per_atom']
        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom]]
        header_target = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom']
        properties = pd.DataFrame(properties)
        properties.to_csv(target_dir+filename, sep=';', header=header_target, index=False, mode='w')


def main():
    global args

    # safeguard
    _ = input("Attention, all existing training data will be deleted and regenerated.. \
        \n>> Hit Enter to continue, Ctrl+c to terminate..")
    
    # read xrd raw data
    if not args.debug:
        filename = "./data_raw/compute_xrd.csv"
        root_dir = '../data/'
    else:
        root_dir = './data_raw/debug_data/'
        filename = root_dir + "debug_compute_xrd.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # remove existing output files
    if not os.path.exists(root_dir):
        print('making directory {}'.format(root_dir))
        os.mkdir(root_dir)
    target_dir = root_dir + 'target/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=False)
    os.mkdir(target_dir)
    features_dir = root_dir + 'features/'
    if os.path.exists(features_dir):
        shutil.rmtree(features_dir, ignore_errors=False)
    os.mkdir(features_dir)
    
    # parameters
    nworkers = max(multiprocessing.cpu_count()-4, 1)
    npoints = 4096 # number of k-points
    """ 
    P.S. no k-point truncation in compute_xrd.csv,
         all k-points in the Ewald sphere are stored,
         therefore each material will have its unique
         number of k-points
    """
    
    # process in chunks due to large size
    data_all = pd.read_csv(filename, sep=';', header=0, index_col=None, chunksize=nworkers*100)
    cnt = 0
    for idx, xrd_data in enumerate(data_all):
        # parallel processing
        xrd_data_chunk = np.array_split(xrd_data, nworkers)
        pool = Pool(nworkers)
        args = [(data, features_dir, target_dir, npoints) for data in xrd_data_chunk]
        pool.starmap(generate_point_cloud, args)
        pool.close()
        pool.join()
        cnt += xrd_data.shape[0]
        print('finished processing {} materials'.format(cnt))


if __name__ == "__main__":
    main()


