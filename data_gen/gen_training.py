import os
import sys
import math
import shutil
import ast
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from PIL import Image
from multiprocessing import Pool
from pyquaternion import Quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

def generate_dataset(xrd_data, features_dir, target_dir):
    # store point cloud representation for each material
    for _, irow in xrd_data.iterrows():
        # unique material ID
        material_id = irow['material_id']
        filename = str(material_id)

        # all primitive features
        recip_latt = np.array(ast.literal_eval(irow['recip_latt']))
        features = np.array(ast.literal_eval(irow['features']))

        # reciprocal points in Cartesian coordinate
        recip_pos = np.dot(features[:,:-1], recip_latt)

        n_grid = 65 # (-32 -- 0 -- 32)
        multi_view = np.zeros((21, n_grid, n_grid))

        # rotation matrices
        rot_matrices = [
            # identity
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            # 60 degrees along z axis
            [[np.cos(np.pi/3), np.sin(np.pi/3), 0], 
             [-np.sin(np.pi/3), np.cos(np.pi/3), 0], 
             [0, 0, 1]],
            # 120 degrees along z axis
            [[np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0], 
             [-np.sin(2*np.pi/3), np.cos(2*np.pi/3), 0], 
             [0, 0, 1]],
            # 60 degrees along y axis
            [[np.cos(np.pi/3), 0, np.sin(np.pi/3)], 
             [0, 1, 0],
             [-np.sin(np.pi/3), 0, np.cos(np.pi/3)]], 
            # 120 degrees along y axis
            [[np.cos(2*np.pi/3), 0, np.sin(2*np.pi/3)], 
             [0, 1, 0],
             [-np.sin(2*np.pi/3), 0, np.cos(2*np.pi/3)]], 
            # 60 degrees along x axis
            [[1, 0, 0],
             [0, np.cos(np.pi/3), np.sin(np.pi/3)], 
             [0, -np.sin(np.pi/3), np.cos(np.pi/3)]], 
            # 120 degrees along x axis
            [[1, 0, 0],
             [0, np.cos(2*np.pi/3), np.sin(2*np.pi/3)], 
             [0, -np.sin(2*np.pi/3), np.cos(2*np.pi/3)]]
        ]

        max_r = 2. / 1.54184
        dx = 2 * max_r / n_grid
        view_id = 0
        for rot_matrix in rot_matrices:
            image = np.zeros((n_grid, n_grid, n_grid))
            # rotate
            recip_pos_rot = np.dot(recip_pos, rot_matrix)
            # assign to gird
            x_grid = (n_grid//2+np.round(recip_pos_rot[:,0]/dx)).astype(int)
            y_grid = (n_grid//2+np.round(recip_pos_rot[:,1]/dx)).astype(int)
            z_grid = (n_grid//2+np.round(recip_pos_rot[:,2]/dx)).astype(int)
            for idx in range(len(recip_pos_rot)):
                image[x_grid[idx], y_grid[idx], z_grid[idx]] += features[idx,-1]
            # normalize
            image /= np.amax(image)

            multi_view[view_id,:,:] = image.sum(axis=0)
            multi_view[view_id+1,:,:] = image.sum(axis=1)
            multi_view[view_id+2,:,:] = image.sum(axis=2)
            view_id += 3
        
        # write to file
        np.save(features_dir+filename, multi_view)

        # target properties
        band_gap = irow['band_gap'] 
        energy_per_atom = irow['energy_per_atom'] 
        formation_energy_per_atom = irow['formation_energy_per_atom']
        MIT = irow['MIT']
        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom, MIT]]
        header_target = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom', 'MIT']
        properties = pd.DataFrame(properties)
        properties.to_csv(target_dir+filename+'.csv', sep=';', \
                          header=header_target, index=False, mode='w')


def main():
    global args

    # safeguard
    _ = input("Attention, all existing training data will be deleted and regenerated.. \
        \n>> Hit Enter to continue, Ctrl+c to terminate..")
    
    # read xrd raw data
    if not args.debug:
        root_dir = './data/'
        filename = "./raw_data/compute_xrd.csv"
    else:
        root_dir = './raw_data/debug_data/'
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
    nworkers = max(multiprocessing.cpu_count()-2, 1)
    
    # process in chunks due to large size
    data_all = pd.read_csv(filename, sep=';', header=0, index_col=None, chunksize=nworkers*50)
    cnt = 0
    for idx, xrd_data in enumerate(data_all):
        # parallel processing
        xrd_data_chunk = np.array_split(xrd_data, nworkers)
        pool = Pool(nworkers)
        pargs = [(data, features_dir, target_dir) for data in xrd_data_chunk]
        pool.starmap(generate_dataset, pargs)
        pool.close()
        pool.join()
        cnt += xrd_data.shape[0]
        print('finished processing {} materials'.format(cnt))


if __name__ == "__main__":
    main()


