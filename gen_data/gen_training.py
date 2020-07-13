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

parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

def generate_dataset(xrd_data, features_dir, target_dir):
    # add MIT column
    xrd_data['MIT'] = (xrd_data['band_gap'] > 1E-3).astype(float)

    # store point cloud representation for each material
    for _, irow in xrd_data.iterrows():
        # unique material ID
        material_id = irow['material_id']
        filename = str(material_id)

        # all primitive features
        recip_latt = np.array(ast.literal_eval(irow['recip_latt']))
        features = ast.literal_eval(irow['features'])
        npoints = len(features)

        recip_pos = np.array([np.dot(recip_latt.T, np.array(features[idx][0])) \
                             for idx in range(npoints)])
        
        n_grid = 64
        image = np.zeros((n_grid, n_grid, n_grid))
        max_r = 2. / 1.54184
        dx = (2 * max_r) / n_grid
        x_grid = (n_grid/2 + recip_pos[:, 0]//dx).astype(int)
        y_grid = (n_grid/2 + recip_pos[:, 1]//dx).astype(int)
        z_grid = (n_grid/2 + recip_pos[:, 2]//dx).astype(int)
        for idx in range(len(recip_pos)):
            image[x_grid[idx], y_grid[idx], z_grid[idx]] += features[idx][1]
        # normalize
        image /= np.amax(image)

        multi_view = np.zeros((3, n_grid, n_grid))
        multi_view[0,:,:] = image.sum(axis=0)
        multi_view[1,:,:] = image.sum(axis=1)
        multi_view[2,:,:] = image.sum(axis=2)
        # write to file
        np.save(features_dir+filename, multi_view)

        # target properties
        band_gap = irow['band_gap'] 
        energy_per_atom = irow['energy_per_atom'] 
        formation_energy_per_atom = irow['formation_energy_per_atom']
        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom]]
        header_target = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom']
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
        root_dir = '../data/'
        filename = "./data_raw/compute_xrd.csv"
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
    
    # process in chunks due to large size
    data_all = pd.read_csv(filename, sep=';', header=0, index_col=None, chunksize=nworkers*50)
    cnt = 0
    for idx, xrd_data in enumerate(data_all):
        # parallel processing
        xrd_data_chunk = np.array_split(xrd_data, nworkers)
        pool = Pool(nworkers)
        args = [(data, features_dir, target_dir) for data in xrd_data_chunk]
        pool.starmap(generate_dataset, args)
        pool.close()
        pool.join()
        cnt += xrd_data.shape[0]
        print('finished processing {} materials'.format(cnt))


if __name__ == "__main__":
    main()


