import os
import sys
import math
import shutil
import random
import ast
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./', metavar='DATA_DIR')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--wavelength',  default='CuKa', metavar='X-RAY WAVELENGTH')
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
        if args.wavelength == 'CuKa':
            max_r = 2 / 1.54184
        elif args.wavelength == 'mywave':
            max_r = 2 / 1.0
        else:
            raise NotImplementedError
        recip_pos /= max_r
        assert(np.amax(recip_pos) <= 1.0)
        assert(np.amin(recip_pos) >= -1.0)

        # normalize diffraction intensity
        intensity = np.log(1+features[:,-1]) / 3
        intensity = intensity.reshape(-1, 1)
        assert(np.amax(intensity) <= 1.5)
        assert(np.amin(intensity) >= 0.)

        # generate point cloud and write to file
        point_cloud = np.concatenate((recip_pos, intensity), axis=1)
        np.save(features_dir+filename, point_cloud)

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
    print("Attention, all existing training data will be deleted and regenerated..")
    
    # read xrd raw data
    if not args.debug:
        root_dir = os.path.join(args.root, 'data_pointnet/')
        xrd_file = os.path.join(args.root, 'raw_data/compute_xrd_'+args.wavelength+'.csv')
    else:
        root_dir = os.path.join(args.root, 'raw_data/debug_data/data_pointnet/')
        xrd_file = os.path.join(args.root, 'raw_data/debug_data/debug_compute_xrd_' \
                                            +args.wavelength+'.csv')
    if not os.path.isfile(xrd_file):
        print("{} file does not exist, please generate it first..".format(xrd_file))
        sys.exit(1)
    # remove existing output files
    if not os.path.exists(root_dir):
        print('making directory {}'.format(root_dir))
        os.mkdir(root_dir)
    target_dir = os.path.join(root_dir, 'target/')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=False)
    os.mkdir(target_dir)
    features_dir = os.path.join(root_dir, 'features/')
    if os.path.exists(features_dir):
        shutil.rmtree(features_dir, ignore_errors=False)
    os.mkdir(features_dir)
    
    # parameters
    nworkers = max(multiprocessing.cpu_count(), 1)
    
    # process in chunks due to large size
    data_all = pd.read_csv(xrd_file, sep=';', header=0, index_col=None, chunksize=nworkers*50)
    cnt = 0
    for _, xrd_data in enumerate(data_all):
        # parallel processing
        xrd_data_chunk = np.array_split(xrd_data, nworkers)
        pool = Pool(nworkers)
        pargs = [(data, features_dir, target_dir) for data in xrd_data_chunk]
        pool.starmap(generate_dataset, pargs)
        pool.close()
        pool.join()
        cnt += xrd_data.shape[0]
        print('finished processing {} materials'.format(cnt))


def train_val_test_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    global args
    print('train val test split..')
    if not args.debug:
        root_dir = os.path.join(args.root, 'data_pointnet/')
    else:
        root_dir = os.path.join(args.root, 'raw_data/debug_data/data_pointnet/')
    features_dir = os.path.join(root_dir, 'features/')
    target_dir = os.path.join(root_dir, 'target/')
    random.seed(123)
    file_names = [fname.split('.')[0] for fname in os.listdir(target_dir)]
    random.shuffle(file_names)
    # train
    train_dir = os.path.join(root_dir, 'train/')
    os.mkdir(train_dir)
    train_split = int(np.floor(len(file_names) * train_ratio))
    for ifile in file_names[:train_split]:
        shutil.copyfile(os.path.join(features_dir, ifile+'.npy'),
                        os.path.join(train_dir, ifile+'.npy'))
        shutil.copyfile(os.path.join(target_dir, ifile+'.csv'),
                        os.path.join(train_dir, ifile+'.csv'))
    # valid
    val_dir = os.path.join(root_dir, 'valid/')
    os.mkdir(val_dir)
    val_split = train_split + int(np.floor((len(file_names) * val_ratio)))
    for jfile in file_names[train_split:val_split]:
        shutil.copyfile(os.path.join(features_dir, jfile+'.npy'),
                        os.path.join(val_dir, jfile+'.npy'))
        shutil.copyfile(os.path.join(target_dir, jfile+'.csv'),
                        os.path.join(val_dir, jfile+'.csv'))
    # test
    test_dir = os.path.join(root_dir, 'test/')
    os.mkdir(test_dir)
    for kfile in file_names[val_split:]:
        shutil.copyfile(os.path.join(features_dir, kfile+'.npy'),
                        os.path.join(test_dir, kfile+'.npy'))
        shutil.copyfile(os.path.join(target_dir, kfile+'.csv'),
                        os.path.join(test_dir, kfile+'.csv'))
        
    shutil.rmtree(target_dir, ignore_errors=False)
    shutil.rmtree(features_dir, ignore_errors=False)
    

def check_npoint(wavelength='CuKa'):
    xrd_file = './raw_data/compute_xrd_'+wavelength+'.csv'
    data_all = pd.read_csv(xrd_file, sep=';', header=0, index_col=None, chunksize=100)
    npoints = []
    for idx, xrd_data in enumerate(data_all):
        feat_len = xrd_data['features'].apply(ast.literal_eval).apply(len)
        feat_len_list = feat_len.tolist()
        print('Chunk min: {}, max: {}, mean: {}, median: {}, std: {}'.format(
               np.min(feat_len_list), np.max(feat_len_list), np.mean(feat_len_list),
               np.median(feat_len_list), np.std(feat_len_list)))
        npoints += feat_len_list

    print('Total min: {}, max: {}, mean: {}, median: {}, std: {}'.format(
           np.min(npoints), np.max(npoints), np.mean(npoints),
           np.median(npoints), np.std(npoints)))


if __name__ == "__main__":
    if True:
        main()
        train_val_test_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    if False:
        # CuKa:
        #   min: 50, max: 53000, mean: 4000, median: 3000, std: 4000
        check_npoint(wavelength='CuKa')


