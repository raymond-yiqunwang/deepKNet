import os
import sys
import math
import shutil
import random
import ast
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

def generate_dataset(xrd_data, topo_data, out_dir):
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
        # CuKa by default
        max_r = 2 / 1.54184
        recip_pos /= max_r
        assert(np.amax(recip_pos) <= 1.0)
        assert(np.amin(recip_pos) >= -1.0)

        # normalize diffraction intensity
        intensity = np.log(1+features[:,-1]) / 3
        intensity = intensity.reshape(-1, 1)
        assert(np.amax(intensity) <= 1.3)
        assert(np.amin(intensity) >= 0.)
        
        # generate point cloud and write to file
        point_cloud = np.concatenate((recip_pos, intensity), axis=1)
        np.save(out_dir+filename, point_cloud)

        # target properties
        band_gap = irow['band_gap'] 
        energy_per_atom = irow['energy_per_atom'] 
        formation_energy_per_atom = irow['formation_energy_per_atom']
        crystal_system = irow['crystal_system']
        e_above_hull = irow['e_above_hull']
        shear_mod = irow['shear_mod']
        bulk_mod = irow['bulk_mod']
        poisson_ratio = irow['poisson_ratio']

        # topo properties
        try:
            topo_row = topo_data.loc[topo_data['material_id'] == material_id].iloc[0]
            topo_class = topo_row['topo_class'][:-1] # get rid of *
            topo_sub_class = topo_row['sub_class'][:-1] # get rid of *
            topo_cross_type = topo_row['cross_type']
        except:
            topo_class, topo_sub_class, topo_cross_type = 'UNK', 'UNK', 'UNK'

        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom,
                       crystal_system, e_above_hull, topo_class,
                       topo_sub_class, topo_cross_type,
                       shear_mod, bulk_mod, poisson_ratio]]
        header_target = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom',
                         'crystal_system', 'e_above_hull', 'topo_class', 
                         'topo_sub_class', 'topo_cross_type',
                         'shear_mod', 'bulk_mod', 'poisson_ratio']
        properties = pd.DataFrame(properties)
        properties.to_csv(out_dir+filename+'.csv', sep=';', \
                          header=header_target, index=False, mode='w')


if __name__ == "__main__":
    # safeguard
    print("Attention, all existing training data will be deleted and regenerated..")
    
    # read xrd raw data
    xrd_file = "./raw_data/compute_xrd_Xsys_P343.csv"
    out_dir = "./raw_data/data_Xsys_P343/"
    #xrd_file = "./raw_data/compute_xrd_MIC343.csv"
    #out_dir = "./raw_data/data_MIC343/"

    if not os.path.isfile(xrd_file):
        print("{} file does not exist, please generate it first..".format(xrd_file))
        sys.exit(1)
    # remove existing output files
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    print('making directory {}'.format(out_dir))
    os.mkdir(out_dir)

    # read topo data
    topo_data = pd.read_csv('../misc/topo_MPdata_14k.csv', sep=';', header=0, index_col=None)

    # parameters
    nworkers = max(multiprocessing.cpu_count(), 1)
    
    # process in chunks due to large size
    data_all = pd.read_csv(xrd_file, sep=';', header=0, index_col=None, chunksize=nworkers*50)
    cnt = 0
    for _, xrd_data in enumerate(data_all):
        # parallel processing
        xrd_data_chunk = np.array_split(xrd_data, nworkers)
        pool = Pool(nworkers)
        pargs = [(data, topo_data, out_dir) for data in xrd_data_chunk]
        pool.starmap(generate_dataset, pargs)
        pool.close()
        pool.join()
        cnt += xrd_data.shape[0]
        print('finished processing {} materials'.format(cnt))


