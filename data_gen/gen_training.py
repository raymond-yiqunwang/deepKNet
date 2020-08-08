import os
import sys
import shutil
import numpy as np
import pandas as pd

def gen_gap(gap_split_file, out_dir):
    print('gen band gap train val test data using {}'.format(gap_split_file))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    os.mkdir(out_dir)
    file_names = pd.read_csv(gap_split_file, header=0, index_col=None)

    root_dir = "./raw_data/data_pointnet/"

    # train
    train_files = file_names['train'].dropna()
    train_dir = os.path.join(out_dir, 'train/')
    os.mkdir(train_dir)
    for ifile in train_files:
        shutil.copyfile(os.path.join(root_dir,  ifile+'.npy'),
                        os.path.join(train_dir, ifile+'.npy'))
        shutil.copyfile(os.path.join(root_dir,  ifile+'.csv'),
                        os.path.join(train_dir, ifile+'.csv'))        
    # valid
    val_files = file_names['valid'].dropna()
    val_dir = os.path.join(out_dir, 'valid/')
    os.mkdir(val_dir)
    for jfile in val_files:
        shutil.copyfile(os.path.join(root_dir, jfile+'.npy'),
                        os.path.join(val_dir,  jfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir, jfile+'.csv'),
                        os.path.join(val_dir,  jfile+'.csv'))
    # test
    test_files = file_names['test'].dropna()
    test_dir = os.path.join(out_dir, 'test/')
    os.mkdir(test_dir)
    for kfile in test_files:
        shutil.copyfile(os.path.join(root_dir, kfile+'.npy'),
                        os.path.join(test_dir, kfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir, kfile+'.csv'),
                        os.path.join(test_dir, kfile+'.csv'))


if __name__ == "__main__":
    if True:
        # gen band gap
        gap_split_file = "../misc/band_gap_split1.csv"
        gap_out_dir = "./data_band_gap_split1/"
        gen_gap(gap_split_file, gap_out_dir)

    """
    if True:
        # gen topological data
        topo_split_file = "../misc/topo_split1.csv"
        topo_out_dir = "./data_topo_split1/"
        gen_topo(topo_split_file, topo_out_dir)
    """


