import os
import sys
import shutil
import numpy as np
import pandas as pd

def gen_Xsys(root_dir, Xsys_split_file, out_dir):
    print('gen crystal system train valid test data using {}'.format(Xsys_split_file))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    os.mkdir(out_dir)
    file_names = pd.read_csv(Xsys_split_file, header=0, index_col=None)

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
    valid_files = file_names['valid'].dropna()
    valid_dir = os.path.join(out_dir, 'valid/')
    os.mkdir(valid_dir)
    for jfile in valid_files:
        shutil.copyfile(os.path.join(root_dir,  jfile+'.npy'),
                        os.path.join(valid_dir, jfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir,  jfile+'.csv'),
                        os.path.join(valid_dir, jfile+'.csv'))
    # test
    test_files = file_names['test'].dropna()
    test_dir = os.path.join(out_dir, 'test/')
    os.mkdir(test_dir)
    for kfile in test_files:
        shutil.copyfile(os.path.join(root_dir, kfile+'.npy'),
                        os.path.join(test_dir, kfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir, kfile+'.csv'),
                        os.path.join(test_dir, kfile+'.csv'))


def gen_gap(gap_split_file, out_dir):
    print('gen band gap train valid test data using {}'.format(gap_split_file))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    os.mkdir(out_dir)
    file_names = pd.read_csv(gap_split_file, header=0, index_col=None)

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
    valid_files = file_names['valid'].dropna()
    valid_dir = os.path.join(out_dir, 'valid/')
    os.mkdir(valid_dir)
    for jfile in valid_files:
        shutil.copyfile(os.path.join(root_dir,  jfile+'.npy'),
                        os.path.join(valid_dir, jfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir,  jfile+'.csv'),
                        os.path.join(valid_dir, jfile+'.csv'))
    # test
    test_files = file_names['test'].dropna()
    test_dir = os.path.join(out_dir, 'test/')
    os.mkdir(test_dir)
    for kfile in test_files:
        shutil.copyfile(os.path.join(root_dir, kfile+'.npy'),
                        os.path.join(test_dir, kfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir, kfile+'.csv'),
                        os.path.join(test_dir, kfile+'.csv'))


def gen_topo(topo_split_file, out_dir):
    print('gen topo train valid test data using {}'.format(topo_split_file))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    os.mkdir(out_dir)
    file_names = pd.read_csv(topo_split_file, header=0, index_col=None)

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
    valid_files = file_names['valid'].dropna()
    valid_dir = os.path.join(out_dir, 'valid/')
    os.mkdir(valid_dir)
    for jfile in valid_files:
        shutil.copyfile(os.path.join(root_dir,  jfile+'.npy'),
                        os.path.join(valid_dir, jfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir,  jfile+'.csv'),
                        os.path.join(valid_dir, jfile+'.csv'))
    # test
    test_files = file_names['test'].dropna()
    test_dir = os.path.join(out_dir, 'test/')
    os.mkdir(test_dir)
    for kfile in test_files:
        shutil.copyfile(os.path.join(root_dir, kfile+'.npy'),
                        os.path.join(test_dir, kfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir, kfile+'.csv'),
                        os.path.join(test_dir, kfile+'.csv'))


def gen_elastic(elastic_split_file, out_dir):
    print('gen elastic train valid test data using {}'.format(elastic_split_file))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=False)
    os.mkdir(out_dir)
    file_names = pd.read_csv(elastic_split_file, header=0, index_col=None)

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
    valid_files = file_names['valid'].dropna()
    valid_dir = os.path.join(out_dir, 'valid/')
    os.mkdir(valid_dir)
    for jfile in valid_files:
        shutil.copyfile(os.path.join(root_dir,  jfile+'.npy'),
                        os.path.join(valid_dir, jfile+'.npy'))
        shutil.copyfile(os.path.join(root_dir,  jfile+'.csv'),
                        os.path.join(valid_dir, jfile+'.csv'))
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
    # gen crystal system
    if True:
        root_dir = "./raw_data/data_Xsys27/"
        Xsys_split_file = "../misc/Xsys_split1.csv"
        Xsys_out_dir = "./data_Xsys27_split1/"
        gen_Xsys(root_dir, Xsys_split_file, Xsys_out_dir)

    # gen band gap 
    if False:
        gap_split_file = "../misc/gap_split2.csv"
        gap_out_dir = "./data_gap_split2/"
        gen_gap(gap_split_file, gap_out_dir)

    # gen topological data
    if False:
        topo_split_file = "../misc/topo_split5.csv"
        topo_out_dir = "./data_topo_split5/"
        gen_topo(topo_split_file, topo_out_dir)

    # gen elastic data
    if False:
        elastic_split_file = "../misc/elastic_split5.csv"
        elastic_out_dir = "./data_elastic_split5/"
        gen_elastic(elastic_split_file, elastic_out_dir)


