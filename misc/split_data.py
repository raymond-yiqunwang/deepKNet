import os
import pandas as pd
import numpy as np
import random

def split_gap(out_file, train_ratio, valid_ratio, test_ratio):
    data_root = "../data_gen/raw_data/data_pointnet/"
    file_names = [fname.split('.')[0] for fname in os.listdir(data_root) \
                 if fname.split('.')[-1] == 'csv']
    random.shuffle(file_names)
    train_split = int(np.floor(len(file_names) * train_ratio))
    valid_split = train_split + int(np.floor((len(file_names) * valid_ratio)))
    train_data = file_names[:train_split]
    valid_data = file_names[train_split:valid_split]
    test_data = file_names[valid_split:]
    out = [train_data, valid_data, test_data]
    out = pd.DataFrame(out).transpose()
    out.to_csv(out_file, header=['train', 'valid', 'test'], index=False)


def split_topo(out_file, train_ratio, valid_ratio, test_ratio):
    topo_data = pd.read_csv("./topo_MPdata_14k.csv", sep=';', header=0, index_col=None)
    file_names = topo_data['material_id'].values.tolist()
    random.shuffle(file_names)
    train_split = int(np.floor(len(file_names) * train_ratio))
    valid_split = train_split + int(np.floor((len(file_names) * valid_ratio)))
    train_data = file_names[:train_split]
    valid_data = file_names[train_split:valid_split]
    test_data = file_names[valid_split:]
    out = [train_data, valid_data, test_data]
    out = pd.DataFrame(out).transpose()
    out.to_csv(out_file, header=['train', 'valid', 'test'], index=False)


def split_elastic(out_file, train_ratio, valid_ratio, test_ratio):
    custom_MPdata = pd.read_csv("../data_gen/raw_data/custom_MPdata.csv", \
                                sep=';', header=0, index_col=None)
    elastic_data = custom_MPdata[['material_id', 'elasticity']].dropna()
    file_names = elastic_data['material_id'].values.tolist()
    random.shuffle(file_names)
    train_split = int(np.floor(len(file_names) * train_ratio))
    valid_split = train_split + int(np.floor((len(file_names) * valid_ratio)))
    train_data = file_names[:train_split]
    valid_data = file_names[train_split:valid_split]
    test_data = file_names[valid_split:]
    out = [train_data, valid_data, test_data]
    out = pd.DataFrame(out).transpose()
    out.to_csv(out_file, header=['train', 'valid', 'test'], index=False)


if __name__ == "__main__":
    # band gap
    if False:
        out_file = "gap_splitx.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_gap(out_file, train_ratio, valid_ratio, test_ratio)

    # topo
    if False:
        out_file = "topo_splitx.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_topo(out_file, train_ratio, valid_ratio, test_ratio)

    # elastic
    if True:
        out_file = "elastic_split5.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_elastic(out_file, train_ratio, valid_ratio, test_ratio)


