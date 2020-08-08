import os
import pandas as pd
import numpy as np
import random

def split_gap(out_file, train_ratio, val_ratio, test_ratio):
    data_root = "../data_gen/raw_data/data_pointnet/"
    file_names = [fname.split('.')[0] for fname in os.listdir(data_root) \
                 if fname.split('.')[-1] == 'csv']
    random.shuffle(file_names)
    train_split = int(np.floor(len(file_names) * train_ratio))
    val_split = train_split + int(np.floor((len(file_names) * val_ratio)))
    train_data = file_names[:train_split]
    val_data = file_names[train_split:val_split]
    test_data = file_names[val_split:]
    out = [train_data, val_data, test_data]
    out = pd.DataFrame(out).transpose()
    out.to_csv(out_file, header=['train', 'valid', 'test'], index=False)


def split_topo(out_file, train_ratio, val_ratio, test_ratio):
    topo_data = pd.read_csv("./topo_MPdata_14k.csv", sep=';', header=0, index_col=None)
    file_names = topo_data['material_id'].values.tolist()
#    random.shuffle(file_names)
    print(file_names[:5])
    train_split = int(np.floor(len(file_names) * train_ratio))
    val_split = train_split + int(np.floor((len(file_names) * val_ratio)))
    train_data = file_names[:train_split]
    val_data = file_names[train_split:val_split]
    test_data = file_names[val_split:]
    out = [train_data, val_data, test_data]
    out = pd.DataFrame(out).transpose()
    out.to_csv(out_file, header=['train', 'valid', 'test'], index=False)


if __name__ == "__main__":
    # band gap
    if False:
        out_file = "band_gap_splitx.csv"
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        split_gap(out_file, train_ratio, val_ratio, test_ratio)

    # topo
    if True:
        out_file = "topo_split1.csv"
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        split_topo(out_file, train_ratio, val_ratio, test_ratio)


