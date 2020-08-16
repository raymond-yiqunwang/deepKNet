import os
import ast
import pandas as pd
import numpy as np
import random

def split_Xsys(root, out_file, train_ratio, valid_ratio, test_ratio):
    file_names = [fname.split('.')[0] for fname in os.listdir(root) \
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


def split_MIC(root, out_file, train_ratio, valid_ratio, test_ratio):
    file_names = [fname.split('.')[0] for fname in os.listdir(root) \
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


def split_elastic(data_root, out_file, train_ratio, valid_ratio, test_ratio):
    custom_MPdata = pd.read_csv(data_root, sep=';', header=0, index_col=None)
    elastic_data = custom_MPdata[['material_id', 'elasticity']].dropna()
    print('elastic_data size:', elastic_data.shape[0])
    drop_list = []
    Gs, Ks, Ps = [], [], []
    cnt = 0
    for idx, irow in elastic_data.iterrows():
        elastic_dict = ast.literal_eval(irow['elasticity'])
        shear_mod = elastic_dict['G_Voigt_Reuss_Hill']
        Gs.append(shear_mod)
        bulk_mod = elastic_dict['K_Voigt_Reuss_Hill']
        Ks.append(bulk_mod)
        poisson_ratio = elastic_dict['poisson_ratio']
        Ps.append(poisson_ratio)
        if (shear_mod < -1E-6) or (bulk_mod < -1E-6) \
            or (poisson_ratio < -1E-6):
            drop_list.append(idx)
        if shear_mod > 50 and bulk_mod >= 100:
            cnt += 1
    print('number of items to drop:', len(drop_list))
    print('Shear modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
          'min = {:.5f}, max = {:.2f}' \
          .format(np.mean(Gs), np.median(Gs), np.std(Gs), np.min(Gs), np.max(Gs)))
    print('Bulk modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
          'min = {:.5f}, max = {:.2f}' \
          .format(np.mean(Ks), np.median(Ks), np.std(Ks), np.min(Ks), np.max(Ks)))
    print('Poisson ratio: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
          'min = {:.5f}, max = {:.2f}' \
          .format(np.mean(Ps), np.median(Ps), np.std(Ps), np.min(Ps), np.max(Ps)))
    elastic_data = elastic_data.drop(drop_list)
    print('final size of elastic data:', elastic_data.shape[0])
    print('percentile of this threshold:', float(cnt)/elastic_data.shape[0])            
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


"""
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
"""


if __name__ == "__main__":
    # crystal system classification
    if False:
        root = "../data_gen/raw_data/data_Xsys3/"
        out_file = "Xsys_split1.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_Xsys(root, out_file, train_ratio, valid_ratio, test_ratio)

    # metal-insulator classification
    if False:
        root = "../data_gen/raw_data/data_MIC125/"
        out_file = "MIC_split1.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_MIC(root, out_file, train_ratio, valid_ratio, test_ratio)

    # elastic
    if True:
        data_root = "../data_gen/raw_data/custom_Xsys_data.csv"
        out_file = "elastic_split1.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_elastic(data_root, out_file, train_ratio, valid_ratio, test_ratio)

    """
    # topo
    if False:
        out_file = "topo_splitx.csv"
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        split_topo(out_file, train_ratio, valid_ratio, test_ratio)
    """


