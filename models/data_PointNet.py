import os
import sys
import ast
import functools
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_PointNet_train_valid_test_loader(root, target, max_Miller, diffraction, cell_type,
                                         randomly_scale_intensity,
                                         systematic_absence,
                                         batch_size, num_data_workers):
    # construct dataset
    dataset = PointNetDataset(root=root,
                              target=target,
                              max_Miller=max_Miller,
                              diffraction=diffraction,
                              cell_type=cell_type,
                              randomly_scale_intensity=randomly_scale_intensity,
                              systematic_absence=systematic_absence)
    train_ratio = 0.7
    valid_ratio = 0.15
    total_size = len(dataset)
    indices = list(range(total_size))
    train_split = int(np.floor(total_size * train_ratio))
    train_sampler = SubsetRandomSampler(indices[:train_split])
    valid_split = train_split + int(np.floor(total_size * valid_ratio))
    valid_sampler = SubsetRandomSampler(indices[train_split:valid_split])
    test_sampler = SubsetRandomSampler(indices[valid_split:])    
    # train DataLoader
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              num_workers=num_data_workers,
                              sampler=train_sampler)
    # valid DataLoader
    valid_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              num_workers=num_data_workers,
                              sampler=valid_sampler)
    # test DataLoader
    test_loader = DataLoader(dataset, 
                             batch_size=batch_size,
                             num_workers=num_data_workers,
                             sampler=test_sampler)
    
    return train_loader, valid_loader, test_loader


class PointNetDataset(Dataset):
    def __init__(self, root, target, max_Miller, diffraction, cell_type, randomly_scale_intensity, systematic_absence):
        self.root = root
        self.target = target
        self.max_Miller = max_Miller
        self.dtype = diffraction
        self.ctype = cell_type
        self.scale = randomly_scale_intensity
        self.sys_abs = systematic_absence
        id_prop_data = pd.read_csv(os.path.join(root, 'id_prop.csv'), \
                                   header=0, sep=',', index_col=None)
        id_prop_data = id_prop_data[['material_id', self.target]]
        id_prop_data = id_prop_data.sample(frac=1)
        self.id_prop = id_prop_data.values
        print('randomly scale intensity: {}, systematic absence: {}'.format(self.scale, self.sys_abs))

    def __getitem__(self, idx):
        material_id, target_prop = self.id_prop[idx]
        # load point cloud data (h, k, l, x, y, z, I_hkl)
        feat_all = np.load(os.path.join(self.root, material_id+f'_{self.dtype}_{self.ctype}.npy'))
        if self.max_Miller > 0:
            condition1 = np.where((np.max(feat_all[:,:3], axis=1) <= self.max_Miller) & \
                                  (np.min(feat_all[:,:3], axis=1) >= -self.max_Miller))
            feat_select = feat_all[condition1]
            assert(feat_select.shape[0] == (2*self.max_Miller+1)**3)
        elif self.max_Miller == -1:
            condition2 = np.where((np.max(feat_all[:,:3], axis=1) <= 1) & \
                                  (np.min(feat_all[:,:3], axis=1) >= 0) & \
                                  (np.sum(feat_all[:,:3], axis=1) <= 1))
            feat_select = feat_all[condition2]
            assert(feat_select.shape[0] == 4)
        else:
            condition3 = np.where((np.min(feat_all[:,:3], axis=1) == 0) & \
                                  (np.sum(feat_all[:,:3], axis=1) == 1)) 
            feat_select = feat_all[condition3]
            assert(feat_select.shape[0] == 3)

        # ensure permutation invariance
        np.random.shuffle(feat_select)
        # rescale xyz into [0, 1] and randomly rotate
        xyz = feat_select[:,3:6] / 5.
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        xyz = xyz @ Q
        # rescale intensity into [0, 1]
        intensity = feat_select[:,-1:]
        intensity = (np.log(intensity+1E-6) - np.log(1E-6)) / 15.
        # randomly scale intensity
        if self.scale:
            assert (not self.sys_abs)
            rand = np.random.random()
            intensity *= rand
        # systematic absence:
        if self.sys_abs:
            assert (not self.scale)
            intensity = (intensity>1E-3).astype(int)
        input_feat = np.concatenate((xyz, intensity), axis=1)

        if self.target == 'band_gap':
            threshold = 1E-3
        elif self.target == 'e_above_hull':
            threshold = 0.02
        elif self.target == 'bulk_modulus':
            threshold = 85.
        elif self.target == 'shear_modulus':
            threshold = 34.
        else:
            raise NotImplementedError

        return torch.Tensor(input_feat.transpose()), torch.LongTensor([target_prop>threshold])

    def __len__(self):
        return self.id_prop.shape[0]


