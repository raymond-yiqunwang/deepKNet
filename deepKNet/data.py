import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_train_val_test_loader(root, target, cut, pad, daug, rot_all, permut,
                              batch_size, num_data_workers, pin_memory):

    print('data aug -- train: {}, val/test: {}'.format(daug, (daug and rot_all)))
    print('permutation: {}'.format(permut))
    train_dataset = deepKNetDataset(root=root+'/train/', target=target, cutoff=cut, 
                                    padding=pad, data_aug=daug, permutation=permut)
    val_dataset = deepKNetDataset(root=root+'/valid/', target=target, cutoff=cut,
                                  padding=pad, data_aug=(daug and rot_all),
                                  permutation=permut)
    test_dataset = deepKNetDataset(root=root+'/test/', target=target,
                                   cutoff=cut, padding=pad, data_aug=(daug and rot_all),
                                   permutation=permut)

    # init DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_data_workers,
                              shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_data_workers,
                            shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_data_workers,
                             shuffle=True, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


class deepKNetDataset(Dataset):
    def __init__(self, root, target, cutoff, padding, data_aug, permutation):
        self.root = root
        self.target = target
        self.cutoff = cutoff
        self.padding = padding
        self.data_aug = data_aug
        self.permutation = permutation
        self.file_names = [fname.split('.')[0] for fname in \
                           os.listdir(self.root)
                           if fname.split('.')[-1] == 'csv']
        # for safety shuffle init
        random.shuffle(self.file_names)

    def __getitem__(self, idx):
        # load point cloud data
        point_cloud = np.load(self.root+self.file_names[idx]+'.npy')
        
        # padding and cutoff
        if self.padding == 'zero':
            if point_cloud.shape[0] < self.cutoff:
                point_cloud = np.pad(point_cloud, 
                                     ((0, self.cutoff-point_cloud.shape[0]), (0, 0)),
                                     mode='constant')
            else:
                point_cloud = point_cloud[:self.cutoff, :]
        elif self.padding == 'periodic':
            while point_cloud.shape[0] < self.cutoff:
                point_cloud = np.repeat(point_cloud, 2, axis=0)
            point_cloud = point_cloud[:self.cutoff, :]
        else:
            raise NotImplementedError

        # apply random 3D rotation for data augmentation
        if self.data_aug:
            alpha, beta, gamma = 0.25 * np.pi * np.random.random(3)
            rot_matrix = [
                np.cos(alpha)*np.cos(beta),
                np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
                np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma),
                np.sin(alpha)*np.cos(beta),
                np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),
                np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma),
                -1*np.sin(beta),
                np.cos(beta)*np.sin(gamma),
                np.cos(beta)*np.cos(gamma)
            ]
            rot_matrix = np.array(rot_matrix).reshape(3,3)
            point_cloud[:,:-1] = np.dot(point_cloud[:,:-1], rot_matrix.T)
        
        if self.permutation:
            np.random.shuffle(point_cloud[1:])

        point_cloud = torch.Tensor(point_cloud.transpose())

        # load target property
        properties = pd.read_csv(self.root+self.file_names[idx]+'.csv',
                                 sep=';', header=0, index_col=None)
        band_gap = properties['band_gap'].values[0]
        e_above_hull = properties['e_above_hull'].values[0]
        topo_class = properties['topo_class'].values[0]
        topo_sub_class = properties['topo_sub_class'].values[0]
        topo_cross_type = properties['topo_cross_type'].values[0]
        crystal_system = properties['crystal_system'].values[0]
        shear_mod = properties['shear_mod'].values[0]
        bulk_mod = properties['bulk_mod'].values[0]
        poisson_ratio = properties['poisson_ratio'].values[0]

        # 6-class crystal family
        if self.target == 'crystal_family':
            cryst_sys_dict = {
                'cubic': 0, 'orthorhombic': 1, 'tetragonal': 2,
                'hexagonal': 3, 'trigonal': 3, 
                'monoclinic': 4, 'triclinic': 5
            }
            prop = torch.Tensor([cryst_sys_dict[crystal_system]])
        # 7-class crystal system
        elif self.target == 'crystal_system':
            cryst_sys_dict = {
                'hexagonal': 0, 'trigonal': 1, 
                'cubic': 2, 'orthorhombic': 3, 'tetragonal': 4,
                'monoclinic': 5, 'triclinic': 6
            }
            prop = torch.Tensor([cryst_sys_dict[crystal_system]])
        # binary metal-insulator classification
        elif self.target == 'MIC':
            prop = torch.Tensor([band_gap>1E-6])
        # binary trivial vs. non-trivial
        elif self.target == 'TIC2':
            assert(topo_class in ['trivial', 'TI', 'SM'])
            prop = torch.Tensor([topo_class=='trivial'])
        # ternary topological classification
        elif self.target == 'TIC3':
            topo_dict = {'trivial': 0, 'TI': 1, 'SM': 2}
            prop = torch.Tensor([topo_dict[topo_class]])
        # binary stability
        elif self.target == 'stability':
            prop = torch.Tensor([e_above_hull<0.01])
        # elasticity
        elif self.target == 'super_hard':
            criterion = bulk_mod >= 200. and shear_mod >= 100. # AUC 0.9+
            #criterion = shear_mod >= 100. # AUC 0.88
            #criterion = bulk_mod >= 200. # AUC 0.9+
            prop = torch.Tensor([criterion])
        else:
            raise NotImplementedError

        return point_cloud, prop.long()

    def __len__(self):
        return len(self.file_names)


