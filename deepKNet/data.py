import os
import ast
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_train_val_test_loader(root, target, npt, pt_dim, pad, daug, rot_all,
                              permut, batch_size, num_data_workers, pin_memory):

    print('data aug -- train: {}, val/test: {}'.format(daug, (daug and rot_all)))
    print('permutation: {}'.format(permut))
    train_dataset = deepKNetDataset(root=os.path.join(root, 'train'), 
                                    target=target, npoint=npt, point_dim=pt_dim,
                                    padding=pad, data_aug=daug, permutation=permut)
    val_dataset = deepKNetDataset(root=os.path.join(root, 'valid'),
                                  target=target, npoint=npt, point_dim=pt_dim,
                                  padding=pad, data_aug=(daug and rot_all), permutation=permut)
    test_dataset = deepKNetDataset(root=os.path.join(root, 'test'),
                                   target=target, npoint=npt, point_dim=pt_dim,
                                   padding=pad, data_aug=(daug and rot_all), permutation=permut)

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
    def __init__(self, root, target, npoint, point_dim, padding, data_aug, permutation):
        self.root = root
        self.target = target
        self.npoint = npoint
        self.point_dim = point_dim
        self.padding = padding
        self.data_aug = data_aug
        self.permutation = permutation
        id_prop_data = pd.read_csv(os.path.join(root, 'id_prop.csv'), \
                                   header=0, sep=',', index_col=None)
        self.id_prop = id_prop_data.values
        # for safety shuffle init
        random.shuffle(self.id_prop)

    def __getitem__(self, idx):
        material_id, target_prop = self.id_prop[idx]
        # load point cloud data
        point_cloud = np.load(os.path.join(self.root, material_id+'.npy'))
        
        # padding and cutoff
        if self.padding == 'zero':
            if point_cloud.shape[0] < self.npoint:
                point_cloud = np.pad(point_cloud, 
                                     ((0, self.npoint-point_cloud.shape[0]), (0, 0)),
                                     mode='constant')
            else:
                point_cloud = point_cloud[:self.npoint, :]
        elif self.padding == 'periodic':
            while point_cloud.shape[0] < self.npoint:
                point_cloud = np.repeat(point_cloud, 2, axis=0)
            point_cloud = point_cloud[:self.npoint, :]
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

        if self.point_dim != 4:
            assert(self.point_dim == 3)
            point_cloud = point_cloud[:,:-1]

        point_cloud = torch.Tensor(point_cloud.transpose())

        # 6-class crystal family
        if self.target == 'crystal_family':
            cryst_sys_dict = {
                'cubic': 0, 'orthorhombic': 1, 'tetragonal': 2,
                'monoclinic': 3, 'triclinic': 4,
                'hexagonal': 5, 'trigonal': 5 
            }
            prop = torch.Tensor([cryst_sys_dict[target_prop]])
        # 7-class crystal system
        elif self.target == 'crystal_system':
            cryst_sys_dict = {
                'cubic': 0, 'orthorhombic': 1, 'tetragonal': 2,
                'monoclinic': 3, 'triclinic': 4,
                'hexagonal': 5, 'trigonal': 6
            }
            prop = torch.Tensor([cryst_sys_dict[target_prop]])
        elif self.target == 'tri_hex_cls':
            assert(target_prop in ['hexagonal', 'trigonal'])
            prop = torch.Tensor([target_prop == 'hexagonal'])
        # binary metal-insulator classification
        elif self.target == 'MIC':
            prop = torch.Tensor([target_prop>1E-6])
        # elasticity
        elif self.target == 'elasticity':
            target_prop = ast.literal_eval(target_prop)
            shear_mod, bulk_mod, poisson_ratio = \
                target_prop[0], target_prop[1], target_prop[2]
            criterion = bulk_mod >= 100. # AUC 0.9+
            #criterion = shear_mod >= 100. # AUC 0.88
            prop = torch.Tensor([criterion])
        # binary stability
        elif self.target == 'stability':
            prop = torch.Tensor([target_prop<0.02])
        # binary trivial vs. non-trivial
        elif self.target == 'TIC2':
            assert(target_prop in ['trivial', 'TI', 'SM'])
            prop = torch.Tensor([target_prop=='trivial'])
        # ternary topological classification
        elif self.target == 'TIC3':
            topo_dict = {'trivial': 0, 'TI': 1, 'SM': 2}
            prop = torch.Tensor([topo_dict[target_prop]])
        else:
            raise NotImplementedError

        return point_cloud, prop.long()

    def __len__(self):
        return self.id_prop.shape[0]


