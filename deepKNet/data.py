import os
import ast
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_train_valid_test_loader(root, target, npt, pt_dim, pad, daug, rnd_intensity, sys_abs,
                                rot_all, permut, batch_size, num_data_workers, pin_memory):
    print('data aug -- train: {}, val/test: {}'.format(daug, (daug and rot_all)))
    print('permutation: {}'.format(permut))
    print('randomize intensity: {}'.format(rnd_intensity))
    print('systematic absence: {}'.format(sys_abs))
    
    # train DataLoader
    train_dataset = deepKNetDataset(root=os.path.join(root, 'train'), 
                                    target=target, npoint=npt, point_dim=pt_dim,
                                    padding=pad, data_aug=daug, 
                                    rand_intensity=rnd_intensity, sys_absence=sys_abs,
                                    permutation=permut)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_data_workers,
                              shuffle=True, pin_memory=pin_memory)
    
    # valid DataLoader
    valid_dataset = deepKNetDataset(root=os.path.join(root, 'valid'),
                                    target=target, npoint=npt, point_dim=pt_dim,
                                    padding=pad, data_aug=(daug and rot_all), 
                                    rand_intensity=rnd_intensity, sys_absence=sys_abs,
                                    permutation=permut)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_data_workers,
                              shuffle=True, pin_memory=pin_memory)
    
    # test DataLoader
    test_dataset = deepKNetDataset(root=os.path.join(root, 'test'),
                                   target=target, npoint=npt, point_dim=pt_dim,
                                   padding=pad, data_aug=(daug and rot_all),
                                   rand_intensity=rnd_intensity, sys_absence=sys_abs,
                                   permutation=permut)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_data_workers,
                             shuffle=True, pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader


class deepKNetDataset(Dataset):
    def __init__(self, root, target, npoint, point_dim, padding, \
                       data_aug, rand_intensity, sys_absence, permutation):
        self.root = root
        self.target = target
        self.npoint = npoint
        self.point_dim = point_dim
        self.padding = padding
        self.data_aug = data_aug
        self.random_intensity = rand_intensity
        self.systematic_absence = sys_absence
        self.permutation = permutation
        id_prop_data = pd.read_csv(os.path.join(root, 'id_prop.csv'), \
                                   header=0, sep=',', index_col=None)
        self.id_prop = id_prop_data.values

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
            alpha, beta, gamma = 2. * np.pi * (np.random.random(3)-0.5)
            rot1 = [np.cos(alpha), -1*np.sin(alpha), 0,
                    np.sin(alpha), np.cos(alpha), 0,
                    0, 0, 1]
            rot1 = np.array(rot1).reshape(3,3)
            rot2 = [np.cos(beta), 0, -1*np.sin(beta),
                    0, 1, 0,
                    np.sin(beta), 0, np.cos(beta)]
            rot2 = np.array(rot2).reshape(3,3)
            rot3 = [1, 0, 0,
                    0, np.cos(gamma), -1*np.sin(gamma),
                    0, np.sin(gamma), np.cos(gamma)]
            rot3 = np.array(rot3).reshape(3,3)
            point_cloud[:,:-1] = np.dot(point_cloud[:,:-1], rot1.T)
            point_cloud[:,:-1] = np.dot(point_cloud[:,:-1], rot2.T)
            point_cloud[:,:-1] = np.dot(point_cloud[:,:-1], rot3.T)
            if target_prop in ['cubic', 'orthorhombic', 'tetragonal']:
                assert(np.dot(point_cloud[0,:-1], point_cloud[1,:-1]) < 1E-10)
                assert(np.dot(point_cloud[1,:-1], point_cloud[2,:-1]) < 1E-10)
                assert(np.dot(point_cloud[0,:-1], point_cloud[2,:-1]) < 1E-10)

        # randomly scale all intensity by a factor
        if self.random_intensity:
            point_cloud[:,-1] *= np.random.random()
        
        # only differentiate zero and non-zero intensity values
        if self.systematic_absence:
            # cannot do both at the same time
            assert(not self.random_intensity)
            non_zeros = point_cloud[point_cloud[:,-1]>1E-8]
            non_zeros[:,-1] = 1.0
            zeros = point_cloud[point_cloud[:,-1]<=1E-8]
            zeros[:,-1] = 0.0
            point_cloud = np.concatenate((zeros, non_zeros), axis=0)
            assert((point_cloud[np.where(point_cloud[:,-1]==1.0)].shape[0] + \
                    point_cloud[np.where(point_cloud[:,-1]==0.0)].shape[0]) == point_cloud.shape[0])

        # randomly permute all points except the origin
        if self.permutation:
            np.random.shuffle(point_cloud[1:])

        if self.point_dim != 4:
            assert(self.point_dim == 3)
            point_cloud = point_cloud[:,:-1]

        point_cloud_feat = torch.Tensor(point_cloud.transpose())

        # 6-class crystal family
        if self.target == 'crystal_family':
            cryst_family_dict = {
                'cubic': 0, 'orthorhombic': 1, 'tetragonal': 2,
                'monoclinic': 3, 'triclinic': 4,
                'hexagonal': 5, 'trigonal': 5 
            }
            prop = torch.Tensor([cryst_family_dict[target_prop]])
        # 7-class crystal system
        elif self.target == 'crystal_system':
            cryst_sys_dict = {
                'cubic': 0, 'orthorhombic': 1, 'tetragonal': 2,
                'monoclinic': 3, 'triclinic': 4,
                'hexagonal': 5, 'trigonal': 6
            }
            prop = torch.Tensor([cryst_sys_dict[target_prop]])
        elif self.target == 'THC':
            assert(target_prop in ['hexagonal', 'trigonal'])
            prop = torch.Tensor([target_prop == 'hexagonal'])
        # binary metal-insulator classification
        elif self.target == 'MIC':
            prop = torch.Tensor([target_prop>1E-6])
        # binary bulk modulus
        elif self.target == 'bulk_modulus':
            target_prop = ast.literal_eval(target_prop)
            bulk_mod = target_prop[1]
            criterion = bulk_mod >= 100. 
            prop = torch.Tensor([criterion])
        # binary shear modulus
        elif self.target == 'shear_modulus':
            target_prop = ast.literal_eval(target_prop)
            shear_mod = target_prop[0]
            criterion = shear_mod >= 50. 
            prop = torch.Tensor([criterion])
        # binary Poisson ratio
        elif self.target == 'poisson_ratio':
            target_prop = ast.literal_eval(target_prop)
            poisson_ratio = target_prop[2]
            criterion = poisson_ratio >= 0.3
            prop = torch.Tensor([criterion])
        # binary stability
        elif self.target == 'stability':
            prop = torch.Tensor([target_prop<0.01])
        # binary topologically trivial vs. non-trivial
        elif self.target == 'TIC2':
            assert(target_prop in ['trivial*', 'TI*', 'SM*'])
            prop = torch.Tensor([target_prop=='trivial*'])
        # ternary topological classification
        elif self.target == 'TIC3':
            topo_dict = {'trivial*': 0, 'TI*': 1, 'SM*': 2}
            prop = torch.Tensor([topo_dict[target_prop]])
        else:
            raise NotImplementedError

        return point_cloud_feat, prop.long(), material_id

    def __len__(self):
        return self.id_prop.shape[0]
