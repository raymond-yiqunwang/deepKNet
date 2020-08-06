import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_train_val_test_loader(root, target, cutoff, padding, data_aug, rot_all,
                              batch_size, num_data_workers, pin_memory):

    train_dataset = deepKNetDataset(root=root+'/train/', target=target,
                                    cutoff=cutoff, padding=padding,
                                    data_aug=(data_aug=='True' or
                                              rot_all=='True'))
    val_dataset = deepKNetDataset(root=root+'/valid/', target=target,
                                  cutoff=cutoff, padding=padding,
                                  data_aug=(data_aug=='True' and
                                            rot_all=='True'))
    test_dataset = deepKNetDataset(root=root+'/test/', target=target,
                                   cutoff=cutoff, padding=padding,
                                   data_aug=(data_aug=='True' and
                                             rot_all=='True'))
    # init DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_data_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_data_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_data_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


class deepKNetDataset(Dataset):
    def __init__(self, root, target, cutoff, padding, data_aug):
        self.root = root
        self.target = target
        self.cutoff = cutoff
        self.padding = padding
        self.data_aug = data_aug
        self.file_names = [fname.split('.')[0] for fname in \
                           os.listdir(self.root)
                           if fname.split('.')[-1] == 'csv']

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
            np.random.seed(8)
            alpha, beta, gamma = np.pi * np.random.random(3)
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
        
        point_cloud = torch.Tensor(point_cloud.transpose())

        # load target property
        properties = pd.read_csv(self.root+self.file_names[idx]+'.csv',
                                 sep=';', header=0, index_col=None)
        prop = torch.Tensor(properties[self.target].values)
        return point_cloud, prop

    def __len__(self):
        return len(self.file_names)


