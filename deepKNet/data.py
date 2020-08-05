import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_val_test_loader(dataset, train_ratio, val_ratio, test_ratio,
                              batch_size, num_data_workers, pin_memory,
                              collate_fn=default_collate):
    # train-val split
    total_size = len(dataset)
    indices = list(range(total_size))
    train_split = int(np.floor(total_size * train_ratio))
    train_sampler = SubsetRandomSampler(indices[:train_split])
    val_split = train_split + int(np.floor(total_size * val_ratio))
    val_sampler = SubsetRandomSampler(indices[train_split:val_split])
    test_sampler = SubsetRandomSampler(indices[val_split:])
    # init DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                              sampler=train_sampler, num_workers=num_data_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            sampler=val_sampler, num_workers=num_data_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                             sampler=test_sampler, num_workers=num_data_workers,
                             pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


class deepKNetDataset(Dataset):
    def __init__(self, root, target, train_ratio,
                 cutoff, padding, data_aug, rot_all):
        self.root = root
        self.target = target
        self.cutoff = cutoff
        self.padding = padding
        self.data_aug = data_aug
        self.rot_all = rot_all
        self.file_names = [fname.split('.')[0] for fname in \
                           os.listdir(os.path.join(self.root, 'target/'))]
        random.seed(5)
        random.shuffle(self.file_names)
        total_size = len(self.file_names)
        self.train_split = int(np.floor(total_size * train_ratio))

    def __getitem__(self, idx):
        # load point cloud data
        point_cloud = np.load(self.root+'/features/'+self.file_names[idx]+'.npy')
        
        # padding and cutoff
        if self.padding == 'zero':
            if point_cloud.shape[0] < self.cutoff:
                point_cloud = np.pad(point_cloud, ((0, self.cutoff-point_cloud.shape[0]), (0, 0)))
            else:
                point_cloud = point_cloud[:self.cutoff, :]
        elif self.padding == 'periodic':
            while point_cloud.shape[0] < self.cutoff:
                point_cloud = np.repeat(point_cloud, 2, axis=0)
            point_cloud = point_cloud[:self.cutoff, :]
        else:
            raise NotImplementedError

        # apply random 3D rotation for data augmentation
        if self.data_aug and (idx < self.train_split or self.rot_all):
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
        properties = pd.read_csv(self.root+'/target/'+self.file_names[idx]+'.csv',
                                 sep=';', header=0, index_col=None)
        prop = torch.Tensor(properties[self.target].values)
        return point_cloud, prop

    def __len__(self):
        return len(self.file_names)


