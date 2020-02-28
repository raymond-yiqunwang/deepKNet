import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, batch_size=32, train_size=0.9, val_size=0.1,
                              num_workers=1, pin_memory=False):
    # train-val split
    total_size = len(dataset)
    indices = list(range(total_size))
    split = int(np.floor(total_size * train_size))
    train_sampler = SubsetRandomSampler(indices[:split])
    val_sampler = SubsetRandomSampler(indices[split:])
    # init DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


class deepKNetDataset(Dataset):
    def __init__(self, root, target='band_gap'):
        self.root = root
        self.target = target
        self.file_names = list(sorted(os.listdir(os.path.join(self.root, 'target')),
                                 key=lambda x:int(x[3:].split('.')[0])))

    def __getitem__(self, idx):
        # load point_cloud
        point_cloud = pd.read_csv(self.root+'/features/'+self.file_names[idx],
                                  sep=';', header=None, index_col=None)
        point_cloud = torch.Tensor(point_cloud.values)
        # load target property
        properties = pd.read_csv(self.root+'/target/'+self.file_names[idx],
                             sep=';', header=0, index_col=None)
        prop = torch.Tensor(properties[self.target].values)
        return point_cloud, prop

    def __len__(self):
        return len(self.file_names)


