import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class deepKNetDataset(Dataset):
    def __init__(self, root_dir, target='band_gap'):
        self.root_dir = root_dir
        self.target = target
        self.data_files = sorted(os.listdir(os.path.join(self.root_dir, 'target')),
                                key=lambda x:int(x[3:].split('.')[0]))

    def __getitem__(self, idx):
        point_cloud = pd.read_csv(self.root_dir+'/features/'+self.data_files[idx], sep=';', header=None, index_col=None)
        point_cloud = torch.Tensor(point_cloud.values)
        properties = pd.read_csv(self.root_dir+'/target/'+self.data_files[idx], sep=';', header=0, index_col=None)
        prop = torch.Tensor(properties[self.target].values)
        return point_cloud, prop

    def __len__(self):
        return len(self.data_files)


