import os
import torch
import pandas as pd
from torch.utils.data import Dataset

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


