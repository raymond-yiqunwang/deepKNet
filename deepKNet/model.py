import torch
import numpy as np
import torch.nn as nn

class deepKNet(nn.Module):
    def __init__(self):
        super(deepKNet, self).__init__()
        self.fc1 = nn.Linear(10, 1)

    def forward(self, point_cloud):
        inp = torch.tensor([np.float(0.1)]*10)
        out = self.fc1(inp)
        return out
 

