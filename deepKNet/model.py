import torch
import torch.nn as nn
import torch.nn.functional as F

class deepKNet(nn.Module):
    def __init__(self):
        super(deepKNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(123, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 1)

    def forward(self, point_cloud):
        # point_cloud size -- (batch_size, npoint, nfeature)
        point_cloud = point_cloud.transpose(2, 1)
        out = F.relu(self.bn1(self.conv1(point_cloud)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.mean(out, dim=2)
        out = out.view(-1, 1024)
        out = self.fc1(out)
        return out
 

