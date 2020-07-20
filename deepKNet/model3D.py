import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STNkd(nn.Module):
    def __init__(self, k=4):
        super(STNkd, self).__init__()
        self.k = k
        
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.Tensor(torch.from_numpy(np.eye(self.k).flatten() \
                                               .astype(np.float32))) \
                           .view(1, self.k*self.k) \
                           .repeat(batchsize,1)
        if x.is_cuda:
            cuda_device = 'cuda:{}'.format(x.get_device())
            iden = iden.cuda(device=cuda_device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, k=4):
        super(PointNetfeat, self).__init__()
        self.k = k
        self.stn = STNkd(k=self.k)
        self.conv1 = torch.nn.Conv1d(self.k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        self.k = k
        self.feat = PointNetfeat(k=self.k)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        out = self.logsoftmax(self.fc3(x))
        return out


"""
class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.MultiheadAttention(128, 8)
        self.norm1 = nn.LayerNorm(128)

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        y = self.norm1(x + y)
        z = torch.relu(self.fc1(y))
        z = torch.relu(self.fc2(z))
        return self.norm2(y + z)


class BertNet(nn.Module):
    def __init__(self):
        super().__init__()

#        self.nbert = 12
        self.nbert = 5
#        self.prenet = nn.Linear(98, 128)
        self.prenet = nn.Linear(4, 128)
        self.prenorm = nn.LayerNorm(128)

        self.bert = nn.ModuleList([BertLayer() for _ in range(self.nbert)])

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(2, 0, 1) # L x BS x NF
        x = self.prenet(x)
        x = self.prenorm(x)

        for bert in self.bert:
            x = bert(x)

        # L x BS x NF -> BS x NF
        # Average pooling
        x = x.mean(0)
        return self.fc(x)
"""

if __name__ == '__main__':
    sim_data = torch.Tensor(torch.rand(32, 4, 2500))
    k = 4
    trans = STNkd(k=k)
    out = trans(sim_data)
    print(out.shape)

    cls = PointNet(k=k)
    out = cls(sim_data)
    print(out.shape)
