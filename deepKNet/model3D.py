import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, k=3):
        super(STN3d, self).__init__()
        self.k = k
        
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        iden = torch.Tensor(torch.from_numpy(np.eye(self.k).flatten() \
                                               .astype(np.float32))) \
                           .view(1, self.k*self.k) \
                           .repeat(batchsize,1)
        if x.is_cuda:
            cuda_device = 'cuda:{}'.format(x.get_device())
            iden = iden.cuda(device=cuda_device)
        x += iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, k=4, dp=0.1, stn=False, attn=True, classification=True):
        super(PointNet, self).__init__()
        self.k = k
        self.dp = dp
        self.stn = stn
        self.attn = attn
        self.classification = classification

        if self.stn:
            self.stn3d = STN3d()
        self.conv1 = torch.nn.Conv1d(self.k, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        if self.attn:
            self.nbert = 5
            self.bert = nn.ModuleList([BertLayer() for _ in range(self.nbert)])

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(self.dp)
            self.fc_out = nn.Linear(128, 2)
        else:
            self.fc_out = nn.Linear(128, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        if self.stn:
            x1 = x[:,:3,:]
            x2 = x[:,-1:,:]
            trans = self.stn3d(x1)
            x1 = x1.transpose(2, 1)
            x1 = torch.bmm(x1, trans)
            x1 = x1.transpose(2, 1)
            x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.attn:
            x = x.permute(2, 0, 1)
            for bert in self.bert:
                x = bert(x)
            x = x.permute(1, 2, 0)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        
        x = F.relu(self.bn4(self.fc1(x)))
        if self.classification:
            x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        out = self.fc_out(x)
        if self.classification:
            out = self.logsoftmax(out)
        return out


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.MultiheadAttention(512, 2)
        self.norm1 = nn.LayerNorm(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        y = self.norm1(x + y)
        z = torch.relu(self.fc1(y))
        z = torch.relu(self.fc2(z))
        return self.norm2(y + z)


