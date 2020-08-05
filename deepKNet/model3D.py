import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, k, dp, attn, nbert, embed_dim, classification, pool):
        super(PointNet, self).__init__()
        self.k = k
        self.dp = dp
        self.attn = attn
        self.nbert = nbert
        self.embed_dim = embed_dim
        self.classification = classification
        self.pool = pool

        self.conv1 = torch.nn.Conv1d(self.k, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, self.embed_dim, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)

        if self.attn:
            self.bert = nn.ModuleList([BertLayer(self.embed_dim) for _ in range(self.nbert)])

        self.fc1 = nn.Linear(self.embed_dim, 256)
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.attn:
            x = x.permute(2, 0, 1)
            for bert in self.bert:
                x = bert(x)
            x = x.permute(1, 2, 0)

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'CLS':
            x = x[:, :, 0]
        else:
            raise NotImplementedError
        x = x.view(-1, self.embed_dim)
        
        x = F.relu(self.bn4(self.fc1(x)))
        if self.classification:
            x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        out = self.fc_out(x)
        if self.classification:
            out = self.logsoftmax(out)
        return out


class BertLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.attn = nn.MultiheadAttention(self.embed_dim, 2)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim/2))
        self.fc2 = nn.Linear(int(self.embed_dim/2), self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        y = self.norm1(x + y)
        z = torch.relu(self.fc1(y))
        z = torch.relu(self.fc2(z))
        return self.norm2(y + z)


