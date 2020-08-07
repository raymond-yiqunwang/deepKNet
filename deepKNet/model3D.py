import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, k, nclass, conv_dims, fc_dims, nbert, pool, dp):
        super(PointNet, self).__init__()
        self.k = k
        self.nclass = nclass
        self.embed_dim = conv_dims[-1]
        self.nbert = nbert
        self.pool = pool
        self.dp = dp

        assert(conv_dims[0] == 4 and len(conv_dims) > 1)
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_dims[i], conv_dims[i+1], 1) \
                                          for i in range(len(conv_dims)-1)])
        self.conv_bn_layers = nn.ModuleList([nn.BatchNorm1d(conv_dims[i]) \
                                             for i in range(1, len(conv_dims))])

        self.bert = nn.ModuleList([BertLayer(self.embed_dim) \
                                   for _ in range(self.nbert)])

        self.fc_layers = nn.ModuleList([nn.Linear(fc_dims[i], fc_dims[i+1]) \
                                        for i in range(len(fc_dims)-1)])
        
        self.fc_bn_layers = nn.ModuleList([nn.BatchNorm1d(fc_dims[i]) \
                                           for i in range(1, len(fc_dims))])
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(self.dp)
        self.fc_out = nn.Linear(fc_dims[-1], self.nclass)

    def forward(self, x):
        # PointNet
        for idx in range(len(self.conv_layers)):
            x = F.relu(self.conv_bn_layers[idx](self.conv_layers[idx](x)))
        
        # self-attention
        x = x.permute(2, 0, 1)
        for bert in self.bert:
            x = bert(x)
        x = x.permute(1, 2, 0)

        # pooling
        if self.pool == 'CLS':
            x = x[:, :, 0]
        elif self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        else:
            raise NotImplementedError
        x = x.view(-1, self.embed_dim)
        
        # FC
        for idx in range(len(self.fc_layers)):
            x = F.relu(self.fc_bn_layers[idx](self.fc_layers[idx](x)))
            if idx == 0:
                x = self.dropout(x)
        
        out = self.logsoftmax(self.fc_out(x))
        
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
        z = F.relu(self.fc1(y))
        z = F.relu(self.fc2(z))
        return self.norm2(y + z)


