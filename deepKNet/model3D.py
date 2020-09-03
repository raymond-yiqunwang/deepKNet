import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 3*3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        iden = torch.Tensor(torch.from_numpy(np.eye(3).flatten() \
                                 .astype(np.float32))) \
                                 .view(1, 3*3).repeat(batchsize,1)
        if x.is_cuda:
            cuda_device = 'cuda:{}'.format(x.get_device())
            iden = iden.cuda(device=cuda_device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNet(nn.Module):
    def __init__(self, nclass, conv_dims, nbert, fc_dims, pool, dropout, stn):
        super(PointNet, self).__init__()
        self.nclass = nclass
        self.embed_dim = conv_dims[-1]
        self.nbert = nbert
        self.pool = pool
        self.dp = dropout
        if stn:
            print('\n STN3d \n')
            self.stn3d = STN3d()
        else:
            self.stn3d = None

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
        # STN3d
        if self.stn3d:
            assert(x.size(1) == 3)
            trans = self.stn3d(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

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


