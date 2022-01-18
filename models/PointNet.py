import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, conv_filters, fc_dims, dropout):
        super(PointNet, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        conv_filters.insert(0, 4) # initial data point dim
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_filters[i], conv_filters[i+1], 1) \
                                          for i in range(len(conv_filters)-1)])

        self.conv_bn_layers = nn.ModuleList([nn.BatchNorm1d(conv_filters[i]) \
                                             for i in range(1, len(conv_filters))])
        
        self.fc_layers = nn.ModuleList([nn.Linear(fc_dims[i], fc_dims[i+1]) \
                                        for i in range(len(fc_dims)-1)])
        
        self.fc_bn_layers = nn.ModuleList([nn.BatchNorm1d(fc_dims[i]) \
                                           for i in range(1, len(fc_dims))])
        
        self.fc_out = nn.Linear(fc_dims[-1], 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, net):
        # conv
        for idx in range(len(self.conv_layers)):
            net = F.relu(self.conv_bn_layers[idx](self.conv_layers[idx](net)))

        # critical points
#        critical_points = np.argmax(net.detach().cpu().numpy(), axis=2)
#        np.save("critical_points.npy", critical_points)
        
        # max pooling
        net = torch.max(net, dim=2)[0]

        # FC
        for idx in range(len(self.fc_layers)):
            net = F.relu(self.fc_bn_layers[idx](self.dropout(self.fc_layers[idx](net))))
        
        out = self.logsoftmax(self.fc_out(net))

        return out


