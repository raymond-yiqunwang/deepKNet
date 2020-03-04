import torch
import torch.nn as nn
import torch.nn.functional as F

class deepKNet(nn.Module):
    def __init__(self):
        super(deepKNet, self).__init__()
        # layer 0
        self.conv0 = nn.Conv1d(98, 128, 1)
        self.bn0 = nn.BatchNorm1d(128)
        
        # wave1
        self.conv1 = nn.Conv1d(128, 256, 3)
        self.bn1 = nn.BatchNorm1d(256)
        self.nwave1 = 5
        self.wave1_convs = nn.ModuleList([
            nn.Conv1d(256, 256, 3, dilation=2**i, padding=2**i) for i in range(self.nwave1)])
        self.wave1_bns = nn.ModuleList([
            nn.BatchNorm1d(256) for i in range(self.nwave1)])
        
        # wave2
        self.conv2 = nn.Conv1d(256, 512, 3)
        self.bn2 = nn.BatchNorm1d(512)
        self.nwave2 = 5
        self.wave2_convs = nn.ModuleList([
            nn.Conv1d(512, 512, 3, dilation=2**i, padding=2**i) for i in range(self.nwave2)])
        self.wave2_bns = nn.ModuleList([
            nn.BatchNorm1d(512) for i in range(self.nwave2)])
        
        # wave3
        self.conv3 = nn.Conv1d(512, 1024, 3)
        self.bn3 = nn.BatchNorm1d(1024)
        self.nwave3 = 5
        self.wave3_convs = nn.ModuleList([
            nn.Conv1d(1024, 1024, 3, dilation=2**i, padding=2**i) for i in range(self.nwave3)])
        self.wave3_bns = nn.ModuleList([
            nn.BatchNorm1d(1024) for i in range(self.nwave3)])
        
        # fc1
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        # fc2 
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        # fc3 
        self.fc3 = nn.Linear(256, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        
        # fc4
        self.fc4 = nn.Linear(64, 1)

    def forward(self, point_cloud):
        # point_cloud size -- (batch_size, nfeatures, npoints)
        # current settings -- (        16,    3+1+94,    npts)

        net = F.relu(self.bn0(self.conv0(point_cloud)))
        #net = F.relu((self.conv0(point_cloud)))

        # wave1
        net = F.relu(self.bn1(self.conv1(net)))
        #net = F.relu((self.conv1(net)))
        for i in range(self.nwave1):
            intmp = net
            net = self.wave1_bns[i](self.wave1_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)

        # wave2
        net = F.relu(self.bn2(self.conv2(net)))
        #net = F.relu((self.conv2(net)))
        for i in range(self.nwave2):
            intmp = net
            net = self.wave2_bns[i](self.wave2_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)

        # wave3
        net = F.relu(self.bn3(self.conv3(net)))
        #net = F.relu((self.conv3(net)))
        for i in range(self.nwave3):
            intmp = net
            net = self.wave3_bns[i](self.wave3_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)
        
        # max pooling
        net = torch.max(net, dim=2, keepdim=True)[0]

        # fully-connected layers
        net = net.view(-1, 1024) # reshape tensor
        net = F.relu(self.bn_fc1(self.fc1(net)))
        net = F.relu(self.bn_fc2(self.fc2(net)))
        net = F.relu(self.bn_fc3(self.fc3(net)))
        #net = F.relu((self.fc1(net)))
        #net = F.relu((self.fc2(net)))
        #net = F.relu((self.fc3(net)))
        y_pred = self.fc4(net)
        
        return y_pred
 

