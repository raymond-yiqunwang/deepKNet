import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5,stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1   = nn.Linear(16*14*14, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 2)

    def forward(self, image):
        # image size -- (BS, C, H, W)
        net = F.max_pool2d(F.relu(self.conv1(image)), kernel_size=2)
        net = F.max_pool2d(F.relu(self.conv2(net)), kernel_size=2)
        net = net.view(-1, self.num_flat_features(net))
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        y_pred = self.fc3(net)
        return y_pred

    def num_flat_features(self, x):
        num_features = 1
        for s in x.size()[1:]:
            num_features *= s
        return num_features


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.fc1 = nn.Linear(16384, 1024)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, image):
        net = F.relu(self.fc1(image))
        y_pred = self.fc2(net)
        return y_pred


class waveNet(nn.Module):
    def __init__(self):
        super(waveNet, self).__init__()
        # layer 0
#        self.conv0 = nn.Conv1d(98, 128, 1)
        self.conv0 = nn.Conv1d(4, 128, 3)
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

        net = F.relu(self.bn0(self.conv0(point_cloud)))

        # wave1
        net = F.relu(self.bn1(self.conv1(net)))
        """
        for i in range(self.nwave1):
            intmp = net
            net = self.wave1_bns[i](self.wave1_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)
        """

        # wave2
        net = F.relu(self.bn2(self.conv2(net)))
        """
        for i in range(self.nwave2):
            intmp = net
            net = self.wave2_bns[i](self.wave2_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)
        """

        # wave3
        net = F.relu(self.bn3(self.conv3(net)))
        """
        for i in range(self.nwave3):
            intmp = net
            net = self.wave3_bns[i](self.wave3_convs[i](net))
            net += intmp # residue connection
            net = F.relu(net)
        """
        
        # max pooling
        net = torch.max(net, dim=2, keepdim=True)[0]

        # fully-connected layers
        net = net.view(-1, 1024) # reshape tensor
        net = F.relu(self.bn_fc1(self.fc1(net)))
        net = F.relu(self.bn_fc2(self.fc2(net)))
        net = F.relu(self.bn_fc3(self.fc3(net)))
        y_pred = self.fc4(net)
        
        return y_pred


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


