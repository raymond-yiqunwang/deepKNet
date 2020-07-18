import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, classification=True):
        super(LeNet, self).__init__()
        self.classification = classification
        self.conv1 = nn.Conv2d(21, 32, kernel_size=5,stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.pooling1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pooling2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5,stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1   = nn.Linear(81, 32)
        self.bn4 = nn.BatchNorm1d(32)

        if self.classification:
            self.fc_out   = nn.Linear(32, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.fc_out = nn.Linear(32, 1)
            

    def forward(self, image):
        # image size -- (BS, C, H, W)
        net = F.relu(self.bn1(self.conv1(image)))
        net = self.pooling1(net)
        
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.pooling2(net)

        net = F.relu(self.bn3(self.conv3(net)))
        
        # pooling multi-views
        net = torch.max(net, dim=1, keepdim=True)[0]
        #net = torch.mean(net, dim=1, keepdim=True)[0]
        if self.classification:
            net = self.dropout(net)
        
        net = net.view(-1, self.num_flat_features(net))

        net = F.relu(self.bn4(self.fc1(net)))

        out = self.fc_out(net)
        if self.classification:
            out = self.logsoftmax(out)

        return out

    def num_flat_features(self, net):
        num_features = 1
        for s in net.size()[1:]:
            num_features *= s
        return num_features


